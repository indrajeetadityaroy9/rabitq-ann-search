#pragma once

#include "../core/types.hpp"
#include "../core/codes.hpp"
#include "../core/memory.hpp"
#include "../encoder/rabitq_encoder.hpp"
#include "../distance/fastscan_kernel.hpp"
#include "../graph/rabitq_graph.hpp"
#include "../graph/graph_refinement.hpp"
#include "../graph/neighbor_selection.hpp"
#include "../search/rabitq_search.hpp"
#include "params.hpp"
#include <vector>
#include <random>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <type_traits>
#include <unordered_map>

#ifdef _OPENMP
#include <omp.h>
#else
inline int omp_get_thread_num() { return 0; }
inline int omp_get_max_threads() { return 1; }
#endif

namespace cphnsw {

// HNSW hierarchical multi-layer index.
//
// Upper layers (1..max_level): sparse graphs with simple adjacency lists,
// exact distance computation, and greedy 1-NN search for routing.
//
// Layer 0: RaBitQ FastScan graph with colocated memory layout and
// Vamana-style refinement. Uses beam search with quantized distance
// estimation for high-throughput retrieval.
//
// Layer assignment follows geometric distribution: P(l >= L) = exp(-L * ln(M)).
// This gives O(log n) expected layers and O(log n) search complexity.
template <size_t D, size_t R = 32, size_t BitWidth = 1, typename RotationPolicy = RandomHadamardRotation>
class HNSWIndex {
public:
    using CodeType = std::conditional_t<BitWidth == 1,
        RaBitQCode<D>, NbitRaBitQCode<D, BitWidth>>;
    using QueryType = RaBitQQuery<D>;
    using Encoder = std::conditional_t<BitWidth == 1,
        RaBitQEncoder<D, RotationPolicy>,
        NbitRaBitQEncoder<D, BitWidth, RotationPolicy>>;
    using Graph = RaBitQGraph<D, R, BitWidth>;
    using Engine = RaBitQSearchEngine<D, R, BitWidth>;
    using Refinement = GraphRefinement<D, R, BitWidth>;

    static constexpr size_t DIMS = D;
    static constexpr size_t DEGREE = R;
    static constexpr size_t BIT_WIDTH = BitWidth;

    // Upper-layer neighbor list per node per layer.
    // M_upper neighbors max, stored as simple adjacency list.
    static constexpr size_t M_UPPER = R / 2;

    explicit HNSWIndex(const IndexParams& params)
        : params_(params)
        , encoder_(params.dim, params.seed)
        , graph_(params.dim, params.initial_capacity)
        , mL_(1.0 / std::log(static_cast<double>(params.M)))
        , rng_(params.seed)
    {
        if (params.dim == 0) throw std::invalid_argument("dim must be > 0");
    }

    explicit HNSWIndex(size_t dim)
        : HNSWIndex(IndexParams().set_dim(dim)) {}

    // Add vectors in batch (deferred build — call finalize() after).
    void add_batch(const float* vecs, size_t num_vecs,
                   const BuildParams& build_params = BuildParams()) {
        if (num_vecs == 0) return;

        graph_.reserve(graph_.size() + num_vecs);

        std::vector<CodeType> codes(num_vecs);
        encoder_.encode_batch(vecs, num_vecs, codes.data());

        for (size_t i = 0; i < num_vecs; ++i) {
            graph_.add_node(codes[i], vecs + i * params_.dim);
        }

        needs_build_ = true;

        if (build_params.verbose) {
            printf("[HNSWIndex] Added %zu vectors (B=%zu). Call finalize() to build.\n",
                   num_vecs, BitWidth);
        }
    }

    // Build HNSW structure: assign layers, build upper layers via incremental
    // insertion, then refine layer 0 with Vamana-style optimization.
    void finalize(size_t num_threads = 0, bool verbose = false) {
        size_t n = graph_.size();
        if (n == 0 || !needs_build_) { finalized_ = true; return; }

        // 1. Assign layers to all nodes
        if (verbose) printf("[HNSW] Assigning layers to %zu nodes...\n", n);
        assign_layers(n);

        // 2. Build upper layers (sequential incremental insertion)
        if (verbose) printf("[HNSW] Building %d upper layers...\n", max_level_);
        build_upper_layers(verbose);

        // 3. Build and refine layer 0 using existing Vamana pipeline
        if (verbose) printf("[HNSW] Optimizing layer 0 (Vamana refinement)...\n");
        Refinement::optimize_graph(graph_, encoder_, params_.ef_construction, num_threads, verbose);

        needs_build_ = false;
        finalized_ = true;
        if (verbose) printf("[HNSW] Build complete. max_level=%d, entry=%u\n",
                           max_level_, entry_point_);
    }

    // Multi-layer search: greedy descent through upper layers, then
    // FastScan beam search at layer 0.
    std::vector<SearchResult> search(
        const float* query,
        const SearchParams& params = SearchParams()) const
    {
        if (graph_.empty()) return {};

        QueryType encoded = encoder_.encode_query(query);
        encoded.error_epsilon = params.error_epsilon;

        // If no upper layers, fall back to flat search
        if (max_level_ <= 0 || entry_point_ == INVALID_NODE) {
            return Engine::search(encoded, query, graph_, params.ef, params.k);
        }

        // Greedy descent through upper layers to find best entry point for layer 0
        NodeId ep = entry_point_;
        for (int level = max_level_; level >= 1; --level) {
            ep = greedy_search_layer(query, ep, level);
        }

        // Layer 0: full FastScan beam search from the routed entry point
        return Engine::search_from(encoded, query, graph_, ep, params.ef, params.k);
    }

    std::vector<SearchResult> search(const float* query, size_t k) const {
        return search(query, SearchParams().set_k(k));
    }

    size_t size() const { return graph_.size(); }
    bool empty() const { return graph_.empty(); }
    size_t dim() const { return params_.dim; }
    bool is_finalized() const { return finalized_; }
    int max_level() const { return max_level_; }
    const IndexParams& params() const { return params_; }
    const Graph& graph() const { return graph_; }
    const Encoder& encoder() const { return encoder_; }

    struct Stats {
        size_t num_nodes;
        float avg_degree;
        size_t max_degree;
        size_t isolated_nodes;
        int max_level;
        size_t nodes_above_layer0;
    };

    Stats get_stats() const {
        size_t above = 0;
        for (size_t i = 0; i < node_levels_.size(); ++i) {
            if (node_levels_[i] > 0) ++above;
        }
        return Stats{
            graph_.size(),
            graph_.average_degree(),
            graph_.max_degree(),
            graph_.count_isolated(),
            max_level_,
            above
        };
    }

private:
    IndexParams params_;
    Encoder encoder_;
    Graph graph_;
    bool finalized_ = false;
    bool needs_build_ = false;

    // HNSW layer structure
    double mL_;  // 1/ln(M), controls layer probability
    std::mt19937_64 rng_;

    int max_level_ = 0;
    NodeId entry_point_ = INVALID_NODE;

    // Per-node assigned level (0 = base layer only)
    std::vector<int> node_levels_;

    // Upper-layer adjacency lists.
    // upper_neighbors_[level-1][node] = vector of neighbor NodeIds at that level.
    // Only allocated for nodes with level >= that layer.
    struct UpperLayerNode {
        std::vector<NodeId> neighbors;
    };
    std::vector<std::unordered_map<NodeId, UpperLayerNode>> upper_layers_;  // [level-1][node_id]

    // Assign a random level to each node: floor(-log(uniform) * mL_)
    void assign_layers(size_t n) {
        node_levels_.resize(n);
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        max_level_ = 0;
        entry_point_ = INVALID_NODE;

        for (size_t i = 0; i < n; ++i) {
            double r = dist(rng_);
            // Avoid log(0)
            if (r < 1e-15) r = 1e-15;
            int level = static_cast<int>(-std::log(r) * mL_);
            node_levels_[i] = level;
            if (level > max_level_) {
                max_level_ = level;
                entry_point_ = static_cast<NodeId>(i);
            }
        }

        // Allocate upper-layer structures (sparse maps, no per-node allocation)
        upper_layers_.resize(max_level_);
    }

    // Build upper layers via incremental HNSW insertion.
    // For each node at level >= 1, greedily search to find nearest neighbors
    // at each layer, then add bidirectional edges with pruning.
    void build_upper_layers(bool verbose) {
        size_t n = graph_.size();

        // Insertion order: process nodes with highest levels first to ensure
        // the entry point and high-level nodes are connected early.
        std::vector<NodeId> insertion_order(n);
        for (size_t i = 0; i < n; ++i) insertion_order[i] = static_cast<NodeId>(i);
        std::sort(insertion_order.begin(), insertion_order.end(),
                  [this](NodeId a, NodeId b) { return node_levels_[a] > node_levels_[b]; });

        for (size_t idx = 0; idx < n; ++idx) {
            NodeId node = insertion_order[idx];
            int node_level = node_levels_[node];
            if (node_level == 0) break;  // All remaining nodes are layer-0 only

            // Find entry point: greedy descent from global entry through layers
            // above this node's level
            NodeId ep = entry_point_;
            for (int level = max_level_; level > node_level; --level) {
                ep = greedy_search_layer(graph_.get_vector(node), ep, level);
            }

            // Insert at each layer from node_level down to 1
            for (int level = std::min(node_level, max_level_); level >= 1; --level) {
                // Search for ef_construction nearest neighbors at this layer
                auto candidates = search_upper_layer(
                    graph_.get_vector(node), ep, level,
                    std::min<size_t>(params_.ef_construction, 64));

                // Select M_UPPER neighbors using diversity-aware heuristic
                auto dist_fn = [this](NodeId a, NodeId b) {
                    return exact_distance(graph_.get_vector(a), graph_.get_vector(b));
                };
                auto selected = select_neighbors_heuristic(std::move(candidates), M_UPPER, dist_fn);

                // Set forward edges: node -> selected neighbors
                auto& node_neighbors = upper_layers_[level - 1][node].neighbors;
                node_neighbors.clear();
                node_neighbors.reserve(selected.size());
                for (const auto& s : selected) {
                    node_neighbors.push_back(s.id);
                }

                // Set reverse edges: each neighbor -> node (with pruning)
                for (const auto& s : selected) {
                    auto& nb = upper_layers_[level - 1][s.id].neighbors;
                    nb.push_back(node);
                    if (nb.size() > M_UPPER) {
                        prune_upper_neighbors(s.id, level);
                    }
                }

                // Use closest neighbor as entry point for next lower layer
                if (!selected.empty()) {
                    ep = selected[0].id;
                }
            }
        }

        if (verbose) {
            for (int l = max_level_; l >= 1; --l) {
                const auto& layer = upper_layers_[l - 1];
                size_t count = layer.size();
                size_t edges = 0;
                for (const auto& [node_id, node_data] : layer) {
                    edges += node_data.neighbors.size();
                }
                printf("[HNSW] Layer %d: %zu nodes, %.1f avg degree\n",
                       l, count, count > 0 ? static_cast<float>(edges) / count : 0.0f);
            }
        }
    }

    // Greedy 1-NN search at a given upper layer. Returns the nearest node found.
    NodeId greedy_search_layer(const float* query, NodeId ep, int level) const {
        if (ep == INVALID_NODE) return INVALID_NODE;

        float best_dist = exact_distance(query, graph_.get_vector(ep));
        NodeId best_id = ep;
        bool improved = true;

        while (improved) {
            improved = false;
            auto it = upper_layers_[level - 1].find(best_id);
            if (it == upper_layers_[level - 1].end()) break;
            const auto& neighbors = it->second.neighbors;
            for (NodeId nb : neighbors) {
                float d = exact_distance(query, graph_.get_vector(nb));
                if (d < best_dist) {
                    best_dist = d;
                    best_id = nb;
                    improved = true;
                }
            }
        }

        return best_id;
    }

    // Beam search at an upper layer, returning ef nearest candidates.
    std::vector<NeighborCandidate> search_upper_layer(
        const float* query, NodeId ep, int level, size_t ef) const
    {
        if (ep == INVALID_NODE) return {};

        // Min-heap of candidates to explore
        MinHeap candidates;
        // Max-heap of current nearest
        MaxHeap nearest;

        float ep_dist = exact_distance(query, graph_.get_vector(ep));
        candidates.push({ep, ep_dist});
        nearest.push({ep, ep_dist});

        // Epoch-based visited set — avoids O(n) allocation per search
        thread_local VisitationTable visited_table(0);
        if (visited_table.capacity() < graph_.size()) {
            visited_table.resize(graph_.size() + 1024);
        }
        uint64_t qid = visited_table.new_query();
        visited_table.check_and_mark(ep, qid);

        while (!candidates.empty()) {
            auto current = candidates.top();
            candidates.pop();

            // If the closest candidate is farther than the farthest in our results, stop
            if (nearest.size() >= ef && current.distance > nearest.top().distance) {
                break;
            }

            auto it = upper_layers_[level - 1].find(current.id);
            if (it == upper_layers_[level - 1].end()) continue;
            const auto& neighbors = it->second.neighbors;
            for (NodeId nb : neighbors) {
                if (visited_table.check_and_mark(nb, qid)) continue;

                float d = exact_distance(query, graph_.get_vector(nb));

                if (nearest.size() < ef || d < nearest.top().distance) {
                    candidates.push({nb, d});
                    nearest.push({nb, d});
                    if (nearest.size() > ef) {
                        nearest.pop();
                    }
                }
            }
        }

        // Convert to NeighborCandidate vector
        std::vector<NeighborCandidate> results;
        results.reserve(nearest.size());
        while (!nearest.empty()) {
            results.push_back({nearest.top().id, nearest.top().distance});
            nearest.pop();
        }
        std::sort(results.begin(), results.end());
        return results;
    }

    // Prune an upper-layer node's neighbors to at most M_UPPER, keeping closest.
    void prune_upper_neighbors(NodeId node, int level) {
        auto& nb = upper_layers_[level - 1][node].neighbors;
        if (nb.size() <= M_UPPER) return;

        const float* vec = graph_.get_vector(node);
        std::vector<NeighborCandidate> candidates;
        candidates.reserve(nb.size());
        for (NodeId id : nb) {
            float d = exact_distance(vec, graph_.get_vector(id));
            candidates.push_back({id, d});
        }

        auto dist_fn = [this](NodeId a, NodeId b) {
            return exact_distance(graph_.get_vector(a), graph_.get_vector(b));
        };
        auto selected = select_neighbors_heuristic(std::move(candidates), M_UPPER, dist_fn);
        nb.clear();
        nb.reserve(selected.size());
        for (const auto& s : selected) {
            nb.push_back(s.id);
        }
    }

    static float exact_distance(const float* a, const float* b) {
        return l2_distance_simd<D>(a, b);
    }
};

// 1-bit aliases
using HNSWIndex128 = HNSWIndex<128, 32>;
using HNSWIndex256 = HNSWIndex<256, 32>;
using HNSWIndex512 = HNSWIndex<512, 32>;
using HNSWIndex1024 = HNSWIndex<1024, 32>;

// Multi-bit aliases
using HNSWIndex128_2bit = HNSWIndex<128, 32, 2>;
using HNSWIndex128_4bit = HNSWIndex<128, 32, 4>;

}  // namespace cphnsw
