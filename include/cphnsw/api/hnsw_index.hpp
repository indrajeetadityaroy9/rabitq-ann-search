#pragma once

#include "../core/types.hpp"
#include "../core/codes.hpp"
#include "../core/memory.hpp"
#include "../core/adaptive_defaults.hpp"
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

#include "../core/omp_compat.hpp"

namespace cphnsw {

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

    static constexpr size_t M_UPPER = R / 2;

    explicit HNSWIndex(const IndexParams& params)
        : params_(params)
        , encoder_(params.dim, params.seed)
        , graph_(params.dim)
        , mL_(1.0 / std::log(static_cast<double>(M_UPPER)))
        , rng_(params.seed)
    {
        if (params.dim == 0) throw std::invalid_argument("dim must be > 0");
    }

    explicit HNSWIndex(size_t dim)
        : HNSWIndex(IndexParams().set_dim(dim)) {}

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

    void finalize(const BuildParams& params) {
        size_t n = graph_.size();
        if (n == 0 || !needs_build_) { finalized_ = true; return; }

        // Resolve adaptive defaults from sentinels
        size_t ef_c = params.ef_construction > 0
            ? params.ef_construction
            : AdaptiveDefaults::ef_construction(n, R);
        float err_tol = params.error_tolerance >= 0.0f
            ? params.error_tolerance
            : AdaptiveDefaults::error_tolerance(D);
        float err_eps = params.error_epsilon > 0.0f
            ? params.error_epsilon
            : AdaptiveDefaults::ERROR_EPSILON_BUILD;

        if (params.verbose) printf("[HNSW] Assigning layers to %zu nodes...\n", n);
        assign_layers(n);

        if (params.verbose) printf("[HNSW] Building %d upper layers...\n", max_level_);
        build_upper_layers(params.verbose);

        if (params.verbose) printf("[HNSW] Optimizing layer 0 (ef_c=%zu, err_tol=%.4f, err_eps=%.3f)...\n",
                                   ef_c, err_tol, err_eps);
        Refinement::optimize_graph_adaptive(graph_, encoder_,
            ef_c, err_tol, err_eps, params.num_threads, params.verbose);

        needs_build_ = false;
        finalized_ = true;
        if (params.verbose) printf("[HNSW] Build complete. max_level=%d, entry=%u\n",
                                      max_level_, entry_point_);
    }

    void finalize() {
        finalize(BuildParams{});
    }

    std::vector<SearchResult> search(
        const float* query,
        const SearchParams& params = SearchParams()) const
    {
        if (graph_.empty()) return {};

        // Resolve adaptive defaults from sentinels
        size_t ef = params.ef > 0
            ? params.ef
            : AdaptiveDefaults::ef_search(params.k, params.recall_target);
        float eps = params.error_epsilon > 0.0f
            ? params.error_epsilon
            : AdaptiveDefaults::error_epsilon_search(params.recall_target);

        QueryType encoded = encoder_.encode_query(query);
        encoded.error_epsilon = eps;

        if (max_level_ <= 0 || entry_point_ == INVALID_NODE) {
            return Engine::search(encoded, query, graph_, ef, params.k);
        }

        NodeId ep = entry_point_;
        for (int level = max_level_; level >= 1; --level) {
            ep = greedy_search_layer(query, ep, level);
        }

        return Engine::search_from(encoded, query, graph_, ep, ef, params.k);
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

    double mL_;
    std::mt19937_64 rng_;

    int max_level_ = 0;
    NodeId entry_point_ = INVALID_NODE;

    std::vector<int> node_levels_;

    struct UpperLayerEdge {
        NodeId node;
        std::vector<NodeId> neighbors;
        bool operator<(const UpperLayerEdge& other) const { return node < other.node; }
    };
    std::vector<std::vector<UpperLayerEdge>> upper_layers_;

    UpperLayerEdge* find_edge(int level, NodeId node) {
        auto& layer = upper_layers_[level - 1];
        UpperLayerEdge target{node, {}};
        auto it = std::lower_bound(layer.begin(), layer.end(), target);
        if (it != layer.end() && it->node == node) return &(*it);
        return nullptr;
    }

    const UpperLayerEdge* find_edge(int level, NodeId node) const {
        const auto& layer = upper_layers_[level - 1];
        UpperLayerEdge target{node, {}};
        auto it = std::lower_bound(layer.begin(), layer.end(), target);
        if (it != layer.end() && it->node == node) return &(*it);
        return nullptr;
    }

    UpperLayerEdge& get_or_create_edge(int level, NodeId node) {
        auto& layer = upper_layers_[level - 1];
        UpperLayerEdge target{node, {}};
        auto it = std::lower_bound(layer.begin(), layer.end(), target);
        if (it != layer.end() && it->node == node) return *it;
        return *layer.insert(it, UpperLayerEdge{node, {}});
    }

    void assign_layers(size_t n) {
        node_levels_.resize(n);
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        max_level_ = 0;
        entry_point_ = INVALID_NODE;

        for (size_t i = 0; i < n; ++i) {
            double r = dist(rng_);
            if (r < 1e-15) r = 1e-15;
            int level = static_cast<int>(-std::log(r) * mL_);
            node_levels_[i] = level;
            if (level > max_level_) {
                max_level_ = level;
                entry_point_ = static_cast<NodeId>(i);
            }
        }

        upper_layers_.resize(max_level_);
    }

    void build_upper_layers(bool verbose) {
        size_t n = graph_.size();

        std::vector<NodeId> insertion_order(n);
        for (size_t i = 0; i < n; ++i) insertion_order[i] = static_cast<NodeId>(i);
        std::sort(insertion_order.begin(), insertion_order.end(),
                  [this](NodeId a, NodeId b) { return node_levels_[a] > node_levels_[b]; });

        for (size_t idx = 0; idx < n; ++idx) {
            NodeId node = insertion_order[idx];
            int node_level = node_levels_[node];
            if (node_level == 0) break;

            NodeId ep = entry_point_;
            for (int level = max_level_; level > node_level; --level) {
                ep = greedy_search_layer(graph_.get_vector(node), ep, level);
            }

            for (int level = std::min(node_level, max_level_); level >= 1; --level) {
                auto candidates = search_upper_layer(
                    graph_.get_vector(node), ep, level,
                    std::min<size_t>(params_.ef_construction, 64));

                auto dist_fn = [this](NodeId a, NodeId b) {
                    return exact_distance(graph_.get_vector(a), graph_.get_vector(b));
                };
                auto selected = select_neighbors_heuristic(std::move(candidates), M_UPPER, dist_fn);

                auto& node_neighbors = get_or_create_edge(level, node).neighbors;
                node_neighbors.clear();
                node_neighbors.reserve(selected.size());
                for (const auto& s : selected) {
                    node_neighbors.push_back(s.id);
                }

                for (const auto& s : selected) {
                    auto& nb = get_or_create_edge(level, s.id).neighbors;
                    nb.push_back(node);
                    if (nb.size() > M_UPPER) {
                        prune_upper_neighbors(s.id, level);
                    }
                }

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
                for (const auto& edge : layer) {
                    edges += edge.neighbors.size();
                }
                printf("[HNSW] Layer %d: %zu nodes, %.1f avg degree\n",
                       l, count, count > 0 ? static_cast<float>(edges) / count : 0.0f);
            }
        }
    }

    NodeId greedy_search_layer(const float* query, NodeId ep, int level) const {
        if (ep == INVALID_NODE) return INVALID_NODE;

        float best_dist = exact_distance(query, graph_.get_vector(ep));
        NodeId best_id = ep;
        bool improved = true;

        while (improved) {
            improved = false;
            const auto* edge = find_edge(level, best_id);
            if (!edge) break;
            const auto& neighbors = edge->neighbors;
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

    std::vector<NeighborCandidate> search_upper_layer(
        const float* query, NodeId ep, int level, size_t ef) const
    {
        if (ep == INVALID_NODE) return {};

        MinHeap candidates;
        MaxHeap nearest;

        float ep_dist = exact_distance(query, graph_.get_vector(ep));
        candidates.push({ep, ep_dist});
        nearest.push({ep, ep_dist});

        thread_local VisitationTable visited_table(0);
        if (visited_table.capacity() < graph_.size()) {
            visited_table.resize(graph_.size() + 1024);
        }
        uint64_t qid = visited_table.new_query();
        visited_table.check_and_mark(ep, qid);

        while (!candidates.empty()) {
            auto current = candidates.top();
            candidates.pop();

            if (nearest.size() >= ef && current.distance > nearest.top().distance) {
                break;
            }

            const auto* edge = find_edge(level, current.id);
            if (!edge) continue;
            const auto& neighbors = edge->neighbors;
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

        std::vector<NeighborCandidate> results;
        results.reserve(nearest.size());
        while (!nearest.empty()) {
            results.push_back({nearest.top().id, nearest.top().distance});
            nearest.pop();
        }
        std::sort(results.begin(), results.end());
        return results;
    }

    void prune_upper_neighbors(NodeId node, int level) {
        auto& nb = get_or_create_edge(level, node).neighbors;
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

}  // namespace cphnsw
