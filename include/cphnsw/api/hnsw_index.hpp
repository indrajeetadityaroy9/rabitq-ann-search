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
#include "../graph/visitation_table.hpp"
#include "params.hpp"
#include <vector>
#include <random>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <type_traits>

#include "../core/omp_compat.hpp"

namespace cphnsw {

template <size_t D, size_t R = 32, size_t BitWidth = 1>
class HNSWIndex {
public:
    using CodeType = std::conditional_t<BitWidth == 1,
        RaBitQCode<D>, NbitRaBitQCode<D, BitWidth>>;
    using QueryType = RaBitQQuery<D>;
    using Encoder = std::conditional_t<BitWidth == 1,
        RaBitQEncoder<D>,
        NbitRaBitQEncoder<D, BitWidth>>;
    using Graph = RaBitQGraph<D, R, BitWidth>;
    static constexpr size_t DIMS = D;
    static constexpr size_t DEGREE = R;
    static constexpr size_t BIT_WIDTH = BitWidth;

    static constexpr size_t M_UPPER = adaptive_defaults::upper_layer_degree(R, D);

    explicit HNSWIndex(const IndexParams& params)
        : params_(params)
        , encoder_(params.dim, params.seed)
        , graph_(params.dim)
        , mL_(1.0 / std::log(static_cast<double>(M_UPPER)))
        , rng_(params.seed)
    {
        if (params.dim == 0) throw std::invalid_argument("dim must be > 0");
    }

    HNSWIndex(const IndexParams& params, Graph&& graph)
        : params_(params)
        , encoder_(params.dim, params.seed)
        , graph_(std::move(graph))
        , finalized_(true)
        , mL_(1.0 / std::log(static_cast<double>(M_UPPER)))
        , rng_(params.seed)
    {
        if (params.dim == 0) throw std::invalid_argument("dim must be > 0");
    }

    void add_batch(const float* vecs, size_t num_vecs) {
        if (num_vecs == 0) return;

        graph_.reserve(graph_.size() + num_vecs);

        std::vector<CodeType> codes(num_vecs);
        encoder_.encode_batch(vecs, num_vecs, codes.data());

        for (size_t i = 0; i < num_vecs; ++i) {
            graph_.add_node(codes[i], vecs + i * params_.dim);
        }

        needs_build_ = true;
    }

    void finalize(const BuildParams& params) {
        size_t n = graph_.size();
        if (n == 0 || !needs_build_) { finalized_ = true; return; }

        if (params.verbose) {
            printf("event=hnsw_build_start nodes=%zu bit_width=%zu\n", n, BitWidth);
        }
        assign_layers(n);

        build_upper_layers();

        graph_refinement::optimize_graph_adaptive(
            graph_, encoder_, params.num_threads, params.verbose, params_.seed);

        needs_build_ = false;
        finalized_ = true;
        if (params.verbose) {
            printf("event=hnsw_build_done nodes=%zu max_level=%d entry=%u avg_degree=%.3f max_degree=%zu\n",
                   graph_.size(), max_level_, entry_point_, graph_.average_degree(), graph_.max_degree());
        }
    }

    std::vector<SearchResult> search(
        const float* query,
        const SearchParams& params = SearchParams()) const
    {
        if (graph_.empty()) return {};

        float gamma = (params.gamma_override >= 0.0f)
            ? params.gamma_override
            : adaptive_defaults::gamma_from_recall(params.recall_target, D);
        float eps = adaptive_defaults::error_epsilon_search(params.recall_target);

        QueryType encoded = encoder_.encode_query_raw(query);
        encoded.error_epsilon = eps;

        thread_local TwoLevelVisitationTable visited(0);
        if (visited.capacity() < graph_.size()) {
            visited.resize(graph_.size() + adaptive_defaults::visitation_headroom(graph_.size()));
        }

        if (max_level_ <= 0 || entry_point_ == INVALID_NODE) {
            return rabitq_search::search<D, R, BitWidth>(encoded, query, graph_, params.k, gamma, visited);
        }

        NodeId ep = entry_point_;
        for (int level = max_level_; level >= 1; --level) {
            ep = greedy_search_layer(query, ep, level);
        }

        return rabitq_search::search<D, R, BitWidth>(encoded, query, graph_, params.k, gamma, visited, 0, ep);
    }

    size_t size() const { return graph_.size(); }
    size_t dim() const { return params_.dim; }
    bool is_finalized() const { return finalized_; }
    const Graph& graph() const { return graph_; }

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

    void build_upper_layers() {
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
                size_t upper_ef = adaptive_defaults::upper_layer_ef(R, level);
                auto candidates = search_upper_layer(
                    graph_.get_vector(node), ep, level, upper_ef);

                auto dist_fn = [this](NodeId a, NodeId b) {
                    return exact_distance(graph_.get_vector(a), graph_.get_vector(b));
                };
                auto selected = select_neighbors_alpha_cg(
                    std::move(candidates), M_UPPER, dist_fn, zero_error,
                    adaptive_defaults::alpha_default(D));

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
            visited_table.resize(graph_.size() + adaptive_defaults::visitation_headroom(graph_.size()));
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
        auto selected = select_neighbors_alpha_cg(
            std::move(candidates), M_UPPER, dist_fn, zero_error,
            adaptive_defaults::alpha_default(D));
        nb.clear();
        nb.reserve(selected.size());
        for (const auto& s : selected) {
            nb.push_back(s.id);
        }
    }

    static float zero_error(NodeId) { return 0.0f; }

    static float exact_distance(const float* a, const float* b) {
        return l2_distance_simd<D>(a, b);
    }
};

}  // namespace cphnsw
