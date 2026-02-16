#pragma once

#include "../core/types.hpp"
#include "../core/codes.hpp"
#include "../core/memory.hpp"
#include "../core/adaptive_defaults.hpp"
#include "../core/adaptive_gamma.hpp"
#include "../encoder/rabitq_encoder.hpp"
#include "../distance/fastscan_kernel.hpp"
#include "../graph/rabitq_graph.hpp"
#include "../graph/graph_refinement.hpp"
#include "../graph/neighbor_selection.hpp"
#include "../search/rabitq_search.hpp"
#include "../graph/visitation_table.hpp"
#include "../io/serialization.hpp"
#include "params.hpp"
#include <vector>
#include <random>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <type_traits>

#include <omp.h>

namespace cphnsw {

template <size_t D, size_t R = 32, size_t BitWidth = 1>
class Index {
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

    explicit Index(const IndexParams& params)
        : params_(params)
        , encoder_(params.dim, params.seed)
        , graph_(params.dim)
        , mL_(1.0 / std::log(static_cast<double>(M_UPPER)))
        , rng_(params.seed)
    {
        if (params.dim == 0) throw std::invalid_argument("dim must be > 0");
    }

    Index(const IndexParams& params, Graph&& graph, const HNSWLayerSnapshot& layer_data)
        : params_(params)
        , encoder_(params.dim, params.seed)
        , graph_(std::move(graph))
        , finalized_(true)
        , mL_(1.0 / std::log(static_cast<double>(M_UPPER)))
        , rng_(params.seed)
        , max_level_(layer_data.max_level)
        , entry_point_(layer_data.entry_point)
        , upper_tau_(layer_data.upper_tau)
        , node_levels_(layer_data.node_levels)
    {
        if (params.dim == 0) throw std::invalid_argument("dim must be > 0");
        // Convert HNSWLayerSnapshot edges to internal UpperLayerEdge format
        upper_layers_.resize(layer_data.upper_layers.size());
        for (size_t l = 0; l < layer_data.upper_layers.size(); ++l) {
            upper_layers_[l].reserve(layer_data.upper_layers[l].size());
            for (const auto& ext_edge : layer_data.upper_layers[l]) {
                upper_layers_[l].push_back(UpperLayerEdge{ext_edge.node, ext_edge.neighbors});
            }
        }
        compute_gamma_statistics();
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

        compute_gamma_statistics();

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

        float gamma;
        if (params.gamma_override >= 0.0f) {
            gamma = params.gamma_override;
        } else if (gamma_estimator_.is_initialized()) {
            gamma = estimate_per_query_gamma(query, params.recall_target);
        } else {
            float p = 1.0f - std::clamp(params.recall_target, 0.5f, 0.9999f);
            gamma = -std::log(p);
        }
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

    HNSWLayerSnapshot get_layer_snapshot() const {
        HNSWLayerSnapshot snap;
        snap.max_level = max_level_;
        snap.entry_point = entry_point_;
        snap.upper_tau = upper_tau_;
        snap.node_levels = node_levels_;
        snap.upper_layers.resize(upper_layers_.size());
        for (size_t l = 0; l < upper_layers_.size(); ++l) {
            snap.upper_layers[l].reserve(upper_layers_[l].size());
            for (const auto& edge : upper_layers_[l]) {
                snap.upper_layers[l].push_back(HNSWLayerEdge{edge.node, edge.neighbors});
            }
        }
        return snap;
    }

    size_t size() const { return graph_.size(); }
    size_t dim() const { return params_.dim; }
    bool is_finalized() const { return finalized_; }
    const Graph& graph() const { return graph_; }

private:
    IndexParams params_;
    Encoder encoder_;
    Graph graph_;
    AdaptiveGammaEstimator gamma_estimator_;
    bool finalized_ = false;
    bool needs_build_ = false;

    double mL_;
    std::mt19937_64 rng_;

    int max_level_ = 0;
    NodeId entry_point_ = INVALID_NODE;
    float upper_tau_ = 0.0f;

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

        // Estimate upper_tau_ from nearest-neighbor distances of upper-layer nodes
        std::vector<float> upper_nn_dists;
        for (size_t idx = 0; idx < n && upper_nn_dists.size() < 200; ++idx) {
            NodeId node = insertion_order[idx];
            if (node_levels_[node] == 0) break;
            // Find nearest among other upper-layer nodes
            float best = std::numeric_limits<float>::max();
            for (size_t jdx = 0; jdx < n && jdx < 500; ++jdx) {
                NodeId other = insertion_order[jdx];
                if (other == node) continue;
                if (node_levels_[other] == 0) break;
                float d = exact_distance(graph_.get_vector(node), graph_.get_vector(other));
                if (d < best) best = d;
            }
            if (best < std::numeric_limits<float>::max()) {
                upper_nn_dists.push_back(best);
            }
        }
        if (!upper_nn_dists.empty()) {
            std::sort(upper_nn_dists.begin(), upper_nn_dists.end());
            size_t p10 = upper_nn_dists.size() / 10;
            upper_tau_ = upper_nn_dists[p10] * adaptive_defaults::tau_scaling_factor();
        }

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
                auto selected = select_neighbors_alpha_cng(
                    std::move(candidates), M_UPPER, dist_fn, zero_error,
                    adaptive_defaults::alpha_default(D), upper_tau_);

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
        auto selected = select_neighbors_alpha_cng(
            std::move(candidates), M_UPPER, dist_fn, zero_error,
            adaptive_defaults::alpha_default(D), upper_tau_);
        nb.clear();
        nb.reserve(selected.size());
        for (const auto& s : selected) {
            nb.push_back(s.id);
        }
    }

    void compute_gamma_statistics() {
        size_t n = graph_.size();
        size_t sample_size = std::min(n, size_t(500));
        if (sample_size == 0) return;

        std::vector<float> rotated_sample(sample_size * D);
        std::mt19937 rng(params_.seed + 12345);
        std::vector<size_t> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);

        for (size_t i = 0; i < sample_size; ++i) {
            NodeId id = static_cast<NodeId>(indices[i]);
            encoder_.rotate_raw_vector(graph_.get_vector(id), rotated_sample.data() + i * D);
        }

        gamma_estimator_.compute_statistics(rotated_sample.data(), sample_size, D);
    }

    float estimate_per_query_gamma(const float* query, float recall_target) const {
        alignas(32) float query_rotated[D];
        encoder_.rotate_raw_vector(query, query_rotated);

        NodeId ep = (entry_point_ != INVALID_NODE) ? entry_point_ : graph_.entry_point();
        if (ep == INVALID_NODE) {
            float p = 1.0f - std::clamp(recall_target, 0.5f, 0.9999f);
            return -std::log(p);
        }

        const auto& nb = graph_.get_neighbors(ep);
        thread_local std::vector<float> dist_samples;
        dist_samples.clear();
        dist_samples.reserve(R);
        for (size_t i = 0; i < nb.size(); ++i) {
            NodeId nid = nb.neighbor_ids[i];
            if (nid == INVALID_NODE) continue;
            dist_samples.push_back(l2_distance_simd<D>(query, graph_.get_vector(nid)));
        }

        return gamma_estimator_.estimate_gamma(
            query_rotated, dist_samples.data(), dist_samples.size(), recall_target);
    }

    static float zero_error(NodeId) { return 0.0f; }

    static float exact_distance(const float* a, const float* b) {
        return l2_distance_simd<D>(a, b);
    }
};

}  // namespace cphnsw
