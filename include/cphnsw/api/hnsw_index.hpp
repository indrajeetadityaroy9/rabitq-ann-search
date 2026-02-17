#pragma once

#include "../core/types.hpp"
#include "../core/codes.hpp"
#include "../core/memory.hpp"
#include "../core/adaptive_defaults.hpp"
#include "../core/evt_crc.hpp"
#include "../distance/fastscan_kernel.hpp"
#include "../encoder/rabitq_encoder.hpp"
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
#include <numeric>
#include <cstdint>
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
    static constexpr uint64_t DEFAULT_LAYER_SEED = 42;
    static constexpr uint64_t DEFAULT_CALIBRATION_SEED = 99999;
    static constexpr size_t DEFAULT_CALIBRATION_SAMPLES = 2000;

    explicit Index(const IndexParams& params)
        : params_(params)
        , encoder_(params.dim)
        , graph_(params.dim)
        , mL_(1.0 / std::log(static_cast<double>(M_UPPER)))
        , rng_(DEFAULT_LAYER_SEED)
    {
        if (params.dim == 0) throw std::invalid_argument("dim must be > 0");
    }

    Index(const IndexParams& params, Graph&& graph, const HNSWLayerSnapshot& layer_data,
          const CalibrationSnapshot& cal = CalibrationSnapshot())
        : params_(params)
        , encoder_(params.dim)
        , graph_(std::move(graph))
        , finalized_(true)
        , mL_(1.0 / std::log(static_cast<double>(M_UPPER)))
        , rng_(DEFAULT_LAYER_SEED)
        , max_level_(layer_data.max_level)
        , entry_point_(layer_data.entry_point)
        , upper_tau_(layer_data.upper_tau)
        , node_levels_(layer_data.node_levels)
    {
        if (params.dim == 0) throw std::invalid_argument("dim must be > 0");
        upper_layers_.resize(layer_data.upper_layers.size());
        for (size_t l = 0; l < layer_data.upper_layers.size(); ++l) {
            upper_layers_[l].reserve(layer_data.upper_layers[l].size());
            for (const auto& ext_edge : layer_data.upper_layers[l]) {
                upper_layers_[l].push_back(UpperLayerEdge{ext_edge.node, ext_edge.neighbors});
            }
        }
        set_calibration_snapshot(cal);
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

    void finalize() {
        size_t n = graph_.size();
        if (n == 0 || !needs_build_) { finalized_ = true; return; }

        assign_layers(n);

        build_upper_layers();

        // Reuse prior residual scale if present; default to neutral scale.
        float resid_sigma = calibration_.resid_sigma > 0
            ? calibration_.resid_sigma : 1.0f;
        auto perm = graph_refinement::optimize_graph_adaptive(
            graph_, encoder_, resid_sigma);

        // Re-map upper-layer graph after BFS permutation.
        if (!perm.old_to_new.empty()) {
            for (auto& layer : upper_layers_) {
                for (auto& edge : layer) {
                    if (edge.node < perm.old_to_new.size()) {
                        edge.node = perm.old_to_new[edge.node];
                    }
                    for (auto& nb : edge.neighbors) {
                        if (nb < perm.old_to_new.size()) {
                            nb = perm.old_to_new[nb];
                        }
                    }
                }
                std::sort(layer.begin(), layer.end());
            }
            if (entry_point_ != INVALID_NODE && entry_point_ < perm.old_to_new.size()) {
                entry_point_ = perm.old_to_new[entry_point_];
            }
            if (!node_levels_.empty()) {
                std::vector<int> new_levels(node_levels_.size());
                for (size_t i = 0; i < node_levels_.size(); ++i) {
                    new_levels[perm.old_to_new[i]] = node_levels_[i];
                }
                node_levels_ = std::move(new_levels);
            }
        }

        calibrate_estimator(DEFAULT_CALIBRATION_SAMPLES);

        needs_build_ = false;
        finalized_ = true;
    }

    std::vector<SearchResult> search(
        const float* query,
        const SearchParams& params = SearchParams()) const
    {
        if (graph_.empty()) return {};
        if (!calibration_.calibrated || !calibration_.evt.fitted) {
            throw std::runtime_error(
                "Index is not calibrated with EVT-CRC. Rebuild with finalize().");
        }

        constexpr float kPruneRiskFrac = 0.7f;
        constexpr int kSlackLevels = 16;
        QueryType encoded = encoder_.encode_query_raw(query);
        encoded.affine_a = calibration_.affine_a;
        encoded.affine_b = calibration_.affine_b;
        encoded.ip_qo_floor = calibration_.ip_qo_floor;
        encoded.resid_q99_dot = calibration_.resid_q99_dot;
        float dot_slack_levels[32];
        float delta = 1.0f - std::clamp(params.recall_target, 0.5f, 0.9999f);
        float delta_prune = kPruneRiskFrac * delta;
        float delta_term = (1.0f - kPruneRiskFrac) * delta;

        int evt_L = std::clamp(kSlackLevels, 1, 32);
        for (int i = 1; i <= evt_L; ++i) {
            float alpha_i = evt_crc::alpha_spend(i, delta_prune);
            dot_slack_levels[i - 1] = evt_crc::evt_quantile(
                alpha_i, calibration_.evt, calibration_.resid_q99_dot);
        }
        encoded.dot_slack = dot_slack_levels[0];

        float alpha_term = delta_term / static_cast<float>(
            std::max(params.k * 4, size_t(1)));
        float dot_slack_term = evt_crc::evt_quantile(
            alpha_term, calibration_.evt, calibration_.resid_q99_dot);
        float term_slack_sq = 2.0f * calibration_.evt.nop_p95 * dot_slack_term;

        float slack_factor = std::sqrt(std::max(term_slack_sq, 0.01f));
        size_t ef_cap = static_cast<size_t>(
            adaptive_defaults::ef_cap(graph_.size(), params.k, slack_factor));
        ef_cap = std::clamp(ef_cap, params.k * 4, size_t(8192));

        thread_local TwoLevelVisitationTable visited(0);
        if (visited.capacity() < graph_.size()) {
            visited.resize(graph_.size() + adaptive_defaults::visitation_headroom(graph_.size()));
        }

        NodeId ep = INVALID_NODE;
        if (max_level_ > 0 && entry_point_ != INVALID_NODE) {
            ep = entry_point_;
            for (int level = max_level_; level >= 1; --level) {
                ep = greedy_search_layer(query, ep, level);
            }
        }

        return rabitq_search::search<D, R, BitWidth>(
            encoded, query, graph_, params.k, term_slack_sq, visited, ef_cap, ep,
            dot_slack_levels, evt_L);
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

    float calibration_affine_a() const { return calibration_.affine_a; }
    float calibration_affine_b() const { return calibration_.affine_b; }
    float calibration_ip_qo_floor() const { return calibration_.ip_qo_floor; }
    float calibration_resid_sigma() const { return calibration_.resid_sigma; }
    float calibration_resid_q99_dot() const { return calibration_.resid_q99_dot; }
    float calibration_median_nn_dist_sq() const { return calibration_.median_nn_dist_sq; }
    float calibration_corr() const { return calibration_.calibration_corr; }
    bool calibration_calibrated() const { return calibration_.calibrated; }

    CalibrationSnapshot get_calibration_snapshot() const {
        CalibrationSnapshot snap;
        snap.affine_a = calibration_.affine_a;
        snap.affine_b = calibration_.affine_b;
        snap.ip_qo_floor = calibration_.ip_qo_floor;
        snap.resid_q99_dot = calibration_.resid_q99_dot;
        snap.resid_sigma = calibration_.resid_sigma;
        snap.median_nn_dist_sq = calibration_.median_nn_dist_sq;
        snap.calibration_corr = calibration_.calibration_corr;
        snap.flags = calibration_.calibrated ? 1u : 0u;
        snap.evt = calibration_.evt;
        return snap;
    }

    void set_calibration_snapshot(const CalibrationSnapshot& snap) {
        calibration_.affine_a = snap.affine_a;
        calibration_.affine_b = snap.affine_b;
        calibration_.ip_qo_floor = snap.ip_qo_floor;
        calibration_.resid_q99_dot = snap.resid_q99_dot;
        calibration_.resid_sigma = snap.resid_sigma;
        calibration_.median_nn_dist_sq = snap.median_nn_dist_sq;
        calibration_.calibration_corr = snap.calibration_corr;
        calibration_.calibrated = (snap.flags & 1u) != 0;
        calibration_.evt = snap.evt;
    }

private:
    IndexParams params_;
    Encoder encoder_;
    Graph graph_;
    bool finalized_ = false;
    bool needs_build_ = false;

    struct CalibrationState {
        float affine_a = 1.0f;
        float affine_b = 0.0f;
        float ip_qo_floor = 0.0f;
        float resid_sigma = 0.0f;
        float resid_q99_dot = 0.0f;
        float median_nn_dist_sq = 0.0f;
        float calibration_corr = 0.0f;
        bool calibrated = false;
        EVTState evt;
    };
    CalibrationState calibration_;

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

        std::vector<float> upper_nn_dists;
        for (size_t idx = 0; idx < n && upper_nn_dists.size() < 200; ++idx) {
            NodeId node = insertion_order[idx];
            if (node_levels_[node] == 0) break;
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

    void calibrate_estimator(size_t num_samples) {
        constexpr float kEvtThresholdQ = 0.90f;
        constexpr size_t kEvtMinTail = 64;
        constexpr float kEdgeNopQuantile = 0.95f;
        size_t n = graph_.size();
        if (n < 50) {
            throw std::runtime_error("Calibration requires at least 50 nodes.");
        }

        num_samples = std::min(num_samples, n);

        // Mix DB and synthetic queries for calibration coverage.
        std::mt19937 rng(static_cast<uint32_t>(DEFAULT_LAYER_SEED + DEFAULT_CALIBRATION_SEED));
        std::vector<size_t> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);

        size_t n_db = std::min(num_samples, n);
        size_t n_synth = std::min(num_samples / 2, n);

        std::vector<float> dim_var(D, 0.0f);
        size_t var_sample = std::min(n, size_t(500));
        for (size_t i = 0; i < var_sample; ++i) {
            const float* v = graph_.get_vector(static_cast<NodeId>(indices[i]));
            for (size_t d = 0; d < D; ++d) {
                dim_var[d] += v[d] * v[d];
            }
        }
        std::vector<float> dim_mean(D, 0.0f);
        for (size_t i = 0; i < var_sample; ++i) {
            const float* v = graph_.get_vector(static_cast<NodeId>(indices[i]));
            for (size_t d = 0; d < D; ++d) {
                dim_mean[d] += v[d];
            }
        }
        for (size_t d = 0; d < D; ++d) {
            dim_mean[d] /= static_cast<float>(var_sample);
            dim_var[d] = dim_var[d] / static_cast<float>(var_sample) - dim_mean[d] * dim_mean[d];
            if (dim_var[d] < 1e-12f) dim_var[d] = 1e-12f;
        }

        std::vector<float> ip_qo_values;
        std::vector<float> per_sample_ip_corrected;
        std::vector<float> per_sample_ip_qo;
        std::vector<float> truths;
        std::vector<float> nn_dists_sq;

        std::vector<float> nop_samples;

        ip_qo_values.reserve(num_samples * 4);
        per_sample_ip_corrected.reserve(num_samples * 4);
        per_sample_ip_qo.reserve(num_samples * 4);
        truths.reserve(num_samples * 4);
        nop_samples.reserve(num_samples * 4);

        auto process_query = [&](const float* query_vec) {
            NodeId ep = (entry_point_ != INVALID_NODE) ? entry_point_ : graph_.entry_point();
            if (ep == INVALID_NODE) return;

            const auto& nb = graph_.get_neighbors(ep);
            if (nb.size() == 0) return;

            float best_dist = l2_distance_simd<D>(query_vec, graph_.get_vector(ep));
            NodeId parent = ep;
            for (size_t i = 0; i < nb.size(); ++i) {
                NodeId nid = nb.neighbor_ids[i];
                if (nid == INVALID_NODE) continue;
                float d = l2_distance_simd<D>(query_vec, graph_.get_vector(nid));
                if (d < best_dist) {
                    best_dist = d;
                    parent = nid;
                }
            }
            nn_dists_sq.push_back(best_dist);

            const auto& pnb = graph_.get_neighbors(parent);
            if (pnb.size() == 0) return;

            QueryType encoded = encoder_.encode_query_raw(query_vec);

            float dist_qp_sq = l2_distance_simd<D>(query_vec, graph_.get_vector(parent));

            size_t num_batches = (pnb.size() + 31) / 32;
            for (size_t batch = 0; batch < num_batches; ++batch) {
                size_t batch_start = batch * 32;
                size_t batch_count = std::min(size_t(32), pnb.size() - batch_start);

                alignas(64) uint32_t fastscan_sums[32];
                if constexpr (BitWidth == 1) {
                    fastscan::compute_inner_products(
                        encoded.lut, pnb.code_blocks[batch], fastscan_sums);
                } else {
                    alignas(64) uint32_t msb_sums[32];
                    fastscan::compute_nbit_inner_products<D, BitWidth>(
                        encoded.lut, pnb.code_blocks[batch],
                        fastscan_sums, msb_sums);
                }

                for (size_t j = 0; j < batch_count; ++j) {
                    size_t ni = batch_start + j;
                    NodeId neighbor = pnb.neighbor_ids[ni];
                    if (neighbor == INVALID_NODE) continue;

                    float ip_qo = pnb.ip_qo[ni];
                    ip_qo_values.push_back(ip_qo);

                    float nop = pnb.nop[ni];
                    if (nop < 1e-12f) continue;
                    nop_samples.push_back(nop);

                    float A = encoded.coeff_fastscan;
                    float B = encoded.coeff_popcount;
                    float C = encoded.coeff_constant;

                    float ip_approx;
                    if constexpr (BitWidth == 1) {
                        ip_approx = A * static_cast<float>(fastscan_sums[j])
                                  + B * static_cast<float>(pnb.popcounts[ni]) + C;
                    } else {
                        constexpr float K = static_cast<float>((1u << BitWidth) - 1);
                        constexpr float inv_K = 1.0f / K;
                        ip_approx = A * inv_K * static_cast<float>(fastscan_sums[j])
                                  + B * inv_K * static_cast<float>(pnb.weighted_popcounts[ni]) + C;
                    }

                    float ip_corrected = ip_approx - pnb.ip_cp[ni];
                    if (std::abs(ip_qo) < adaptive_defaults::ip_quality_epsilon()) continue;

                    const float* p_vec = graph_.get_vector(parent);
                    const float* o_vec = graph_.get_vector(neighbor);
                    float true_ip = 0.0f;
                    for (size_t d = 0; d < D; ++d) {
                        true_ip += (query_vec[d] - p_vec[d]) * (o_vec[d] - p_vec[d]);
                    }
                    true_ip /= nop;

                    per_sample_ip_corrected.push_back(ip_corrected);
                    per_sample_ip_qo.push_back(ip_qo);
                    truths.push_back(true_ip);
                }
            }
        };

        for (size_t i = 0; i < n_db; ++i) {
            const float* v = graph_.get_vector(static_cast<NodeId>(indices[i]));
            process_query(v);
        }

        std::normal_distribution<float> normal_dist(0.0f, 1.0f);
        std::vector<float> synth_query(D);
        for (size_t i = 0; i < n_synth; ++i) {
            const float* base = graph_.get_vector(static_cast<NodeId>(indices[i % n]));
            for (size_t d = 0; d < D; ++d) {
                synth_query[d] = base[d] + normal_dist(rng) * std::sqrt(dim_var[d]);
            }
            process_query(synth_query.data());
        }

        if (ip_qo_values.empty()) {
            throw std::runtime_error("Calibration failed: no ip_qo samples.");
        }
        std::sort(ip_qo_values.begin(), ip_qo_values.end());
        size_t p5_idx = ip_qo_values.size() * 5 / 100;
        calibration_.ip_qo_floor = ip_qo_values[p5_idx];

        std::vector<float> floored_estimates;
        floored_estimates.reserve(per_sample_ip_corrected.size());
        for (size_t i = 0; i < per_sample_ip_corrected.size(); ++i) {
            float floored_qo = std::max(per_sample_ip_qo[i], calibration_.ip_qo_floor);
            floored_estimates.push_back(per_sample_ip_corrected[i] / floored_qo);
        }

        if (floored_estimates.size() < 20) {
            throw std::runtime_error("Calibration failed: too few estimator/target pairs.");
        }

        // Affine fit: T ~= aE + b.
        size_t np = floored_estimates.size();
        double sum_e = 0, sum_t = 0, sum_ee = 0, sum_et = 0;
        for (size_t i = 0; i < np; ++i) {
            double e = floored_estimates[i];
            double t = truths[i];
            sum_e += e;
            sum_t += t;
            sum_ee += e * e;
            sum_et += e * t;
        }
        double mean_e = sum_e / np;
        double mean_t = sum_t / np;
        double var_e = sum_ee / np - mean_e * mean_e;
        double cov_et = sum_et / np - mean_e * mean_t;

        if (var_e > 1e-12) {
            calibration_.affine_a = static_cast<float>(cov_et / var_e);
            calibration_.affine_b = static_cast<float>(mean_t - (cov_et / var_e) * mean_e);
        }

        std::vector<float> abs_residuals(np);
        double sum_r2 = 0;
        for (size_t i = 0; i < np; ++i) {
            float predicted = calibration_.affine_a * floored_estimates[i] + calibration_.affine_b;
            float r = truths[i] - predicted;
            abs_residuals[i] = std::abs(r);
            sum_r2 += static_cast<double>(r) * r;
        }
        calibration_.resid_sigma = static_cast<float>(std::sqrt(sum_r2 / np));
        std::sort(abs_residuals.begin(), abs_residuals.end());
        calibration_.resid_q99_dot = abs_residuals[np * 99 / 100];

        double sum_tt = 0;
        for (size_t i = 0; i < np; ++i) {
            sum_tt += static_cast<double>(truths[i]) * truths[i];
        }
        double var_t = sum_tt / np - mean_t * mean_t;
        double denom_corr = std::sqrt(var_e * var_t);
        calibration_.calibration_corr = (denom_corr > 1e-12)
            ? static_cast<float>(cov_et / denom_corr) : 0.0f;

        if (!nn_dists_sq.empty()) {
            std::sort(nn_dists_sq.begin(), nn_dists_sq.end());
            calibration_.median_nn_dist_sq = nn_dists_sq[nn_dists_sq.size() / 2];
        }

        calibration_.calibrated = true;

        // Fit EVT tail on |residual| for risk-to-slack conversion.
        calibration_.evt = evt_crc::fit_gpd(
            abs_residuals.data(), np, kEvtThresholdQ, kEvtMinTail);

        if (nop_samples.empty()) {
            throw std::runtime_error("Calibration failed: no nop samples for EVT.");
        }
        std::sort(nop_samples.begin(), nop_samples.end());
        size_t p95_idx = static_cast<size_t>(
            static_cast<float>(nop_samples.size()) * kEdgeNopQuantile);
        p95_idx = std::min(p95_idx, nop_samples.size() - 1);
        calibration_.evt.nop_p95 = nop_samples[p95_idx];

        if (!calibration_.evt.fitted || calibration_.evt.nop_p95 <= 0.0f) {
            throw std::runtime_error("Calibration failed: EVT-CRC fit did not converge.");
        }

    }

    static float zero_error(NodeId) { return 0.0f; }

    static float exact_distance(const float* a, const float* b) {
        return l2_distance_simd<D>(a, b);
    }
};

}  // namespace cphnsw
