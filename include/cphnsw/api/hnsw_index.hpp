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
#include "params.hpp"
#include <vector>
#include <random>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include <type_traits>
#include <mutex>
#include <shared_mutex>
#include <limits>

#include <omp.h>

namespace cphnsw {

struct CalibrationSnapshot {
    float affine_a = 1.0f;
    float affine_b = 0.0f;
    float ip_qo_floor = 0.0f;
    float resid_q99_dot = 0.0f;
    float resid_sigma = 0.0f;
    float median_nn_dist_sq = 0.0f;
    float calibration_corr = 0.0f;
    uint32_t flags = 0;
    EVTState evt;
};

struct HNSWLayerEdge {
    NodeId node;
    std::vector<NodeId> neighbors;
    bool operator<(const HNSWLayerEdge& other) const { return node < other.node; }
};

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
        , encoder_(params.dim, constants::kDefaultRotationSeed)
        , graph_(params.dim)
        , mL_(1.0 / std::log(static_cast<double>(M_UPPER)))
        , rng_(constants::kDefaultLayerSeed)
    {
        if (params.dim == 0) throw std::invalid_argument("dim must be > 0");
    }

    void build(const float* vecs, size_t num_vecs) {
        std::unique_lock<std::shared_mutex> lock(index_mutex_);
        graph_ = Graph(params_.dim);
        calibration_ = CalibrationSnapshot();
        finalized_ = false;
        needs_build_ = false;
        max_level_ = 0;
        entry_point_ = INVALID_NODE;
        upper_tau_ = 0.0f;
        node_levels_.clear();
        upper_layers_.clear();
        add_batch_internal(vecs, num_vecs);
    }

    void finalize() {
        std::unique_lock<std::shared_mutex> lock(index_mutex_);
        size_t n = graph_.size();
        if (n == 0) {
            throw std::runtime_error("Cannot finalize an empty index.");
        }
        if (!needs_build_) {
            throw std::runtime_error("Finalize called without a pending build.");
        }

        assign_layers(n);

        build_upper_layers();

        // Graph optimization uses fixed neutral residual scale.
        float resid_sigma = 1.0f;
        auto perm = graph_refinement::optimize_graph_adaptive(
            graph_, encoder_, resid_sigma);

        // Re-map upper-layer graph after BFS permutation.
        for (auto& layer : upper_layers_) {
            for (auto& edge : layer) {
                edge.node = perm.old_to_new[edge.node];
                for (auto& nb : edge.neighbors) {
                    nb = perm.old_to_new[nb];
                }
            }
            std::sort(layer.begin(), layer.end());
        }
        entry_point_ = perm.old_to_new[entry_point_];
        std::vector<int> new_levels(node_levels_.size());
        for (size_t i = 0; i < node_levels_.size(); ++i) {
            new_levels[perm.old_to_new[i]] = node_levels_[i];
        }
        node_levels_ = std::move(new_levels);

        calibrate_estimator(constants::kDefaultCalibSamples);

        needs_build_ = false;
        finalized_ = true;
    }

    std::vector<SearchResult> search(
        const float* query,
        const SearchRequest& request = SearchRequest()) const
    {
        std::shared_lock<std::shared_mutex> lock(index_mutex_);
        if (graph_.empty()) {
            throw std::runtime_error("Search requested on an empty index.");
        }
        if (!calibration_.flags || !calibration_.evt.fitted) {
            throw std::runtime_error(
                "Index is not calibrated with EVT-CRC. Rebuild with finalize().");
        }

        QueryType encoded = encoder_.encode_query_raw(query);
        encoded.affine_a = calibration_.affine_a;
        encoded.affine_b = calibration_.affine_b;
        encoded.ip_qo_floor = calibration_.ip_qo_floor;
        encoded.resid_q99_dot = calibration_.resid_q99_dot;

        float target_recall = std::clamp(
            request.target_recall, constants::kMinRecallTarget, constants::kMaxRecallTarget);
        size_t k = std::max<size_t>(request.k, 1);

        float dot_slack_levels[constants::kMaxSlackArray];
        float delta = 1.0f - target_recall;
        float delta_prune = constants::kPruneRiskFrac * delta;
        float delta_term = (1.0f - constants::kPruneRiskFrac) * delta;

        int evt_L = std::clamp(constants::kSlackLevels, 1, constants::kMaxSlackArray);
        for (int i = 1; i <= evt_L; ++i) {
            float alpha_i = evt_crc::alpha_spend(i, delta_prune);
            dot_slack_levels[i - 1] = evt_crc::evt_quantile(
                alpha_i, calibration_.evt, calibration_.resid_q99_dot);
        }
        encoded.dot_slack = dot_slack_levels[0];

        float alpha_term = delta_term / static_cast<float>(
            std::max(k * constants::kAlphaTermKMult, size_t(1)));
        float dot_slack_term = evt_crc::evt_quantile(
            alpha_term, calibration_.evt, calibration_.resid_q99_dot);
        float dist_slack = constants::kSlackMultiplier * calibration_.evt.nop_p95 * dot_slack_term;
        float ref_dist = std::max(calibration_.median_nn_dist_sq, constants::kMinSlackSq);
        float gamma = std::clamp(1.0f + dist_slack / ref_dist,
            constants::kGammaMin, constants::kGammaMax);

        size_t ef_cap = static_cast<size_t>(
            adaptive_defaults::ef_cap(graph_.size(), k, gamma));
        ef_cap = std::clamp(ef_cap, k * constants::kEfMinMultiplier, constants::kEfMaxCap);

        thread_local TwoLevelVisitationTable visited(0);
        if (visited.capacity() < graph_.size()) {
            visited.resize(graph_.size() + adaptive_defaults::visitation_headroom(graph_.size()));
        }

        NodeId ep = graph_.entry_point();
        if (max_level_ > 0) {
            ep = entry_point_;
            for (int level = max_level_; level >= 1; --level) {
                ep = greedy_search_layer(query, ep, level);
            }
        }
        if (ep == INVALID_NODE || !graph_.is_alive(ep)) {
            throw std::runtime_error("Search failed: invalid entry point after finalize.");
        }

        return rabitq_search::search<D, R, BitWidth>(
            encoded, query, graph_, k, gamma, visited, ef_cap, ep,
            dot_slack_levels, evt_L);
    }

    size_t size() const { return graph_.size(); }
    size_t dim() const { return params_.dim; }
    bool is_finalized() const { return finalized_; }

private:
    IndexParams params_;
    Encoder encoder_;
    Graph graph_;
    bool finalized_ = false;
    bool needs_build_ = false;

    CalibrationSnapshot calibration_;

    double mL_;
    std::mt19937_64 rng_;

    int max_level_ = 0;
    NodeId entry_point_ = INVALID_NODE;
    float upper_tau_ = 0.0f;

    std::vector<int> node_levels_;

    std::vector<std::vector<HNSWLayerEdge>> upper_layers_;
    mutable std::shared_mutex index_mutex_;

    HNSWLayerEdge* find_edge(int level, NodeId node) {
        auto& layer = upper_layers_[level - 1];
        HNSWLayerEdge target{node, {}};
        auto it = std::lower_bound(layer.begin(), layer.end(), target);
        if (it != layer.end() && it->node == node) return &(*it);
        return nullptr;
    }

    const HNSWLayerEdge* find_edge(int level, NodeId node) const {
        const auto& layer = upper_layers_[level - 1];
        HNSWLayerEdge target{node, {}};
        auto it = std::lower_bound(layer.begin(), layer.end(), target);
        if (it != layer.end() && it->node == node) return &(*it);
        return nullptr;
    }

    HNSWLayerEdge& get_or_create_edge(int level, NodeId node) {
        auto& layer = upper_layers_[level - 1];
        HNSWLayerEdge target{node, {}};
        auto it = std::lower_bound(layer.begin(), layer.end(), target);
        if (it != layer.end() && it->node == node) return *it;
        return *layer.insert(it, HNSWLayerEdge{node, {}});
    }

    void add_batch_internal(const float* vecs, size_t num_vecs) {
        if (num_vecs == 0) {
            throw std::invalid_argument("build requires at least one vector.");
        }

        graph_.reserve(graph_.size() + num_vecs);

        std::vector<CodeType> codes(num_vecs);
        encoder_.encode_batch(vecs, num_vecs, codes.data());

        for (size_t i = 0; i < num_vecs; ++i) {
            graph_.add_node(codes[i], vecs + i * params_.dim);
        }

        needs_build_ = true;
        finalized_ = false;
    }

    void assign_layers(size_t n) {
        node_levels_.resize(n);
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        max_level_ = 0;
        entry_point_ = INVALID_NODE;

        for (size_t i = 0; i < n; ++i) {
            double r = dist(rng_);
            if (r < constants::kMinLayerRandom) r = constants::kMinLayerRandom;
            int level = static_cast<int>(-std::log(r) * mL_);
            node_levels_[i] = level;
            if (entry_point_ == INVALID_NODE || level > max_level_) {
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
        for (size_t idx = 0; idx < n && upper_nn_dists.size() < constants::kUpperLayerDistSamples; ++idx) {
            NodeId node = insertion_order[idx];
            if (node_levels_[node] == 0) break;
            float best = std::numeric_limits<float>::max();
            for (size_t jdx = 0; jdx < n && jdx < constants::kUpperLayerNnLimit; ++jdx) {
                NodeId other = insertion_order[jdx];
                if (other == node) continue;
                if (node_levels_[other] == 0) break;
                float d = l2_distance_simd<D>(graph_.get_vector(node), graph_.get_vector(other));
                if (d < best) best = d;
            }
            if (best < std::numeric_limits<float>::max()) {
                upper_nn_dists.push_back(best);
            }
        }
        if (!upper_nn_dists.empty()) {
            std::sort(upper_nn_dists.begin(), upper_nn_dists.end());
            size_t p10 = upper_nn_dists.size() / constants::kTauPercentileDiv;
            upper_tau_ = upper_nn_dists[p10] * constants::kTauScale;
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
                    return l2_distance_simd<D>(graph_.get_vector(a), graph_.get_vector(b));
                };
                auto selected = select_neighbors_alpha_cng(
                    std::move(candidates), M_UPPER, dist_fn, [](NodeId) { return 0.0f; },
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

        float best_dist = l2_distance_simd<D>(query, graph_.get_vector(ep));
        NodeId best_id = ep;
        bool improved = true;

        while (improved) {
            improved = false;
            const auto* edge = find_edge(level, best_id);
            if (!edge) break;
            const auto& neighbors = edge->neighbors;
            for (NodeId nb : neighbors) {
                float d = l2_distance_simd<D>(query, graph_.get_vector(nb));
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
        if (ep == INVALID_NODE) {
            throw std::runtime_error("Upper-layer search failed: invalid entry.");
        }

        MinHeap candidates;
        MaxHeap nearest;

        float ep_dist = l2_distance_simd<D>(query, graph_.get_vector(ep));
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

                float d = l2_distance_simd<D>(query, graph_.get_vector(nb));

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
            float d = l2_distance_simd<D>(vec, graph_.get_vector(id));
            candidates.push_back({id, d});
        }

        auto dist_fn = [this](NodeId a, NodeId b) {
            return l2_distance_simd<D>(graph_.get_vector(a), graph_.get_vector(b));
        };
        auto selected = select_neighbors_alpha_cng(
            std::move(candidates), M_UPPER, dist_fn, [](NodeId) { return 0.0f; },
            adaptive_defaults::alpha_default(D), upper_tau_);
        nb.clear();
        nb.reserve(selected.size());
        for (const auto& s : selected) {
            nb.push_back(s.id);
        }
    }

    void calibrate_estimator(size_t num_samples) {
        size_t n = graph_.size();
        if (n < constants::kMinCalibNodes) {
            throw std::runtime_error("Calibration requires at least 50 nodes.");
        }

        std::vector<NodeId> sample_ids(n);
        for (size_t i = 0; i < n; ++i) {
            sample_ids[i] = static_cast<NodeId>(i);
        }

        num_samples = std::min(num_samples, n);

        // Mix DB and synthetic queries for calibration coverage.
        std::mt19937 rng(static_cast<uint32_t>(constants::kDefaultLayerSeed + constants::kDefaultCalibrationSeed));
        std::shuffle(sample_ids.begin(), sample_ids.end(), rng);

        size_t n_db = std::min(num_samples, n);
        size_t n_synth = std::min(num_samples / 2, n);

        std::vector<float> dim_var(D, 0.0f);
        size_t var_sample = std::min(n, constants::kVarEstSamples);
        for (size_t i = 0; i < var_sample; ++i) {
            const float* v = graph_.get_vector(sample_ids[i]);
            for (size_t d = 0; d < D; ++d) {
                dim_var[d] += v[d] * v[d];
            }
        }
        std::vector<float> dim_mean(D, 0.0f);
        for (size_t i = 0; i < var_sample; ++i) {
            const float* v = graph_.get_vector(sample_ids[i]);
            for (size_t d = 0; d < D; ++d) {
                dim_mean[d] += v[d];
            }
        }
        for (size_t d = 0; d < D; ++d) {
            dim_mean[d] /= static_cast<float>(var_sample);
            dim_var[d] = dim_var[d] / static_cast<float>(var_sample) - dim_mean[d] * dim_mean[d];
            if (dim_var[d] < constants::kNearZeroSq) dim_var[d] = constants::kNearZeroSq;
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
            NodeId ep = entry_point_;
            if (ep == INVALID_NODE) {
                throw std::runtime_error("Calibration failed: invalid upper-layer entry point.");
            }

            const auto& nb = graph_.get_neighbors(ep);
            if (nb.size() == 0) {
                throw std::runtime_error("Calibration failed: entry point has no neighbors.");
            }

            float best_dist = l2_distance_simd<D>(query_vec, graph_.get_vector(ep));
            NodeId parent = ep;
            for (size_t i = 0; i < nb.size(); ++i) {
                NodeId nid = nb.neighbor_ids[i];
                if (nid == INVALID_NODE) {
                    throw std::runtime_error("Calibration failed: encountered invalid neighbor in entry layer.");
                }
                float d = l2_distance_simd<D>(query_vec, graph_.get_vector(nid));
                if (d < best_dist) {
                    best_dist = d;
                    parent = nid;
                }
            }
            nn_dists_sq.push_back(best_dist);

            const auto& pnb = graph_.get_neighbors(parent);
            if (pnb.size() == 0) {
                throw std::runtime_error("Calibration failed: chosen parent has no neighbors.");
            }

            QueryType encoded = encoder_.encode_query_raw(query_vec);

            float dist_qp_sq = l2_distance_simd<D>(query_vec, graph_.get_vector(parent));

            size_t num_batches = (pnb.size() + constants::kFastScanBatch - 1) / constants::kFastScanBatch;
            for (size_t batch = 0; batch < num_batches; ++batch) {
                size_t batch_start = batch * constants::kFastScanBatch;
                size_t batch_count = std::min(constants::kFastScanBatch, pnb.size() - batch_start);

                alignas(64) uint32_t fastscan_sums[constants::kFastScanBatch];
                if constexpr (BitWidth == 1) {
                    fastscan::compute_inner_products(
                        encoded.lut, pnb.code_blocks[batch], fastscan_sums);
                } else {
                    alignas(64) uint32_t msb_sums[constants::kFastScanBatch];
                    fastscan::compute_nbit_inner_products<D, BitWidth>(
                        encoded.lut, pnb.code_blocks[batch],
                        fastscan_sums, msb_sums);
                }

                for (size_t j = 0; j < batch_count; ++j) {
                    size_t ni = batch_start + j;
                    NodeId neighbor = pnb.neighbor_ids[ni];
                    if (neighbor == INVALID_NODE) {
                        throw std::runtime_error("Calibration failed: invalid neighbor in parent block.");
                    }

                    float ip_qo = pnb.ip_qo[ni];
                    ip_qo_values.push_back(ip_qo);

                    float nop = std::max(pnb.nop[ni], constants::kNearZeroSq);
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
                    float ip_qo_denom = std::max(std::abs(ip_qo), constants::kIpQualityEps);

                    const float* p_vec = graph_.get_vector(parent);
                    const float* o_vec = graph_.get_vector(neighbor);
                    float true_ip = 0.0f;
                    for (size_t d = 0; d < D; ++d) {
                        true_ip += (query_vec[d] - p_vec[d]) * (o_vec[d] - p_vec[d]);
                    }
                    true_ip /= nop;

                    per_sample_ip_corrected.push_back(ip_corrected);
                    per_sample_ip_qo.push_back(ip_qo_denom);
                    truths.push_back(true_ip);
                }
            }
        };

        for (size_t i = 0; i < n_db; ++i) {
            const float* v = graph_.get_vector(sample_ids[i]);
            process_query(v);
        }

        std::normal_distribution<float> normal_dist(0.0f, 1.0f);
        std::vector<float> synth_query(D);
        for (size_t i = 0; i < n_synth; ++i) {
            const float* base = graph_.get_vector(sample_ids[i % n]);
            for (size_t d = 0; d < D; ++d) {
                synth_query[d] = base[d] + normal_dist(rng) * std::sqrt(dim_var[d]);
            }
            process_query(synth_query.data());
        }

        if (ip_qo_values.empty()) {
            throw std::runtime_error("Calibration failed: no ip_qo samples.");
        }
        std::sort(ip_qo_values.begin(), ip_qo_values.end());
        size_t p5_idx = ip_qo_values.size() * constants::kIpQoFloorPct / 100;
        calibration_.ip_qo_floor = ip_qo_values[p5_idx];

        std::vector<float> floored_estimates;
        floored_estimates.reserve(per_sample_ip_corrected.size());
        for (size_t i = 0; i < per_sample_ip_corrected.size(); ++i) {
            float floored_qo = std::max(per_sample_ip_qo[i], calibration_.ip_qo_floor);
            floored_estimates.push_back(per_sample_ip_corrected[i] / floored_qo);
        }

        if (floored_estimates.size() < constants::kMinAffineSamples) {
            throw std::runtime_error("Calibration failed: too few estimator/target pairs.");
        }

        // Huber IRLS affine fit: T ~= aE + b.
        // Step 1: OLS seed.
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

        double a = 1.0, b = 0.0;
        if (var_e > constants::kNearZeroSq) {
            a = cov_et / var_e;
            b = mean_t - a * mean_e;
        }

        // Step 2: IRLS with Huber weights.
        std::vector<float> abs_residuals(np);
        for (int iter = 0; iter < constants::kHuberMaxIter; ++iter) {
            // Compute residuals and MAD.
            for (size_t i = 0; i < np; ++i) {
                float r = truths[i] - static_cast<float>(a * floored_estimates[i] + b);
                abs_residuals[i] = std::abs(r);
            }
            std::sort(abs_residuals.begin(), abs_residuals.end());
            float mad = abs_residuals[np / 2];
            float huber_delta = constants::kHuberDeltaScale * constants::kMadNormFactor * mad;
            if (huber_delta < constants::kNearZeroSq) break;

            // Weighted least squares with Huber weights.
            double wsum_e = 0, wsum_t = 0, wsum_ee = 0, wsum_et = 0, wsum = 0;
            for (size_t i = 0; i < np; ++i) {
                float r = truths[i] - static_cast<float>(a * floored_estimates[i] + b);
                float ar = std::abs(r);
                float w = (ar <= huber_delta) ? 1.0f : huber_delta / ar;
                double wd = w;
                double e = floored_estimates[i];
                double t = truths[i];
                wsum += wd;
                wsum_e += wd * e;
                wsum_t += wd * t;
                wsum_ee += wd * e * e;
                wsum_et += wd * e * t;
            }
            double wm_e = wsum_e / wsum;
            double wm_t = wsum_t / wsum;
            double wvar = wsum_ee / wsum - wm_e * wm_e;
            double wcov = wsum_et / wsum - wm_e * wm_t;
            if (wvar > constants::kNearZeroSq) {
                a = wcov / wvar;
                b = wm_t - a * wm_e;
            }
        }

        // Clamp affine_a to prevent extreme values.
        calibration_.affine_a = std::clamp(static_cast<float>(a),
            constants::kAffineAMin, constants::kAffineAMax);
        calibration_.affine_b = static_cast<float>(b);

        // Final residual statistics.
        double sum_r2 = 0;
        for (size_t i = 0; i < np; ++i) {
            float predicted = calibration_.affine_a * floored_estimates[i] + calibration_.affine_b;
            float r = truths[i] - predicted;
            abs_residuals[i] = std::abs(r);
            sum_r2 += static_cast<double>(r) * r;
        }
        calibration_.resid_sigma = static_cast<float>(std::sqrt(sum_r2 / np));
        std::sort(abs_residuals.begin(), abs_residuals.end());
        calibration_.resid_q99_dot = abs_residuals[np * constants::kResidQuantilePct / 100];

        double sum_tt = 0;
        for (size_t i = 0; i < np; ++i) {
            sum_tt += static_cast<double>(truths[i]) * truths[i];
        }
        double var_t = sum_tt / np - mean_t * mean_t;
        double denom_corr = std::sqrt(var_e * var_t);
        calibration_.calibration_corr = (denom_corr > constants::kNearZeroSq)
            ? static_cast<float>(cov_et / denom_corr) : 0.0f;

        if (nn_dists_sq.empty()) {
            throw std::runtime_error("Calibration failed: no nearest-neighbor distance samples.");
        }
        std::sort(nn_dists_sq.begin(), nn_dists_sq.end());
        calibration_.median_nn_dist_sq = nn_dists_sq[nn_dists_sq.size() / 2];

        calibration_.flags = 1u;

        // Fit EVT tail on |residual| for risk-to-slack conversion.
        calibration_.evt = evt_crc::fit_gpd(
            abs_residuals.data(), np, constants::kEvtThresholdQ, constants::kEvtMinTail);

        if (nop_samples.empty()) {
            throw std::runtime_error("Calibration failed: no nop samples for EVT.");
        }
        std::sort(nop_samples.begin(), nop_samples.end());
        size_t p95_idx = static_cast<size_t>(
            static_cast<float>(nop_samples.size()) * constants::kEdgeNopQuantile);
        p95_idx = std::min(p95_idx, nop_samples.size() - 1);
        calibration_.evt.nop_p95 = nop_samples[p95_idx];

        if (!calibration_.evt.fitted || calibration_.evt.nop_p95 <= 0.0f) {
            throw std::runtime_error("Calibration failed: EVT-CRC fit did not converge.");
        }

    }
};

}  // namespace cphnsw
