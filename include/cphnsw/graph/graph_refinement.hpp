#pragma once

#include "../core/codes.hpp"
#include "../core/memory.hpp"
#include "../core/adaptive_defaults.hpp"
#include "../core/constants.hpp"
#include "../distance/fastscan_layout.hpp"
#include "../encoder/rabitq_encoder.hpp"
#include "rabitq_graph.hpp"
#include "neighbor_selection.hpp"
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdint>
#include <random>
#include <atomic>

#include <omp.h>

namespace cphnsw {
namespace graph_refinement {

struct WorkingNeighbor {
    NodeId id = INVALID_NODE;
    float distance = std::numeric_limits<float>::max();
};


template <size_t D, size_t R, size_t BitWidth, typename EncType>
void prune_and_write(RaBitQGraph<D, R, BitWidth>& graph, const EncType& encoder,
                     NodeId node, std::vector<NeighborCandidate>& candidates,
                     float alpha, float tau, float error_tolerance = 0.0f,
                     float alpha_max = 0.0f) {
    auto dist_fn = [&](NodeId a, NodeId b) -> float {
        return l2_distance_simd<D>(graph.get_vector(a), graph.get_vector(b));
    };
    auto error_fn = [&](NodeId nid) -> float {
        const auto& code = graph.get_code(nid);
        return error_tolerance * code.nop;
    };

    auto selected = select_neighbors_alpha_cng(
        std::move(candidates), R, dist_fn, error_fn, alpha, tau, alpha_max);

    auto& nb = graph.get_neighbors(node);
    nb.count = 0;
    const float* vec_node = graph.get_vector(node);

    alignas(32) float rotated_parent[D];
    encoder.rotate_raw_vector(vec_node, rotated_parent);

    for (size_t j = 0; j < selected.size(); ++j) {
        NodeId v = selected[j].id;
        const float* vec_v = graph.get_vector(v);

        if constexpr (BitWidth == 1) {
            BinaryCodeStorage<EncType::DIMS> pr_code;
            VertexAuxData aux = encoder.compute_neighbor_aux(
                vec_node, vec_v, rotated_parent, pr_code);
            nb.set_neighbor(j, v, pr_code, aux);
        } else {
            auto nbit_result = encoder.compute_neighbor_aux_nbit(
                vec_node, vec_v, rotated_parent);
            nb.set_neighbor(j, v, nbit_result.code, nbit_result.aux);
        }
    }
}


template <size_t D, size_t R, size_t BitWidth>
void init_working_random(
    const RaBitQGraph<D, R, BitWidth>& graph,
    std::vector<std::vector<WorkingNeighbor>>& working,
    size_t actual_threads)
{
    size_t n = graph.size();

    #pragma omp parallel num_threads(actual_threads)
    {
        int tid = omp_get_thread_num();
        std::mt19937 rng(static_cast<uint32_t>(constants::kDefaultGraphSeed + static_cast<uint64_t>(tid)));

        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            NodeId u = static_cast<NodeId>(i);
            const float* vec_u = graph.get_vector(u);
            working[i].clear();

            std::uniform_int_distribution<size_t> dist(0, n - 1);

            std::vector<WorkingNeighbor> candidates;
            // Coupon-collector bound for random init pool
            size_t pool_size = std::min(
                static_cast<size_t>(static_cast<double>(R) *
                    std::ceil(std::log(std::max(static_cast<double>(n) / static_cast<double>(R), 2.0)))),
                n - 1);
            size_t max_attempts = std::min(pool_size + pool_size / 2, n);
            candidates.reserve(pool_size);
            for (size_t j = 0; j < max_attempts && candidates.size() < pool_size; ++j) {
                NodeId v = static_cast<NodeId>(dist(rng));
                if (v == u) continue;
                bool dup = false;
                for (const auto& c : candidates) {
                    if (c.id == v) { dup = true; break; }
                }
                if (dup) continue;
                float d = l2_distance_simd<D>(vec_u, graph.get_vector(v));
                candidates.push_back({v, d});
            }
            std::sort(candidates.begin(), candidates.end(),
                      [](const auto& a, const auto& b) { return a.distance < b.distance; });
            size_t keep = std::min(candidates.size(), static_cast<size_t>(R));
            working[i].assign(candidates.begin(), candidates.begin() + keep);
        }
    }
}


template <size_t D, size_t R, size_t BitWidth>
size_t nndescent_join_pass(
    const RaBitQGraph<D, R, BitWidth>& graph,
    std::vector<std::vector<WorkingNeighbor>>& working,
    std::vector<std::vector<uint8_t>>& new_flags,
    size_t actual_threads,
    size_t omp_chunk)
{
    size_t n = graph.size();

    struct SnapshotEntry {
        NodeId ids[R];
        uint8_t is_new[R];
        uint8_t count;
    };
    std::vector<SnapshotEntry> snapshot(n);

    for (size_t i = 0; i < n; ++i) {
        size_t sz = std::min(working[i].size(), static_cast<size_t>(R));
        snapshot[i].count = static_cast<uint8_t>(sz);
        for (size_t j = 0; j < sz; ++j) {
            snapshot[i].ids[j] = working[i][j].id;
            snapshot[i].is_new[j] = new_flags[i][j];
        }
        std::fill(new_flags[i].begin(), new_flags[i].end(), 0);
    }

    std::vector<std::vector<NodeId>> reverse(n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < snapshot[i].count; ++j) {
            NodeId v = snapshot[i].ids[j];
            if (v != INVALID_NODE && v < n) {
                reverse[v].push_back(static_cast<NodeId>(i));
            }
        }
    }

    std::atomic<size_t> total_updates{0};

    #pragma omp parallel num_threads(actual_threads)
    {
        std::vector<NodeId> candidates;
        candidates.reserve(R * R);

        #pragma omp for schedule(dynamic, omp_chunk)
        for (size_t i = 0; i < n; ++i) {
            NodeId u = static_cast<NodeId>(i);
            const float* vec_u = graph.get_vector(u);
            auto& wl = working[i];

            bool has_new_forward = false;
            for (size_t j = 0; j < snapshot[i].count; ++j) {
                if (snapshot[i].is_new[j]) { has_new_forward = true; break; }
            }

            bool has_new_reverse = false;
            for (NodeId rv : reverse[i]) {
                for (size_t j = 0; j < snapshot[rv].count; ++j) {
                    if (snapshot[rv].is_new[j]) { has_new_reverse = true; break; }
                }
                if (has_new_reverse) break;
            }

            if (!has_new_forward && !has_new_reverse) continue;

            candidates.clear();

            auto is_current_or_self = [&](NodeId w) -> bool {
                if (w == u) return true;
                for (const auto& nb : wl) {
                    if (nb.id == w) return true;
                }
                return false;
            };

            if (has_new_forward) {
                for (size_t j = 0; j < snapshot[i].count; ++j) {
                    if (!snapshot[i].is_new[j]) continue;
                    NodeId v = snapshot[i].ids[j];
                    if (v == INVALID_NODE || v >= n) continue;
                    for (size_t k = 0; k < snapshot[v].count; ++k) {
                        NodeId w = snapshot[v].ids[k];
                        if (w != INVALID_NODE && w < n && !is_current_or_self(w)) {
                            candidates.push_back(w);
                        }
                    }
                }
            }

            for (NodeId rv : reverse[i]) {
                bool rv_has_new = false;
                for (size_t j = 0; j < snapshot[rv].count; ++j) {
                    if (snapshot[rv].is_new[j]) { rv_has_new = true; break; }
                }
                if (!rv_has_new) continue;

                for (size_t k = 0; k < snapshot[rv].count; ++k) {
                    NodeId w = snapshot[rv].ids[k];
                    if (w != INVALID_NODE && w < n && !is_current_or_self(w)) {
                        candidates.push_back(w);
                    }
                }
            }

            if (candidates.empty()) continue;

            std::sort(candidates.begin(), candidates.end());
            candidates.erase(
                std::unique(candidates.begin(), candidates.end()),
                candidates.end());

            size_t local_updates = 0;
            float worst = (wl.size() >= R)
                ? wl.back().distance
                : std::numeric_limits<float>::max();

            for (NodeId w : candidates) {
                float d = l2_distance_simd<D>(vec_u, graph.get_vector(w));
                if (d >= worst && wl.size() >= R) continue;

                if (wl.size() < R) {
                    wl.push_back({w, d});
                    new_flags[i].push_back(1);
                } else {
                    wl.back() = {w, d};
                    new_flags[i].back() = 1;
                }
                for (size_t p = wl.size() - 1; p > 0 && wl[p].distance < wl[p-1].distance; --p) {
                    std::swap(wl[p], wl[p-1]);
                    std::swap(new_flags[i][p], new_flags[i][p-1]);
                }

                worst = (wl.size() >= R) ? wl.back().distance : std::numeric_limits<float>::max();
                local_updates++;
            }

            if (local_updates > 0) {
                total_updates.fetch_add(local_updates, std::memory_order_relaxed);
            }
        }
    }

    return total_updates.load();
}


// Populates GraphStats with neighbor distance distribution, alpha, tau, alpha_max
template <size_t D, size_t R, size_t BitWidth>
GraphStats derive_graph_stats(
    const RaBitQGraph<D, R, BitWidth>& graph,
    const std::vector<std::vector<WorkingNeighbor>>& working,
    size_t sample_size)
{
    GraphStats stats;
    size_t n = working.size();
    if (n == 0) return stats;

    size_t actual_sample = std::min(sample_size, n);
    std::mt19937 rng(static_cast<uint32_t>(constants::kDefaultGraphSeed + 1));
    std::vector<size_t> sample_indices(n);
    std::iota(sample_indices.begin(), sample_indices.end(), 0);
    std::shuffle(sample_indices.begin(), sample_indices.end(), rng);
    sample_indices.resize(actual_sample);

    std::vector<float> neighbor_dists;
    std::vector<float> inter_neighbor_dists;
    std::vector<float> nn_dists;
    neighbor_dists.reserve(actual_sample * R);
    inter_neighbor_dists.reserve(actual_sample * R);
    nn_dists.reserve(actual_sample);

    // Degree statistics (from full working set, not just sample)
    float total_degree = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        total_degree += static_cast<float>(working[i].size());
    }
    stats.avg_degree = total_degree / static_cast<float>(std::max(n, size_t(1)));

    // Inter-neighbor limit: 2·√R, floored at 4 for ≥6 pairs
    size_t inter_limit_val = std::clamp(
        static_cast<size_t>(2.0 * std::sqrt(static_cast<double>(R))),
        size_t(4), static_cast<size_t>(R));

    for (size_t idx : sample_indices) {
        const auto& wl = working[idx];
        for (const auto& nb : wl) {
            if (nb.id != INVALID_NODE) {
                neighbor_dists.push_back(nb.distance);
            }
        }
        if (!wl.empty() && wl[0].id != INVALID_NODE) {
            nn_dists.push_back(wl[0].distance);
        }
        size_t inter_limit = std::min(wl.size(), inter_limit_val);
        for (size_t j = 0; j < inter_limit; ++j) {
            for (size_t k = j + 1; k < inter_limit; ++k) {
                if (wl[j].id == INVALID_NODE || wl[k].id == INVALID_NODE) continue;
                float d = l2_distance_simd<D>(
                    graph.get_vector(wl[j].id), graph.get_vector(wl[k].id));
                inter_neighbor_dists.push_back(d);
            }
        }
    }

    if (neighbor_dists.empty() || inter_neighbor_dists.empty() || nn_dists.empty()) {
        stats.alpha = 1.0f;
        stats.tau = 0.0f;
        stats.alpha_max = 4.0f;
        return stats;
    }

    std::sort(neighbor_dists.begin(), neighbor_dists.end());
    std::sort(inter_neighbor_dists.begin(), inter_neighbor_dists.end());
    std::sort(nn_dists.begin(), nn_dists.end());

    // Neighbor distance statistics
    size_t nd_n = neighbor_dists.size();
    float neighbor_dist_median = neighbor_dists[nd_n / 2];
    float nd_q1 = neighbor_dists[nd_n / 4];
    float nd_q3 = neighbor_dists[3 * nd_n / 4];
    float neighbor_q3_over_q1 = (nd_q1 > constants::norm_epsilon(D))
        ? nd_q3 / nd_q1 : 2.0f;

    // Neighbor distance CV
    float nd_mean = 0.0f;
    for (float d : neighbor_dists) nd_mean += d;
    nd_mean /= static_cast<float>(nd_n);
    float nd_var = 0.0f;
    for (float d : neighbor_dists) nd_var += (d - nd_mean) * (d - nd_mean);
    nd_var /= static_cast<float>(nd_n);
    float neighbor_dist_cv = (nd_mean > constants::norm_epsilon(D))
        ? std::sqrt(nd_var) / nd_mean : 0.2f;

    // NN distance MAD-sigma (robust standard deviation)
    float nn_median = nn_dists[nn_dists.size() / 2];
    std::vector<float> nn_abs_devs(nn_dists.size());
    for (size_t i = 0; i < nn_dists.size(); ++i)
        nn_abs_devs[i] = std::abs(nn_dists[i] - nn_median);
    std::sort(nn_abs_devs.begin(), nn_abs_devs.end());
    float nn_mad = nn_abs_devs[nn_abs_devs.size() / 2];
    float nn_dist_mad_sigma = constants::kMadNormFactor * nn_mad;

    // Alpha: ratio of median neighbor dist to inter-neighbor density
    float d_inter = inter_neighbor_dists[inter_neighbor_dists.size() / 4];

    if (d_inter < constants::norm_epsilon(D)) {
        stats.alpha = 1.0f + neighbor_dist_cv;
    } else {
        stats.alpha = neighbor_dist_median / d_inter;
    }

    // Alpha clamp: Q3/Q1 ratio of neighbor distances (inherent spread),
    // capped at 5.0 (pruning at 5x nearest distance is effectively no pruning)
    stats.alpha_max = std::min(neighbor_q3_over_q1, 5.0f);
    stats.alpha = std::clamp(stats.alpha, 1.0f, stats.alpha_max);

    // Select_neighbors alpha_max: 2·alpha (sqrt scaling cannot more than double base)
    stats.alpha_max = std::max(stats.alpha_max, 2.0f * stats.alpha);

    // Tau: one robust standard deviation of NN distances
    stats.tau = nn_dist_mad_sigma;

    return stats;
}


template <size_t D, size_t R, size_t BitWidth, typename EncType>
void run_reverse_edge_pass(RaBitQGraph<D, R, BitWidth>& graph, const EncType& encoder,
                           float alpha, float tau, float error_tolerance,
                           size_t actual_threads, size_t omp_chunk,
                           float alpha_max = 0.0f) {
    size_t n = graph.size();

    std::vector<std::vector<NeighborCandidate>> reverse_cands(n);
    for (size_t u = 0; u < n; ++u) {
        const auto& nb = graph.get_neighbors(static_cast<NodeId>(u));
        const float* vec_u = graph.get_vector(static_cast<NodeId>(u));
        for (size_t j = 0; j < nb.count; ++j) {
            NodeId v = nb.neighbor_ids[j];
            if (v == INVALID_NODE) continue;
            float d = l2_distance_simd<D>(vec_u, graph.get_vector(v));
            reverse_cands[v].push_back({static_cast<NodeId>(u), d});
        }
    }

    #pragma omp parallel for schedule(dynamic, omp_chunk) num_threads(actual_threads)
    for (size_t i = 0; i < n; ++i) {
        NodeId v = static_cast<NodeId>(i);
        if (reverse_cands[v].empty()) continue;

        const auto& nb = graph.get_neighbors(v);
        std::vector<NeighborCandidate> all;
        all.reserve(nb.count + reverse_cands[v].size());

        const float* vec_v = graph.get_vector(v);
        for (size_t j = 0; j < nb.count; ++j) {
            NodeId w = nb.neighbor_ids[j];
            if (w == INVALID_NODE) continue;
            float d = l2_distance_simd<D>(vec_v, graph.get_vector(w));
            all.push_back({w, d});
        }

        for (const auto& cand : reverse_cands[v]) {
            if (cand.id == v) continue;
            all.push_back(cand);
        }

        prune_and_write<D, R, BitWidth>(graph, encoder, v, all, alpha, tau, error_tolerance, alpha_max);
    }
}


template <size_t D, size_t R, size_t BitWidth>
struct GraphOptimizeResult {
    typename RaBitQGraph<D, R, BitWidth>::Permutation perm;
    GraphStats stats;
};

template <size_t D, size_t R, size_t BitWidth, typename EncType>
GraphOptimizeResult<D, R, BitWidth>
optimize_graph_adaptive(RaBitQGraph<D, R, BitWidth>& graph, const EncType& encoder) {
    using Permutation = typename RaBitQGraph<D, R, BitWidth>::Permutation;
    size_t n = graph.size();
    if (n == 0) return {Permutation{}, GraphStats{}};

    size_t actual_threads = static_cast<size_t>(omp_get_max_threads());
    // Error tolerance ∝ 1/√D: RaBitQ per-coordinate quantization error is O(1/√D)
    // by CLT averaging over D independent coordinate errors.
    float error_tolerance = 1.0f / std::sqrt(static_cast<float>(D));
    size_t omp_chunk = adaptive_defaults::omp_chunk_size(n, actual_threads);

    auto centroid = graph.compute_centroid();
    NodeId entry_point = graph.find_nearest_to_centroid(centroid);
    graph.set_entry_point(entry_point);

    std::vector<std::vector<WorkingNeighbor>> working(n);
    std::vector<std::vector<uint8_t>> new_flags(n);

    init_working_random<D, R, BitWidth>(graph, working, actual_threads);

    for (size_t i = 0; i < n; ++i) {
        new_flags[i].assign(working[i].size(), 1);
    }

    // Adaptive NNDescent convergence: derive parameters from initial rounds
    size_t total_edges = std::max(n * R, size_t(1));

    // Phase 1: Run 2 unconditional rounds to measure convergence behavior
    size_t updates_0 = nndescent_join_pass<D, R, BitWidth>(
        graph, working, new_flags, actual_threads, omp_chunk);
    float rate_0 = static_cast<float>(updates_0) / static_cast<float>(total_edges);

    size_t updates_1 = nndescent_join_pass<D, R, BitWidth>(
        graph, working, new_flags, actual_threads, omp_chunk);
    float rate_1 = static_cast<float>(updates_1) / static_cast<float>(total_edges);

    // EMA alpha from observed decay: faster decay → more responsive smoothing.
    // Bounds [0.2, 0.8] define EMA window range [1.5, 9 rounds].
    float decay_ratio = (rate_0 > constants::eps::kSmall) ? rate_1 / rate_0 : 0.5f;
    float ema_alpha = std::clamp(1.0f - decay_ratio, 0.2f, 0.8f);

    // Converge threshold: fewer than 1 expected update per edge relative to initial
    float converge_rate = std::max(
        rate_0 / static_cast<float>(total_edges),
        1.0f / static_cast<float>(total_edges));

    // Min rounds from geometric decay extrapolation
    size_t min_rounds;
    if (decay_ratio > 0.0f && decay_ratio < 1.0f && rate_0 > converge_rate) {
        min_rounds = static_cast<size_t>(std::ceil(
            std::log(converge_rate / rate_0) / std::log(decay_ratio)));
        min_rounds = std::clamp(min_rounds, size_t(2),
            static_cast<size_t>(std::sqrt(std::log2(
                static_cast<float>(std::max(n, size_t(64)))))));
    } else {
        min_rounds = 2;
    }

    // Hard cap: 3x expected convergence time
    size_t hard_cap = std::clamp(min_rounds * 3, size_t(10),
        std::min(n, std::max(size_t(500), isqrt(n))));

    // Phase 2: Continue with adaptive parameters
    float ema_rate = ema_alpha * rate_1 + (1.0f - ema_alpha) * rate_0;
    size_t total_rounds = 2;

    for (size_t round = 2; round < hard_cap; ++round) {
        size_t updates = nndescent_join_pass<D, R, BitWidth>(
            graph, working, new_flags, actual_threads, omp_chunk);
        float rate = static_cast<float>(updates) / static_cast<float>(total_edges);

        ema_rate = ema_alpha * rate + (1.0f - ema_alpha) * ema_rate;
        total_rounds = round + 1;

        if (round >= min_rounds && ema_rate < converge_rate) break;
    }

    // Compute full graph statistics from working neighbor lists
    size_t alpha_sample = static_cast<size_t>(std::sqrt(static_cast<double>(n)));
    GraphStats stats = derive_graph_stats<D, R, BitWidth>(graph, working, alpha_sample);

    float alpha = stats.alpha;
    float tau = stats.tau;
    float alpha_max = stats.alpha_max;

    #pragma omp parallel for schedule(dynamic, omp_chunk) num_threads(actual_threads)
    for (size_t i = 0; i < n; ++i) {
        NodeId u = static_cast<NodeId>(i);
        std::vector<NeighborCandidate> candidates;
        candidates.reserve(working[i].size());
        for (const auto& nb : working[i]) {
            if (nb.id != INVALID_NODE) {
                candidates.push_back({nb.id, nb.distance});
            }
        }
        prune_and_write<D, R, BitWidth>(graph, encoder, u, candidates,
            alpha, tau, error_tolerance, alpha_max);
    }

    working.clear();
    working.shrink_to_fit();
    new_flags.clear();
    new_flags.shrink_to_fit();

    run_reverse_edge_pass<D, R, BitWidth>(graph, encoder, alpha, tau,
        error_tolerance, actual_threads, omp_chunk, alpha_max);

    NodeId hub = graph.find_hub_entry(centroid);
    graph.set_entry_point(hub);

    auto perm = graph.reorder_bfs(hub);

    return {perm, stats};
}

}
}
