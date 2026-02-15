#pragma once

#include "../core/codes.hpp"
#include "../core/memory.hpp"
#include "../core/adaptive_defaults.hpp"
#include "../distance/fastscan_layout.hpp"
#include "../encoder/rabitq_encoder.hpp"
#include "rabitq_graph.hpp"
#include "neighbor_selection.hpp"
#include "../api/params.hpp"
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstring>
#include <random>
#include <atomic>

#include "../core/omp_compat.hpp"

namespace cphnsw {
namespace graph_refinement {

struct WorkingNeighbor {
    NodeId id = INVALID_NODE;
    float distance = std::numeric_limits<float>::max();
};


template <size_t D, size_t R, size_t BitWidth, typename EncType>
void write_neighbors(RaBitQGraph<D, R, BitWidth>& graph, const EncType& encoder,
                     NodeId node, const std::vector<NeighborCandidate>& selected) {
    auto& nb = graph.get_neighbors(node);
    nb.count = 0;
    const float* vec_node = graph.get_vector(node);

    alignas(32) float rotated_parent[D];
    encoder.rotate_raw_vector(vec_node, rotated_parent);

    for (size_t j = 0; j < selected.size(); ++j) {
        NodeId v = selected[j].id;
        const auto& code_v = graph.get_code(v);
        const float* vec_v = graph.get_vector(v);
        BinaryCodeStorage<EncType::DIMS> pr_code;
        VertexAuxData aux = encoder.compute_neighbor_aux(
            code_v, vec_node, vec_v, rotated_parent, &pr_code);

        if constexpr (BitWidth == 1) {
            nb.set_neighbor(j, v, pr_code, aux);
        } else {
            nb.set_neighbor_with_pr_msb(j, v, pr_code, code_v.codes, aux);
        }
    }
}


template <size_t D, size_t R, size_t BitWidth, typename EncType>
void prune_and_write(RaBitQGraph<D, R, BitWidth>& graph, const EncType& encoder,
                     NodeId node, std::vector<NeighborCandidate>& candidates,
                     float alpha, float error_tolerance = 0.0f) {
    auto dist_fn = [&](NodeId a, NodeId b) -> float {
        return l2_distance_simd<D>(graph.get_vector(a), graph.get_vector(b));
    };
    auto error_fn = [&](NodeId nid) -> float {
        if (error_tolerance <= 0.0f) return 0.0f;
        const auto& code = graph.get_code(nid);
        float ip_qo = code.ip_quantized_original;
        if (ip_qo < adaptive_defaults::norm_epsilon(D)) return 0.0f;
        float ip_qo_sq = ip_qo * ip_qo;
        float Df = static_cast<float>(D);
        float variance = (1.0f - ip_qo_sq) * (Df - 1.0f) / (ip_qo_sq * Df);
        return error_tolerance * std::sqrt(variance) * code.dist_to_centroid;
    };

    auto selected = select_neighbors_alpha_cg(
        std::move(candidates), R, dist_fn, error_fn, alpha);

    write_neighbors<D, R, BitWidth>(graph, encoder, node, selected);
}


template <size_t D, size_t R, size_t BitWidth>
void init_working_random(
    const RaBitQGraph<D, R, BitWidth>& graph,
    std::vector<std::vector<WorkingNeighbor>>& working,
    size_t actual_threads, uint64_t seed)
{
    size_t n = graph.size();

    #pragma omp parallel num_threads(actual_threads)
    {
        int tid = omp_get_thread_num();
        std::mt19937 rng(seed + tid);

        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            NodeId u = static_cast<NodeId>(i);
            const float* vec_u = graph.get_vector(u);
            working[i].clear();

            std::uniform_int_distribution<size_t> dist(0, n - 1);

            std::vector<WorkingNeighbor> candidates;
            size_t pool_size = adaptive_defaults::random_init_pool(R, n);
            size_t max_attempts = adaptive_defaults::random_init_attempts(R, n);
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
                    for (size_t p = wl.size() - 1; p > 0 && wl[p].distance < wl[p-1].distance; --p) {
                        std::swap(wl[p], wl[p-1]);
                        std::swap(new_flags[i][p], new_flags[i][p-1]);
                    }
                } else {
                    wl.back() = {w, d};
                    new_flags[i].back() = 1;
                    for (size_t p = wl.size() - 1; p > 0 && wl[p].distance < wl[p-1].distance; --p) {
                        std::swap(wl[p], wl[p-1]);
                        std::swap(new_flags[i][p], new_flags[i][p-1]);
                    }
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


template <size_t D, size_t R, size_t BitWidth>
float derive_alpha_from_working(
    const RaBitQGraph<D, R, BitWidth>& graph,
    const std::vector<std::vector<WorkingNeighbor>>& working,
    size_t sample_size = 0, uint64_t seed = 42)
{
    size_t n = working.size();
    if (n == 0) return adaptive_defaults::alpha_default(D);
    if (sample_size == 0) sample_size = adaptive_defaults::alpha_sample_size(n);

    size_t actual_sample = std::min(sample_size, n);
    std::mt19937 rng(seed);
    std::vector<size_t> sample_indices(n);
    std::iota(sample_indices.begin(), sample_indices.end(), 0);
    std::shuffle(sample_indices.begin(), sample_indices.end(), rng);
    sample_indices.resize(actual_sample);

    std::vector<float> neighbor_dists;
    std::vector<float> inter_neighbor_dists;
    neighbor_dists.reserve(actual_sample * R);
    inter_neighbor_dists.reserve(actual_sample * R);

    for (size_t idx : sample_indices) {
        const auto& wl = working[idx];
        for (const auto& nb : wl) {
            if (nb.id != INVALID_NODE) {
                neighbor_dists.push_back(nb.distance);
            }
        }
        size_t inter_limit = std::min(wl.size(), adaptive_defaults::alpha_inter_limit(R));
        for (size_t j = 0; j < inter_limit; ++j) {
            for (size_t k = j + 1; k < inter_limit; ++k) {
                if (wl[j].id == INVALID_NODE || wl[k].id == INVALID_NODE) continue;
                float d = l2_distance_simd<D>(
                    graph.get_vector(wl[j].id), graph.get_vector(wl[k].id));
                inter_neighbor_dists.push_back(d);
            }
        }
    }

    if (neighbor_dists.empty() || inter_neighbor_dists.empty()) return adaptive_defaults::alpha_default(D);

    std::sort(neighbor_dists.begin(), neighbor_dists.end());
    std::sort(inter_neighbor_dists.begin(), inter_neighbor_dists.end());

    float d_med = neighbor_dists[neighbor_dists.size() / 2];
    float d_inter = inter_neighbor_dists[inter_neighbor_dists.size() / adaptive_defaults::alpha_percentile_divisor()];

    if (d_inter < adaptive_defaults::norm_epsilon(D)) return adaptive_defaults::alpha_default(D);

    float alpha = std::max(1.0f, d_med / d_inter);
    alpha = std::min(alpha, adaptive_defaults::alpha_ceiling());
    if (alpha < adaptive_defaults::alpha_floor_threshold()) alpha = adaptive_defaults::alpha_default(D);
    return alpha;
}


template <size_t D, size_t R, size_t BitWidth, typename EncType>
void run_reverse_edge_pass(RaBitQGraph<D, R, BitWidth>& graph, const EncType& encoder,
                           float alpha, float error_tolerance,
                           size_t actual_threads, size_t omp_chunk) {
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

        prune_and_write<D, R, BitWidth>(graph, encoder, v, all, alpha, error_tolerance);
    }
}


template <size_t D, size_t R, size_t BitWidth, typename EncType>
void optimize_graph_adaptive(RaBitQGraph<D, R, BitWidth>& graph, const EncType& encoder,
                             size_t num_threads, bool verbose, uint64_t seed = 42) {
    size_t n = graph.size();
    if (n == 0) return;

    size_t actual_threads = num_threads ? num_threads : omp_get_max_threads();
    float error_tolerance = adaptive_defaults::error_tolerance(D);
    size_t omp_chunk = adaptive_defaults::omp_chunk_size(n, actual_threads);

    float delta = adaptive_defaults::nndescent_delta(n, R);
    size_t delta_threshold = std::max<size_t>(1,
        static_cast<size_t>(delta * static_cast<float>(n) * static_cast<float>(R)));
    size_t max_iters = adaptive_defaults::nndescent_max_iters(n);

    NodeId entry_point = graph.find_medoid();
    graph.set_entry_point(entry_point);

    std::vector<std::vector<WorkingNeighbor>> working(n);
    std::vector<std::vector<uint8_t>> new_flags(n);

    if (verbose) {
        printf("event=graph_opt_start nodes=%zu degree=%zu threads=%zu delta=%.4f threshold=%zu max_iters=%zu\n",
               n, R, actual_threads, delta, delta_threshold, max_iters);
    }
    init_working_random<D, R, BitWidth>(graph, working, actual_threads, seed);

    for (size_t i = 0; i < n; ++i) {
        new_flags[i].assign(working[i].size(), 1);
    }

    size_t converged_round = max_iters;
    for (size_t round = 0; round < max_iters; ++round) {
        size_t updates = nndescent_join_pass<D, R, BitWidth>(graph, working, new_flags, actual_threads, omp_chunk);

        if (updates <= delta_threshold) {
            converged_round = round + 1;
            break;
        }
    }

    float alpha = derive_alpha_from_working<D, R, BitWidth>(graph, working, 0, seed);

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
        prune_and_write<D, R, BitWidth>(graph, encoder, u, candidates, alpha, error_tolerance);
    }

    { std::vector<std::vector<WorkingNeighbor>>().swap(working); }
    { std::vector<std::vector<uint8_t>>().swap(new_flags); }

    run_reverse_edge_pass<D, R, BitWidth>(graph, encoder, alpha, error_tolerance, actual_threads, omp_chunk);

    NodeId hub = graph.find_hub_entry();
    graph.set_entry_point(hub);
    if (verbose) {
        printf("event=graph_opt_done converged_round=%zu alpha=%.3f hub=%u hub_degree=%zu avg_degree=%.3f max_degree=%zu\n",
               converged_round, alpha, hub, graph.neighbor_count(hub),
               graph.average_degree(), graph.max_degree());
    }
}

}  // namespace graph_refinement
}  // namespace cphnsw
