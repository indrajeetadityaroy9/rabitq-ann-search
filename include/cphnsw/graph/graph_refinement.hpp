#pragma once

#include "../core/codes.hpp"
#include "../core/memory.hpp"
#include "../core/adaptive_defaults.hpp"
#include "../distance/fastscan_layout.hpp"
#include "../encoder/rabitq_encoder.hpp"
#include "rabitq_graph.hpp"
#include "neighbor_selection.hpp"
#include "../search/rabitq_search.hpp"
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

template <size_t D, size_t R = 32, size_t BitWidth = 1>
class GraphRefinement {
public:
    using Graph = RaBitQGraph<D, R, BitWidth>;
    static constexpr size_t GraphR = R;

    // ── Shared helper: prune candidates and write to neighbor block ──

    template <typename EncType>
    static void prune_and_write(Graph& graph, const EncType& encoder,
                                NodeId node, std::vector<NeighborCandidate>& candidates,
                                float alpha, float error_tolerance = 0.0f) {
        auto dist_fn = [&](NodeId a, NodeId b) -> float {
            return l2_distance_simd<D>(graph.get_vector(a), graph.get_vector(b));
        };
        auto error_fn = [&](NodeId nid) -> float {
            if (error_tolerance <= 0.0f) return 0.0f;
            const auto& code = graph.get_code(nid);
            float ip_qo = code.ip_quantized_original;
            if (ip_qo < 1e-10f) return 0.0f;
            float ip_qo_sq = ip_qo * ip_qo;
            float variance = (1.0f - ip_qo_sq) / (ip_qo_sq * static_cast<float>(D));
            return error_tolerance * std::sqrt(variance) * code.dist_to_centroid;
        };

        auto selected = select_neighbors_robust_prune(
            std::move(candidates), GraphR, dist_fn, error_fn, alpha);

        write_neighbors(graph, encoder, node, selected);
    }

    // Write selected neighbors into a node's neighbor block, computing FastScan aux data.
    template <typename EncType>
    static void write_neighbors(Graph& graph, const EncType& encoder,
                                NodeId node, const std::vector<NeighborCandidate>& selected) {
        auto& nb = graph.get_neighbors(node);
        nb.count = 0;
        const float* vec_node = graph.get_vector(node);

        for (size_t j = 0; j < selected.size(); ++j) {
            NodeId v = selected[j].id;
            const auto& code_v = graph.get_code(v);
            const float* vec_v = graph.get_vector(v);
            VertexAuxData aux = encoder.compute_neighbor_aux(code_v, vec_node, vec_v);

            if constexpr (BitWidth == 1) {
                nb.set_neighbor(j, v, code_v.signs, aux);
            } else {
                nb.set_neighbor(j, v, code_v.codes, aux);
            }
        }
    }

    // ── Phase 1: Sequential incremental insertion (always quantized) ──

    template <typename EncType>
    static void run_insertion_pass(Graph& graph, const EncType& encoder,
                                   size_t ef_construction, float alpha,
                                   float error_tolerance, float error_epsilon,
                                   bool verbose,
                                   const std::vector<NodeId>& order,
                                   NodeId entry_point) {
        size_t n = order.size();
        size_t search_ef = std::max<size_t>(GraphR, ef_construction);

        auto insert_node = [&](NodeId u, const float* vec_u,
                               std::vector<SearchResult> results) {
            std::vector<NeighborCandidate> candidates;
            candidates.reserve(results.size());
            for (const auto& r : results) {
                if (r.id != u) candidates.push_back({r.id, r.distance});
            }

            prune_and_write(graph, encoder, u, candidates, alpha, error_tolerance);

            auto& nb_u = graph.get_neighbors(u);
            for (size_t j = 0; j < nb_u.count; ++j) {
                NodeId v = nb_u.neighbor_ids[j];
                if (v == INVALID_NODE) continue;

                auto& nb_v = graph.get_neighbors(v);
                std::vector<NeighborCandidate> v_cands;
                v_cands.reserve(nb_v.count + 1);
                const float* vec_v = graph.get_vector(v);
                for (size_t ki = 0; ki < nb_v.count; ++ki) {
                    NodeId w = nb_v.neighbor_ids[ki];
                    if (w == INVALID_NODE) continue;
                    float d = l2_distance_simd<D>(vec_v, graph.get_vector(w));
                    v_cands.push_back({w, d});
                }
                float d_uv = l2_distance_simd<D>(vec_u, vec_v);
                v_cands.push_back({u, d_uv});

                if (v_cands.size() <= GraphR) {
                    write_neighbors(graph, encoder, v, v_cands);
                } else {
                    prune_and_write(graph, encoder, v, v_cands, alpha, error_tolerance);
                }
            }
        };

        TwoLevelVisitationTable visited(n + 1024);
        for (size_t idx = 1; idx < n; ++idx) {
            NodeId u = order[idx];
            const float* vec_u = graph.get_vector(u);
            auto results = QuantizedBuildSearchEngine<D, R, BitWidth>::search(
                vec_u, graph, encoder, search_ef, search_ef, visited,
                error_epsilon, entry_point);
            insert_node(u, vec_u, std::move(results));

            if (verbose && n >= 10 && idx % (n / 10) == 0) {
                printf("[Build]   %zu/%zu (%.0f%%)\n", idx, n, 100.0 * idx / n);
            }
        }
    }

    // ── Phase 2: Parallel refinement pass (always quantized) ──

    struct Spinlock {
        std::atomic_flag flag = ATOMIC_FLAG_INIT;
        void lock() { while (flag.test_and_set(std::memory_order_acquire)); }
        void unlock() { flag.clear(std::memory_order_release); }
    };

    template <typename EncType>
    static void run_refinement_pass(Graph& graph, const EncType& encoder,
                                     size_t ef_construction, float alpha,
                                     float error_tolerance, float error_epsilon,
                                     size_t actual_threads) {
        size_t n = graph.size();
        size_t search_ef = std::max<size_t>(GraphR, ef_construction);

        std::vector<Spinlock> node_locks(n);

        #pragma omp parallel num_threads(actual_threads)
        {
            #pragma omp for schedule(dynamic, 256)
            for (size_t i = 0; i < n; ++i) {
                NodeId u = static_cast<NodeId>(i);
                const float* vec_u = graph.get_vector(u);

                auto results = QuantizedBuildSearchEngine<D, R, BitWidth>::search(
                    vec_u, graph, encoder, search_ef, search_ef, error_epsilon);

                std::vector<NeighborCandidate> candidates;
                candidates.reserve(results.size() + GraphR);
                for (const auto& r : results) {
                    if (r.id != u) candidates.push_back({r.id, r.distance});
                }

                const auto& nb = graph.get_neighbors(u);
                for (size_t j = 0; j < nb.count; ++j) {
                    NodeId v = nb.neighbor_ids[j];
                    if (v == INVALID_NODE || v == u) continue;
                    float d = l2_distance_simd<D>(vec_u, graph.get_vector(v));
                    candidates.push_back({v, d});
                }

                node_locks[u].lock();
                prune_and_write(graph, encoder, u, candidates, alpha, error_tolerance);
                node_locks[u].unlock();
            }
        }
    }

    // ── Phase 3: Add reverse edges ──

    template <typename EncType>
    static void run_reverse_edge_pass(Graph& graph, const EncType& encoder,
                                       float alpha, float error_tolerance,
                                       size_t actual_threads) {
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

        #pragma omp parallel for schedule(dynamic, 256) num_threads(actual_threads)
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

            prune_and_write(graph, encoder, v, all, alpha, error_tolerance);
        }
    }

    // ── Main entry point: adaptive two-pass construction pipeline ──

    template <typename EncType>
    static void optimize_graph_adaptive(Graph& graph, const EncType& encoder,
                                        size_t ef_construction, float error_tolerance,
                                        float error_epsilon, size_t num_threads,
                                        bool verbose) {
        size_t n = graph.size();
        if (n == 0) return;

        size_t actual_threads = num_threads ? num_threads : omp_get_max_threads();

        // Find medoid for entry point
        NodeId entry_point = graph.find_medoid();
        graph.set_entry_point(entry_point);

        // Random insertion order with medoid first
        std::mt19937 rng(42);
        std::vector<NodeId> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), rng);
        for (size_t i = 0; i < n; ++i) {
            if (order[i] == entry_point) {
                std::swap(order[0], order[i]);
                break;
            }
        }

        // Phase 1: Sequential incremental insertion (alpha=1.0)
        if (verbose) printf("[QRG] Phase 1: Insertion pass (alpha=%.2f, n=%zu, R=%zu, ef=%zu)\n",
                           AdaptiveDefaults::ALPHA, n, static_cast<size_t>(GraphR), ef_construction);
        run_insertion_pass(graph, encoder, ef_construction, AdaptiveDefaults::ALPHA,
                          error_tolerance, error_epsilon, verbose, order, entry_point);

        // Phase 1b: Re-insertion with different seed and alpha_pass2 (Vamana two-pass)
        if (verbose) printf("[QRG] Phase 1b: Re-insertion pass (alpha=%.2f)\n",
                           AdaptiveDefaults::ALPHA_PASS2);
        std::mt19937 rng2(137);
        std::shuffle(order.begin(), order.end(), rng2);
        for (size_t i = 0; i < n; ++i) {
            if (order[i] == entry_point) {
                std::swap(order[0], order[i]);
                break;
            }
        }
        run_insertion_pass(graph, encoder, ef_construction, AdaptiveDefaults::ALPHA_PASS2,
                          error_tolerance, error_epsilon, verbose, order, entry_point);

        // Phase 2: Parallel refinement
        if (verbose) printf("[QRG] Phase 2: Refinement pass...\n");
        run_refinement_pass(graph, encoder, ef_construction, AdaptiveDefaults::ALPHA_PASS2,
                           error_tolerance, error_epsilon, actual_threads);

        // Phase 3: Reverse edges
        if (verbose) printf("[QRG] Phase 3: Reverse edge pass...\n");
        run_reverse_edge_pass(graph, encoder, AdaptiveDefaults::ALPHA, error_tolerance,
                             actual_threads);

        if (verbose) printf("[QRG] Done.\n");
    }
};

}  // namespace cphnsw
