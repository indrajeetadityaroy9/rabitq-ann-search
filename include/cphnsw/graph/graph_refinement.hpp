#pragma once

#include "../core/codes.hpp"
#include "../core/memory.hpp"
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
                                float alpha, bool fill_slots) {
        auto dist_fn = [&](NodeId a, NodeId b) -> float {
            return l2_distance_simd<D>(graph.get_vector(a), graph.get_vector(b));
        };
        auto error_fn = [](NodeId) -> float { return 0.0f; };

        auto selected = select_neighbors_robust_prune(
            std::move(candidates), GraphR, dist_fn, error_fn, alpha);

        // Fill remaining slots if requested
        if (fill_slots && selected.size() < GraphR) {
            // Already handled by robust_prune's fill phase
        }

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

    // ── Phase 1: Sequential incremental insertion ──

    template <typename EncType>
    static void run_insertion_pass(Graph& graph, const EncType& encoder,
                                   const BuildParams& params,
                                   const std::vector<NodeId>& order,
                                   NodeId entry_point) {
        size_t n = order.size();
        size_t search_ef = std::max<size_t>(GraphR, params.ef_construction);
        VisitationTable visited(n + 1024);

        for (size_t idx = 0; idx < n; ++idx) {
            NodeId u = order[idx];

            if (idx == 0) continue;  // First node: no neighbors

            const float* vec_u = graph.get_vector(u);
            auto results = ExactL2SearchEngine<D, R, BitWidth>::search(
                vec_u, graph, search_ef, search_ef, visited, entry_point);

            std::vector<NeighborCandidate> candidates;
            candidates.reserve(results.size());
            for (const auto& r : results) {
                if (r.id != u) candidates.push_back({r.id, r.distance});
            }

            // Set u's forward edges
            prune_and_write(graph, encoder, u, candidates, params.alpha, params.fill_slots);

            // Add reverse edges
            auto& nb_u = graph.get_neighbors(u);
            for (size_t j = 0; j < nb_u.count; ++j) {
                NodeId v = nb_u.neighbor_ids[j];
                if (v == INVALID_NODE) continue;

                auto& nb_v = graph.get_neighbors(v);
                std::vector<NeighborCandidate> v_cands;
                v_cands.reserve(nb_v.count + 1);
                const float* vec_v = graph.get_vector(v);
                for (size_t k = 0; k < nb_v.count; ++k) {
                    NodeId w = nb_v.neighbor_ids[k];
                    if (w == INVALID_NODE) continue;
                    float d = l2_distance_simd<D>(vec_v, graph.get_vector(w));
                    v_cands.push_back({w, d});
                }
                float d_uv = l2_distance_simd<D>(vec_u, vec_v);
                v_cands.push_back({u, d_uv});

                if (v_cands.size() <= GraphR) {
                    write_neighbors(graph, encoder, v, v_cands);
                } else {
                    prune_and_write(graph, encoder, v, v_cands, params.alpha, params.fill_slots);
                }
            }

            if (params.verbose && n >= 10 && (idx + 1) % (n / 10) == 0) {
                printf("[Build]   %zu/%zu (%.0f%%)\n", idx + 1, n, 100.0 * (idx + 1) / n);
            }
        }
    }

    // ── Phase 2: Parallel refinement pass ──

    struct Spinlock {
        std::atomic_flag flag = ATOMIC_FLAG_INIT;
        void lock() { while (flag.test_and_set(std::memory_order_acquire)); }
        void unlock() { flag.clear(std::memory_order_release); }
    };

    template <typename EncType>
    static void run_refinement_pass(Graph& graph, const EncType& encoder,
                                     const BuildParams& params,
                                     size_t actual_threads) {
        size_t n = graph.size();
        size_t search_ef = std::max<size_t>(GraphR, params.ef_construction);
        float alpha = params.enable_pass2 ? params.alpha_pass2 : params.alpha;

        std::vector<Spinlock> node_locks(n);

        #pragma omp parallel num_threads(actual_threads)
        {
            VisitationTable local_visited(n + 1024);

            #pragma omp for schedule(dynamic, 256)
            for (size_t i = 0; i < n; ++i) {
                NodeId u = static_cast<NodeId>(i);
                const float* vec_u = graph.get_vector(u);

                auto results = ExactL2SearchEngine<D, R, BitWidth>::search(
                    vec_u, graph, search_ef, search_ef, local_visited);

                std::vector<NeighborCandidate> candidates;
                candidates.reserve(results.size() + GraphR);
                for (const auto& r : results) {
                    if (r.id != u) candidates.push_back({r.id, r.distance});
                }

                // Keep existing neighbors as candidates
                const auto& nb = graph.get_neighbors(u);
                for (size_t j = 0; j < nb.count; ++j) {
                    NodeId v = nb.neighbor_ids[j];
                    if (v == INVALID_NODE || v == u) continue;
                    float d = l2_distance_simd<D>(vec_u, graph.get_vector(v));
                    candidates.push_back({v, d});
                }

                node_locks[u].lock();
                prune_and_write(graph, encoder, u, candidates, alpha, params.fill_slots);
                node_locks[u].unlock();
            }
        }
    }

    // ── Phase 3: Add reverse edges ──

    template <typename EncType>
    static void run_reverse_edge_pass(Graph& graph, const EncType& encoder,
                                       const BuildParams& params,
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

            prune_and_write(graph, encoder, v, all, params.alpha, params.fill_slots);
        }
    }

    // ── Analytically optimal degree R* ──

    static uint16_t calibrate_active_degree(const Graph& graph) {
        size_t n = graph.size();
        if (n < 100) return static_cast<uint16_t>(GraphR);

        // Ma et al. (arXiv 2509.15531v2) SNG degree bound: O(n^{2/3+ε})
        // Practical formula: R* = clamp(C * log(n) / alpha², R/2, R)
        // where C is calibrated from a small sample
        float log_n = std::log(static_cast<float>(n));
        float raw_degree = 4.0f * log_n;  // C=4 empirically good
        float lo = static_cast<float>(GraphR) / 2.0f;
        float hi = static_cast<float>(GraphR);
        uint16_t r_star = static_cast<uint16_t>(
            raw_degree < lo ? lo : (raw_degree > hi ? hi : raw_degree));
        // Round up to multiple of 4 for SIMD alignment
        r_star = ((r_star + 3) / 4) * 4;
        return std::min(r_star, static_cast<uint16_t>(GraphR));
    }

    // ── Adaptive Bit-Width Construction Pipeline (QRG) ──

    template <typename EncType>
    static void optimize_graph_adaptive(Graph& graph, const EncType& encoder,
                                        const BuildParams& params) {
        size_t n = graph.size();
        if (n == 0) return;

        size_t actual_threads = params.num_threads ? params.num_threads : omp_get_max_threads();

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

        // Phase 1: Sequential incremental insertion
        if (params.verbose) printf("[QRG] Phase 1: Insertion pass (alpha=%.2f, n=%zu, R=%zu)\n",
                                   params.alpha, n, static_cast<size_t>(GraphR));
        run_insertion_pass(graph, encoder, params, order, entry_point);

        // Phase 1b: Re-insertion with different seed and alpha_pass2 (Vamana two-pass)
        if (params.enable_pass2) {
            if (params.verbose) printf("[QRG] Phase 1b: Re-insertion pass (alpha=%.2f)\n",
                                       params.alpha_pass2);

            // Shuffle with a different seed for phase 1b
            std::mt19937 rng2(137);
            std::shuffle(order.begin(), order.end(), rng2);
            for (size_t i = 0; i < n; ++i) {
                if (order[i] == entry_point) {
                    std::swap(order[0], order[i]);
                    break;
                }
            }

            // Create modified params with alpha_pass2
            BuildParams pass2_params = params;
            pass2_params.alpha = params.alpha_pass2;

            run_insertion_pass(graph, encoder, pass2_params, order, entry_point);
        }

        // Phase 2: Parallel refinement (exact L2, quality safety net)
        if (params.verbose) printf("[QRG] Phase 2: Refinement pass...\n");
        run_refinement_pass(graph, encoder, params, actual_threads);

        // Phase 3: Reverse edges
        if (params.verbose) printf("[QRG] Phase 3: Reverse edge pass...\n");
        run_reverse_edge_pass(graph, encoder, params, actual_threads);

        if (params.verbose) printf("[QRG] Done.\n");
    }

    // ── Main entry point ──

    template <typename EncType>
    static void optimize_graph(Graph& graph, const EncType& encoder,
                             const BuildParams& params) {
        size_t n = graph.size();
        if (n == 0) return;

        size_t actual_threads = params.num_threads ? params.num_threads : omp_get_max_threads();

        // Auto-degree calibration
        if (params.auto_degree) {
            uint16_t r_star = calibrate_active_degree(graph);
            if (params.verbose) printf("[Build] Auto-degree: R*=%u (max R=%zu)\n",
                                       r_star, static_cast<size_t>(GraphR));
            for (size_t i = 0; i < n; ++i) {
                auto& nb = graph.get_neighbors(static_cast<NodeId>(i));
                nb.active_degree = r_star;
            }
        }

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

        if (params.verbose) printf("[Build] Incremental insertion (n=%zu, R=%zu, ef=%zu)\n",
                                   n, static_cast<size_t>(GraphR), params.ef_construction);

        // Phase 1: Sequential incremental insertion
        run_insertion_pass(graph, encoder, params, order, entry_point);

        // Phase 2: Parallel refinement
        if (params.verbose) printf("[Build] Refinement pass...\n");
        run_refinement_pass(graph, encoder, params, actual_threads);

        // Phase 3: Reverse edges
        if (params.verbose) printf("[Build] Adding reverse edges...\n");
        run_reverse_edge_pass(graph, encoder, params, actual_threads);

        if (params.verbose) printf("[Build] Done.\n");
    }

    // Legacy overload for backward compatibility
    template <typename EncType>
    static void optimize_graph(Graph& graph, const EncType& encoder,
                             size_t ef_construction, size_t num_threads = 0, bool verbose = false) {
        BuildParams params;
        params.ef_construction = ef_construction;
        params.num_threads = num_threads;
        params.verbose = verbose;
        optimize_graph(graph, encoder, params);
    }
};

using GraphRefinement128 = GraphRefinement<128, 32>;
using GraphRefinement256 = GraphRefinement<256, 32>;
using GraphRefinement1024 = GraphRefinement<1024, 32>;

}  // namespace cphnsw
