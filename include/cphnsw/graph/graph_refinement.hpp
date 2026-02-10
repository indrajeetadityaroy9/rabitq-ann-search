#pragma once

#include "../core/codes.hpp"
#include "../core/memory.hpp"
#include "../distance/fastscan_layout.hpp"
#include "../encoder/rabitq_encoder.hpp"
#include "rabitq_graph.hpp"
#include "neighbor_selection.hpp"
#include "../search/rabitq_search.hpp"
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstring>
#include <random>
#include <atomic>

#ifdef _OPENMP
#include <omp.h>
#else
inline int omp_get_thread_num() { return 0; }
inline int omp_get_max_threads() { return 1; }
#endif

namespace cphnsw {

template <size_t D, size_t R = 32, size_t BitWidth = 1>
class GraphRefinement {
public:
    using Graph = RaBitQGraph<D, R, BitWidth>;

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

    // ── Main entry point ──────────────────────────────────────────────────
    //
    // Vamana-style incremental insertion:
    //  1. Insert nodes one at a time, each searching the growing graph
    //  2. Bidirectional edges with diversity pruning + slot filling
    //  3. Parallel refinement pass to improve edge quality
    //  4. Final reverse edges for symmetry
    //
    // This produces highly navigable graphs because:
    //  - Early nodes form a high-quality backbone
    //  - Each new node benefits from the existing graph quality
    //  - Quality propagates incrementally (warm start)

    template <typename EncType>
    static void optimize_graph(Graph& graph, const EncType& encoder,
                             size_t ef_construction, size_t num_threads = 0, bool verbose = false) {
        size_t n = graph.size();
        if (n == 0) return;

        size_t actual_threads = num_threads ? num_threads : omp_get_max_threads();
        constexpr size_t GraphR = Graph::DEGREE;

        // Find medoid for entry point
        NodeId entry_point = graph.find_medoid();
        graph.set_entry_point(entry_point);

        // Random insertion order with medoid first
        std::mt19937 rng(42);
        std::vector<NodeId> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), rng);
        // Move medoid to front
        for (size_t i = 0; i < n; ++i) {
            if (order[i] == entry_point) {
                std::swap(order[0], order[i]);
                break;
            }
        }

        if (verbose) printf("[Build] Incremental insertion (n=%zu, R=%zu, ef=%zu)\n",
                           n, GraphR, ef_construction);

        size_t search_ef = std::max<size_t>(GraphR, ef_construction);
        VisitationTable visited(n + 1024);

        auto dist_fn = [&](NodeId a, NodeId b) -> float {
            return l2_distance_simd<D>(graph.get_vector(a), graph.get_vector(b));
        };

        // ── Phase 1: Sequential incremental insertion ──
        for (size_t idx = 0; idx < n; ++idx) {
            NodeId u = order[idx];
            const float* vec_u = graph.get_vector(u);

            if (idx == 0) {
                // First node: no neighbors to find
                continue;
            }

            // Search existing graph for nearest neighbors of u
            auto results = ExactL2SearchEngine<D, R, BitWidth>::search(
                vec_u, graph, search_ef, search_ef, visited, entry_point);

            // Select R neighbors with diversity pruning + slot filling
            std::vector<NeighborCandidate> candidates;
            candidates.reserve(results.size());
            for (const auto& r : results) {
                if (r.id != u) candidates.push_back({r.id, r.distance});
            }

            auto selected = select_neighbors_heuristic(std::move(candidates), GraphR, dist_fn);

            // Set u's forward edges
            write_neighbors(graph, encoder, u, selected);

            // Add reverse edges: for each u→v, add u as candidate for v
            for (const auto& sel : selected) {
                NodeId v = sel.id;
                auto& nb_v = graph.get_neighbors(v);

                // Collect v's existing neighbors + u
                std::vector<NeighborCandidate> v_cands;
                v_cands.reserve(nb_v.count + 1);
                const float* vec_v = graph.get_vector(v);
                for (size_t k = 0; k < nb_v.count; ++k) {
                    NodeId w = nb_v.neighbor_ids[k];
                    if (w == INVALID_NODE) continue;
                    float d = l2_distance_simd<D>(vec_v, graph.get_vector(w));
                    v_cands.push_back({w, d});
                }
                v_cands.push_back({u, sel.distance});

                if (v_cands.size() <= GraphR) {
                    // Room available — write all without pruning
                    write_neighbors(graph, encoder, v, v_cands);
                } else {
                    // Re-prune to R with slot filling
                    auto v_selected = select_neighbors_heuristic(
                        std::move(v_cands), GraphR, dist_fn);
                    write_neighbors(graph, encoder, v, v_selected);
                }
            }

            if (verbose && n >= 10 && (idx + 1) % (n / 10) == 0) {
                printf("[Build]   %zu/%zu (%.0f%%)\n", idx + 1, n, 100.0 * (idx + 1) / n);
            }
        }

        // ── Phase 2: Parallel refinement pass ──
        // Now that all nodes are inserted, one parallel pass to improve edge quality.
        // Each node searches the complete graph for better neighbors.
        if (verbose) printf("[Build] Refinement pass...\n");

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

                auto local_dist_fn = [&](NodeId a, NodeId b) -> float {
                    return l2_distance_simd<D>(graph.get_vector(a), graph.get_vector(b));
                };
                auto selected = select_neighbors_heuristic(
                    std::move(candidates), GraphR, local_dist_fn);

                write_neighbors(graph, encoder, u, selected);
            }
        }

        // ── Phase 3: Add reverse edges from refinement ──
        if (verbose) printf("[Build] Adding reverse edges...\n");
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

            auto local_dist_fn = [&](NodeId a, NodeId b) -> float {
                return l2_distance_simd<D>(graph.get_vector(a), graph.get_vector(b));
            };
            auto selected = select_neighbors_heuristic(
                std::move(all), GraphR, local_dist_fn);

            write_neighbors(graph, encoder, v, selected);
        }

        if (verbose) printf("[Build] Done.\n");
    }
};

using GraphRefinement128 = GraphRefinement<128, 32>;
using GraphRefinement256 = GraphRefinement<256, 32>;
using GraphRefinement1024 = GraphRefinement<1024, 32>;

}  // namespace cphnsw
