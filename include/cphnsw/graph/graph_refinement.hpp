#pragma once

#include "../core/codes.hpp"
#include "../distance/fastscan_layout.hpp"
#include "../encoder/rabitq_encoder.hpp"
#include "rabitq_graph.hpp"
#include "neighbor_selection.hpp"
#include "../search/rabitq_search.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>

#ifdef _OPENMP
#include <omp.h>
#else
inline int omp_get_thread_num() { return 0; }
inline int omp_get_max_threads() { return 1; }
#endif

namespace cphnsw {

template <size_t D, size_t R = 32>
class GraphRefinement {
public:
    using Graph = RaBitQGraph<D, R>;
    using CodeType = RaBitQCode<D>;
    using EncoderType = RaBitQEncoder<D>; // This might be templated in future, but for now it's okay if RaBitQEncoder has default policy.
    // Actually, optimize_graph is passed 'const EncoderType& encoder'. 
    // If the passed encoder has a different type (DenseRotation), this signature might fail to match 
    // if we enforce RaBitQEncoder<D> which defaults to RandomHadamardRotation.
    // We should template the function on EncoderType.

    template <typename EncType>
    static void build_random_graph(Graph& graph, const EncType& encoder, size_t seed = 42) {
        size_t n = graph.size();
        size_t GraphR = Graph::DEGREE;
        if (n <= GraphR) return;

        #pragma omp parallel
        {
            std::mt19937 rng(seed + omp_get_thread_num());
            std::uniform_int_distribution<NodeId> dist(0, n - 1);

            #pragma omp for
            for (size_t i = 0; i < n; ++i) {
                NodeId u = static_cast<NodeId>(i);
                auto& nb = graph.get_neighbors(u);
                if (nb.count > 0) continue;

                const float* vec_u = graph.get_vector(u);

                size_t attempts = 0;
                size_t filled = 0;
                while (filled < GraphR && attempts < GraphR * 2) {
                    NodeId v = dist(rng);
                    if (v == u) continue;
                    
                    // Simple check for dups in small list
                    bool dup = false;
                    for (size_t k = 0; k < filled; ++k) if (nb.neighbor_ids[k] == v) dup = true;
                    if (dup) continue;

                    const auto& code_v = graph.get_code(v);
                    const float* vec_v = graph.get_vector(v);
                    
                    VertexAuxData aux = encoder.compute_neighbor_aux(code_v, vec_u, vec_v);
                    nb.set_neighbor(filled++, v, code_v.signs, aux);
                    attempts++;
                }
            }
        }
    }

    template <typename EncType>
    static void optimize_graph(Graph& graph, const EncType& encoder, 
                             size_t ef_construction, size_t num_threads = 0, bool verbose = false) {
        size_t n = graph.size();
        size_t GraphR = Graph::DEGREE;
        
        // 1. Ensure connectivity (Random Init)
        if (verbose) printf("[Build] Initializing random graph...\n");
        build_random_graph(graph, encoder);
        
        NodeId entry_point = graph.find_medoid();
        graph.set_entry_point(entry_point);

        // 2. Iterative Refinement (Vamana-style)
        // Two passes: 
        // Pass 1: Short ef (fast topology fix)
        // Pass 2: Long ef (final convergence)
        size_t passes = 2;
        size_t efs[] = {ef_construction / 2, ef_construction};
        
        for (size_t p = 0; p < passes; ++p) {
            size_t current_ef = std::max<size_t>(GraphR, efs[p]);
            if (verbose) printf("[Build] Optimization Pass %zu/%zu (ef=%zu)...\n", p+1, passes, current_ef);

            // Thread-local resources
            size_t actual_threads = num_threads ? num_threads : omp_get_max_threads();
            
            #pragma omp parallel num_threads(actual_threads)
            {
                // Thread-local visitation table
                TwoLevelVisitationTable visited(n + 1024);

                #pragma omp for schedule(dynamic, 256)
                for (size_t i = 0; i < n; ++i) {
                    NodeId u = static_cast<NodeId>(i);
                    const float* vec_u = graph.get_vector(u);
                    
                    // Encode u as query
                    auto query = encoder.encode_query(vec_u);
                    
                    // Search for candidates
                    // We start from entry_point
                    auto results = RaBitQSearchEngine<D, R>::search(
                        query, vec_u, graph, current_ef, current_ef, visited);
                    
                    // Convert to candidates
                    std::vector<NeighborCandidate> candidates;
                    candidates.reserve(results.size() + GraphR);
                    for (const auto& r : results) {
                        if (r.id != u) candidates.push_back({r.id, r.distance});
                    }
                    
                    // Add existing neighbors to candidates (to preserve good edges)
                    const auto& nb = graph.get_neighbors(u);
                    for (size_t j = 0; j < nb.count; ++j) {
                        NodeId v = nb.neighbor_ids[j];
                        if (v == INVALID_NODE || v == u) continue;
                        float d = exact_distance(graph, u, v);
                        candidates.push_back({v, d});
                    }

                    // Heuristic Selection (Alpha=1 implied by function)
                    auto dist_fn = [&](NodeId a, NodeId b) { return exact_distance(graph, a, b); };
                    auto selected = select_neighbors_heuristic(std::move(candidates), GraphR, dist_fn);
                    
                    // Update neighbors
                    auto& nb_mut = graph.get_neighbors(u); // Thread-safe (only u modifies u)
                    nb_mut.count = 0;
                    
                    for (size_t j = 0; j < selected.size(); ++j) {
                        NodeId v = selected[j].id;
                        const auto& code_v = graph.get_code(v);
                        const float* vec_v = graph.get_vector(v);
                        VertexAuxData aux = encoder.compute_neighbor_aux(code_v, vec_u, vec_v);
                        nb_mut.set_neighbor(j, v, code_v.signs, aux);
                    }
                }
            }
        }
        if (verbose) printf("[Build] Done.\n");
    }

    static float exact_distance(const Graph& graph, NodeId a, NodeId b) {
        const float* va = graph.get_vector(a);
        const float* vb = graph.get_vector(b);
        size_t dim = graph.dim();

        float dist = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            float d = va[i] - vb[i];
            dist += d * d;
        }
        return dist;
    }
};

using GraphRefinement128 = GraphRefinement<128, 32>;
using GraphRefinement256 = GraphRefinement<256, 32>;
using GraphRefinement1024 = GraphRefinement<1024, 32>;

}  // namespace cphnsw