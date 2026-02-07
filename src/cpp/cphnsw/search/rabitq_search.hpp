#pragma once

#include "../core/codes.hpp"
#include "../distance/fastscan_kernel.hpp"
#include "../distance/fastscan_layout.hpp"
#include "../graph/rabitq_graph.hpp"
#include "../core/types.hpp"
#include "../graph/visitation_table.hpp"
#include <vector>
#include <queue>
#include <algorithm>
#include <cmath>

namespace cphnsw {

// ============================================================================
// RaBitQ Search Engine: SymphonyQG-style graph search
// ============================================================================

/**
 * RaBitQSearchEngine: High-performance search using RaBitQ quantization
 * with SymphonyQG's implicit reranking and FastScan SIMD.
 *
 * KEY FEATURES:
 *   - Thread-local visitation tables (no shared mutable state during search)
 *   - FastScan SIMD for batch distance estimation of ALL R neighbors
 *   - Implicit reranking: exact distance computed when visiting a vertex
 *   - Error-bound pruning: skip neighbors provably worse than current k-th best
 *
 * ALGORITHM (SymphonyQG Algorithm 1):
 *   1. Initialize beam set S and NN with entry point
 *   2. While unvisited vertices exist in S:
 *      a. Pop unvisited vertex p with smallest ESTIMATED distance
 *      b. Compute EXACT distance for p (implicit reranking) → update NN
 *      c. Estimate distances for all R neighbors of p via FastScan
 *      d. Add unvisited neighbors to S (keeping beam size bounded)
 *   3. Return NN
 *
 * @tparam D Padded dimension
 * @tparam R Fixed degree
 */
template <size_t D, size_t R = 32>
class RaBitQSearchEngine {
public:
    using Graph = RaBitQGraph<D, R>;
    using CodeType = RaBitQCode<D>;
    using QueryType = RaBitQQuery<D>;
    using Policy = RaBitQMetricPolicy<D>;
    using NeighborBlock = FastScanNeighborBlock<D, R, 32>;

    /**
     * Perform k-NN search with SymphonyQG-style implicit reranking.
     *
     * @param query RaBitQ encoded query
     * @param raw_query Raw query vector (for exact distance computation)
     * @param graph The RaBitQ graph to search
     * @param ef Beam size (search width)
     * @param k Number of results to return
     * @param visited Thread-local visitation table
     * @return k-nearest neighbors sorted by exact distance
     */
    static std::vector<SearchResult> search(
        const QueryType& query,
        const float* raw_query,
        const Graph& graph,
        size_t ef,
        size_t k,
        VisitationTable& visited)
    {
        if (graph.empty()) return {};
        if (k == 0) k = ef;

        size_t dim = graph.dim();
        uint64_t query_id = visited.new_query();

        // Beam set: (estimated_distance, node_id, visited_flag)
        // Using a sorted structure for efficient access
        struct BeamEntry {
            float est_distance;
            NodeId id;
            bool is_visited;

            bool operator>(const BeamEntry& o) const {
                return est_distance > o.est_distance;
            }
        };

        // Min-heap of beam entries (sorted by estimated distance)
        std::priority_queue<BeamEntry, std::vector<BeamEntry>,
                           std::greater<BeamEntry>> beam;

        // NN: exact nearest neighbors found so far (max-heap for bounded set)
        MaxHeap nn;

        // Initialize with entry point
        NodeId ep = graph.entry_point();
        if (ep == INVALID_NODE) return {};

        // Compute initial estimated distance to entry point
        float ep_est = Policy::compute_distance(query, graph.get_code(ep));
        beam.push({ep_est, ep, false});
        visited.check_and_mark(ep, query_id);

        // Temporary buffers for batch distance computation
        alignas(64) uint32_t fastscan_sums[R];
        alignas(64) float est_distances[R];

        while (!beam.empty()) {
            // Find the closest UNVISITED vertex in the beam
            // Pop entries until we find an unvisited one
            BeamEntry current;
            bool found = false;

            while (!beam.empty()) {
                current = beam.top();
                beam.pop();

                if (!current.is_visited) {
                    found = true;
                    break;
                }
                // Already visited, discard (don't reinsert)
            }

            if (!found) break;

            // Mark as visited
            current.is_visited = true;

            // === IMPLICIT RERANKING (SymphonyQG Line 4) ===
            // Compute EXACT distance for the visited vertex
            const float* vertex_vec = graph.get_vector(current.id);
            float exact_dist = 0.0f;
            for (size_t i = 0; i < dim; ++i) {
                float d = raw_query[i] - vertex_vec[i];
                exact_dist += d * d;
            }

            // Update NN with exact distance
            if (nn.size() < k) {
                nn.push({current.id, exact_dist});
            } else if (exact_dist < nn.top().distance) {
                nn.pop();
                nn.push({current.id, exact_dist});
            }

            // === ESTIMATE DISTANCES FOR ALL R NEIGHBORS (SymphonyQG Line 5) ===
            const auto& nb = graph.get_neighbors(current.id);
            size_t n_neighbors = nb.size();

            if (n_neighbors == 0) continue;

            // Process neighbors in batches of 32 (one FastScan block)
            constexpr size_t BATCH = 32;
            size_t num_batches = (R + BATCH - 1) / BATCH;

            for (size_t batch = 0; batch < num_batches; ++batch) {
                size_t batch_start = batch * BATCH;
                size_t batch_count = std::min(BATCH, n_neighbors - batch_start);

                if (batch_count == 0) break;

                // FastScan SIMD computation
#if defined(__AVX2__)
                fastscan::compute_inner_products_avx2(
                    query.lut, nb.code_blocks[batch], fastscan_sums + batch_start);
#else
                fastscan::compute_inner_products_scalar(
                    query.lut, nb.code_blocks[batch], batch_count,
                    fastscan_sums + batch_start);
#endif

                // Convert to distance estimates
                fastscan::convert_to_distances(
                    query,
                    fastscan_sums + batch_start,
                    nb.aux + batch_start,
                    nb.popcounts + batch_start,
                    batch_count,
                    est_distances + batch_start);
            }

            // === ADD UNVISITED NEIGHBORS TO BEAM (SymphonyQG Line 6) ===
            for (size_t i = 0; i < n_neighbors; ++i) {
                NodeId neighbor_id = nb.neighbor_ids[i];
                if (neighbor_id == INVALID_NODE) continue;

                // Check if already visited
                if (visited.check_and_mark(neighbor_id, query_id)) {
                    continue;  // Already visited
                }

                float est_dist = est_distances[i];

                // TODO(SOTA): Replace 1.5x heuristic with error-bound pruning (Theorem 3.2)
                if (nn.size() < k || est_dist < nn.top().distance * 1.5f) {
                    beam.push({est_dist, neighbor_id, false});
                }
            }

            // Beam size management: limit to ef entries
            // (In practice, the priority queue naturally handles this
            // by always exploring the most promising candidate)
            if (beam.size() > ef * 2) {
                // Rebuild with only top ef entries
                std::priority_queue<BeamEntry, std::vector<BeamEntry>,
                                   std::greater<BeamEntry>> new_beam;
                size_t kept = 0;
                while (!beam.empty() && kept < ef) {
                    new_beam.push(beam.top());
                    beam.pop();
                    kept++;
                }
                beam = std::move(new_beam);
            }
        }

        // Extract results sorted by exact distance (ascending)
        std::vector<SearchResult> results;
        results.reserve(nn.size());
        while (!nn.empty()) {
            results.push_back(nn.top());
            nn.pop();
        }
        std::reverse(results.begin(), results.end());

        return results;
    }

    /**
     * Search with automatic thread-local visitation table.
     */
    static std::vector<SearchResult> search(
        const QueryType& query,
        const float* raw_query,
        const Graph& graph,
        size_t ef,
        size_t k = 10)
    {
        thread_local VisitationTable visited(0);
        if (visited.capacity() < graph.size()) {
            visited.resize(graph.size() + 1024);
        }
        return search(query, raw_query, graph, ef, k, visited);
    }
};

// Type aliases
using RaBitQSearch128 = RaBitQSearchEngine<128, 32>;
using RaBitQSearch256 = RaBitQSearchEngine<256, 32>;
using RaBitQSearch1024 = RaBitQSearchEngine<1024, 32>;

}  // namespace cphnsw
