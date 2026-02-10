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

template <size_t D, size_t R = 32, size_t BitWidth = 1>
class RaBitQSearchEngine {
public:
    using Graph = RaBitQGraph<D, R, BitWidth>;
    using QueryType = RaBitQQuery<D>;
    using Policy = RaBitQMetricPolicy<D>;

    struct BeamEntry {
        float est_distance;
        float lower_bound;
        NodeId id;
        bool operator>(const BeamEntry& o) const { return est_distance > o.est_distance; }
    };

    // Core search with explicit entry point and visitation table.
    // If entry == INVALID_NODE, uses graph.entry_point().
    static std::vector<SearchResult> search(
        const QueryType& query,
        const float* raw_query,
        const Graph& graph,
        size_t ef,
        size_t k,
        TwoLevelVisitationTable& visited,
        NodeId entry = INVALID_NODE)
    {
        if (graph.empty()) return {};
        if (k == 0) k = ef;

        NodeId ep = (entry != INVALID_NODE) ? entry : graph.entry_point();
        if (ep == INVALID_NODE) return {};

        uint64_t query_id = visited.new_query();

        std::priority_queue<BeamEntry, std::vector<BeamEntry>,
                           std::greater<BeamEntry>> beam;
        MaxHeap nn;

        // Entry-point distance estimation
        float ep_est;
        if constexpr (BitWidth == 1) {
            ep_est = Policy::compute_distance(query, graph.get_code(ep));
        } else {
            const auto& nbit_code = graph.get_code(ep);
            RaBitQCode<D> compat;
            compat.signs = nbit_code.codes.msb_as_binary();
            compat.dist_to_centroid = nbit_code.dist_to_centroid;
            compat.ip_quantized_original = nbit_code.ip_quantized_original;
            compat.code_popcount = nbit_code.msb_popcount;
            ep_est = Policy::compute_distance(query, compat);
        }
        beam.push({ep_est, 0.0f, ep});
        visited.check_and_mark_estimated(ep, query_id);

        alignas(64) uint32_t fastscan_sums[R];
        alignas(64) uint32_t msb_sums[R];
        alignas(64) float est_distances[R];
        alignas(64) float lower_bounds[R];

        while (!beam.empty()) {
            BeamEntry current;
            bool found = false;

            while (!beam.empty()) {
                current = beam.top();
                beam.pop();
                if (visited.is_visited(current.id, query_id)) continue;
                found = true;
                break;
            }
            if (!found) break;

            // Early termination: best estimated candidate is worse than k-th exact
            if (nn.size() >= k && current.est_distance > nn.top().distance) break;

            if (!beam.empty()) {
                graph.prefetch_vertex(beam.top().id);
            }

            // Two-stage check: skip exact distance if lower bound exceeds k-th best
            if (nn.size() >= k && current.lower_bound > nn.top().distance) continue;

            visited.check_and_mark_visited(current.id, query_id);
            prefetch_t<0>(graph.get_vector(current.id));

            const float* vertex_vec = graph.get_vector(current.id);
            float exact_dist = l2_distance_simd<D>(raw_query, vertex_vec);

            if (nn.size() < k) {
                nn.push({current.id, exact_dist});
            } else if (exact_dist < nn.top().distance) {
                nn.pop();
                nn.push({current.id, exact_dist});
            }

            const auto& nb = graph.get_neighbors(current.id);
            size_t n_neighbors = nb.size();
            if (n_neighbors == 0) continue;

            float parent_norm = graph.get_code(current.id).dist_to_centroid;
            float dist_qp_sq = exact_dist;

            constexpr size_t BATCH = 32;
            size_t num_batches = (R + BATCH - 1) / BATCH;

            for (size_t batch = 0; batch < num_batches; ++batch) {
                size_t batch_start = batch * BATCH;
                if (batch_start >= n_neighbors) break;
                size_t batch_count = std::min(BATCH, n_neighbors - batch_start);

                if constexpr (BitWidth == 1) {
                    fastscan::compute_inner_products(
                        query.lut, nb.code_blocks[batch],
                        fastscan_sums + batch_start);
                    fastscan::convert_to_distances_with_bounds(
                        query, fastscan_sums + batch_start,
                        nb.aux + batch_start, nb.popcounts + batch_start,
                        batch_count, est_distances + batch_start,
                        lower_bounds + batch_start, parent_norm, dist_qp_sq);
                } else {
                    fastscan::compute_nbit_inner_products<D, BitWidth>(
                        query.lut, nb.code_blocks[batch],
                        fastscan_sums + batch_start, msb_sums + batch_start);
                    fastscan::convert_nbit_to_distances_with_bounds<D, BitWidth>(
                        query, fastscan_sums + batch_start,
                        msb_sums + batch_start, nb.aux + batch_start,
                        nb.popcounts + batch_start,
                        nb.weighted_popcounts + batch_start,
                        batch_count, est_distances + batch_start,
                        lower_bounds + batch_start, parent_norm, dist_qp_sq);
                }
            }

            for (size_t i = 0; i < n_neighbors; ++i) {
                NodeId neighbor_id = nb.neighbor_ids[i];
                if (neighbor_id == INVALID_NODE) continue;
                if (visited.is_visited(neighbor_id, query_id)) continue;

                float est_dist = est_distances[i];
                float lower = lower_bounds[i];
                if (nn.size() >= k && lower >= nn.top().distance) continue;

                visited.check_and_mark_estimated(neighbor_id, query_id);
                beam.push({est_dist, lower, neighbor_id});
                graph.prefetch_vertex(neighbor_id);
            }

            if (beam.size() > ef * 3) {
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

        std::vector<SearchResult> results;
        results.reserve(nn.size());
        while (!nn.empty()) {
            results.push_back(nn.top());
            nn.pop();
        }
        std::reverse(results.begin(), results.end());
        return results;
    }

    // Convenience: thread-local visitation, default entry point
    static std::vector<SearchResult> search(
        const QueryType& query,
        const float* raw_query,
        const Graph& graph,
        size_t ef,
        size_t k = 10)
    {
        thread_local TwoLevelVisitationTable visited(0);
        if (visited.capacity() < graph.size()) {
            visited.resize(graph.size() + 1024);
        }
        return search(query, raw_query, graph, ef, k, visited, INVALID_NODE);
    }

    // Search from a custom entry node (for HNSW upper-layer routing)
    static std::vector<SearchResult> search_from(
        const QueryType& query,
        const float* raw_query,
        const Graph& graph,
        NodeId entry,
        size_t ef,
        size_t k = 10)
    {
        thread_local TwoLevelVisitationTable visited(0);
        if (visited.capacity() < graph.size()) {
            visited.resize(graph.size() + 1024);
        }
        return search(query, raw_query, graph, ef, k, visited, entry);
    }
};

using RaBitQSearch128 = RaBitQSearchEngine<128, 32>;
using RaBitQSearch256 = RaBitQSearchEngine<256, 32>;
using RaBitQSearch1024 = RaBitQSearchEngine<1024, 32>;

// Exact L2 beam search for graph construction (Vamana-style).
// Uses only exact L2 distances — no RaBitQ approximations.
// This produces correct graph topology; RaBitQ is used only at query time.
template <size_t D, size_t R = 32, size_t BitWidth = 1>
class ExactL2SearchEngine {
public:
    using Graph = RaBitQGraph<D, R, BitWidth>;

    struct BeamEntry {
        float distance;  // exact L2²
        NodeId id;
        bool operator>(const BeamEntry& o) const { return distance > o.distance; }
    };

    // Beam search using exact L2 distances for all navigation.
    static std::vector<SearchResult> search(
        const float* raw_query,
        const Graph& graph,
        size_t ef,
        size_t k,
        VisitationTable& visited,
        NodeId entry = INVALID_NODE)
    {
        if (graph.empty()) return {};
        if (k == 0) k = ef;

        NodeId ep = (entry != INVALID_NODE) ? entry : graph.entry_point();
        if (ep == INVALID_NODE) return {};

        uint64_t query_id = visited.new_query();

        std::priority_queue<BeamEntry, std::vector<BeamEntry>,
                           std::greater<BeamEntry>> beam;
        MaxHeap nn;

        // Entry point: exact L2 distance
        float ep_dist = l2_distance_simd<D>(raw_query, graph.get_vector(ep));
        beam.push({ep_dist, ep});
        visited.check_and_mark(ep, query_id);

        while (!beam.empty()) {
            BeamEntry current = beam.top();
            beam.pop();

            // Early termination: best candidate worse than k-th result
            if (nn.size() >= k && current.distance > nn.top().distance) break;

            // Prefetch next beam candidate's vertex data
            if (!beam.empty()) {
                graph.prefetch_vertex(beam.top().id);
            }

            // Update result set
            if (nn.size() < k) {
                nn.push({current.id, current.distance});
            } else if (current.distance < nn.top().distance) {
                nn.pop();
                nn.push({current.id, current.distance});
            }

            // Expand neighbors
            const auto& nb = graph.get_neighbors(current.id);
            size_t n_neighbors = nb.size();

            for (size_t i = 0; i < n_neighbors; ++i) {
                // Prefetch vector 2 positions ahead
                if (i + 2 < n_neighbors) {
                    NodeId prefetch_id = nb.neighbor_ids[i + 2];
                    if (prefetch_id != INVALID_NODE) {
                        prefetch_t<0>(graph.get_vector(prefetch_id));
                    }
                }

                NodeId neighbor_id = nb.neighbor_ids[i];
                if (neighbor_id == INVALID_NODE) continue;
                if (visited.check_and_mark(neighbor_id, query_id)) continue;

                float dist = l2_distance_simd<D>(raw_query, graph.get_vector(neighbor_id));

                // Prune: skip if worse than k-th best
                if (nn.size() >= k && dist >= nn.top().distance) continue;

                beam.push({dist, neighbor_id});
            }

            // Beam size management
            if (beam.size() > ef * 3) {
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

        std::vector<SearchResult> results;
        results.reserve(nn.size());
        while (!nn.empty()) {
            results.push_back(nn.top());
            nn.pop();
        }
        std::reverse(results.begin(), results.end());
        return results;
    }
};

}  // namespace cphnsw
