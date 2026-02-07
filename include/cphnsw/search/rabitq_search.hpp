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

template <size_t D, size_t R = 32>
class RaBitQSearchEngine {
public:
    using Graph = RaBitQGraph<D, R>;
    using CodeType = RaBitQCode<D>;
    using QueryType = RaBitQQuery<D>;
    using Policy = RaBitQMetricPolicy<D>;
    using NeighborBlock = FastScanNeighborBlock<D, R, 32>;

    static std::vector<SearchResult> search(
        const QueryType& query,
        const float* raw_query,
        const Graph& graph,
        size_t ef,
        size_t k,
        TwoLevelVisitationTable& visited)
    {
        if (graph.empty()) return {};
        if (k == 0) k = ef;

        size_t dim = graph.dim();
        uint64_t query_id = visited.new_query();

        struct BeamEntry {
            float est_distance;
            NodeId id;

            bool operator>(const BeamEntry& o) const {
                return est_distance > o.est_distance;
            }
        };

        std::priority_queue<BeamEntry, std::vector<BeamEntry>,
                           std::greater<BeamEntry>> beam;

        MaxHeap nn;

        NodeId ep = graph.entry_point();
        if (ep == INVALID_NODE) return {};

        float ep_est = Policy::compute_distance(query, graph.get_code(ep));
        beam.push({ep_est, ep});
        visited.check_and_mark_estimated(ep, query_id);

        alignas(64) uint32_t fastscan_sums[R];
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

            if (!beam.empty()) {
                const auto& next = beam.top();
                prefetch_t<1>(graph.get_vector(next.id));
                prefetch_t<1>(&graph.get_neighbors(next.id));
            }

            visited.check_and_mark_visited(current.id, query_id);

            const float* vertex_vec = graph.get_vector(current.id);
            float exact_dist = 0.0f;
            for (size_t i = 0; i < dim; ++i) {
                float d = raw_query[i] - vertex_vec[i];
                exact_dist += d * d;
            }

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
                size_t batch_count = std::min(BATCH, n_neighbors - batch_start);

                if (batch_count == 0) break;

                fastscan::compute_inner_products(
                    query.lut, nb.code_blocks[batch], fastscan_sums + batch_start);

                fastscan::convert_to_distances_with_bounds(
                    query,
                    fastscan_sums + batch_start,
                    nb.aux + batch_start,
                    nb.popcounts + batch_start,
                    batch_count,
                    est_distances + batch_start,
                    lower_bounds + batch_start,
                    parent_norm,
                    dist_qp_sq);
            }

            for (size_t i = 0; i < n_neighbors; ++i) {
                NodeId neighbor_id = nb.neighbor_ids[i];
                if (neighbor_id == INVALID_NODE) continue;

                if (visited.is_visited(neighbor_id, query_id)) continue;

                float est_dist = est_distances[i];
                float lower = lower_bounds[i];

                if (nn.size() >= k && lower >= nn.top().distance) continue;

                visited.check_and_mark_estimated(neighbor_id, query_id);
                beam.push({est_dist, neighbor_id});
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
        return search(query, raw_query, graph, ef, k, visited);
    }
};

using RaBitQSearch128 = RaBitQSearchEngine<128, 32>;
using RaBitQSearch256 = RaBitQSearchEngine<256, 32>;
using RaBitQSearch1024 = RaBitQSearchEngine<1024, 32>;

}  // namespace cphnsw