#pragma once

#include "../core/codes.hpp"
#include "../core/adaptive_defaults.hpp"
#include "../distance/fastscan_kernel.hpp"
#include "../distance/fastscan_layout.hpp"
#include "../graph/rabitq_graph.hpp"
#include "../core/types.hpp"
#include "../graph/visitation_table.hpp"
#include <vector>
#include <queue>
#include <algorithm>
#include <limits>

namespace cphnsw {

template <typename T>
class BoundedMaxHeap {
public:
    explicit BoundedMaxHeap(size_t capacity) : capacity_(capacity) {
        data_.reserve(capacity + 1);
    }

    bool empty() const { return data_.empty(); }
    size_t size() const { return data_.size(); }
    const T& top() const { return data_.front(); }

    void push(const T& val) {
        if (data_.size() < capacity_) {
            data_.push_back(val);
            std::push_heap(data_.begin(), data_.end());
        } else if (val < data_.front()) {
            std::pop_heap(data_.begin(), data_.end());
            data_.back() = val;
            std::push_heap(data_.begin(), data_.end());
        }
    }

    void pop() {
        std::pop_heap(data_.begin(), data_.end());
        data_.pop_back();
    }

    std::vector<T> extract_sorted() {
        std::sort_heap(data_.begin(), data_.end());
        return std::move(data_);
    }

    float worst_distance() const {
        return data_.empty() ? std::numeric_limits<float>::max() : data_.front().distance;
    }

private:
    std::vector<T> data_;
    size_t capacity_;
};

namespace rabitq_search {

struct BeamEntry {
    float est_distance;
    float lower_bound;
    NodeId id;
    bool operator>(const BeamEntry& o) const { return est_distance > o.est_distance; }
};

template <size_t D, size_t R = 32, size_t BitWidth = 1>
std::vector<SearchResult> search(
    const RaBitQQuery<D>& query,
    const float* raw_query,
    const RaBitQGraph<D, R, BitWidth>& graph,
    size_t k,
    float gamma,
    TwoLevelVisitationTable& visited,
    size_t ef_cap = 0,
    NodeId entry = INVALID_NODE)
{
    if (graph.empty()) return {};
    if (k == 0) k = adaptive_defaults::default_k();
    if (ef_cap == 0) ef_cap = adaptive_defaults::ef_cap(graph.size(), k, gamma);

    NodeId ep = (entry != INVALID_NODE) ? entry : graph.entry_point();
    if (ep == INVALID_NODE) return {};

    uint64_t query_id = visited.new_query();

    std::priority_queue<BeamEntry, std::vector<BeamEntry>,
                       std::greater<BeamEntry>> beam;
    BoundedMaxHeap<SearchResult> nn(k);

    float ep_est = l2_distance_simd<D>(raw_query, graph.get_vector(ep));
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

        if (nn.size() >= k && current.est_distance > (1.0f + gamma) * nn.worst_distance()) break;

        if (nn.size() >= k && current.lower_bound > nn.worst_distance()) continue;

        if (!beam.empty()) {
            graph.prefetch_vertex(beam.top().id);
            graph.prefetch_vector(beam.top().id);
            BeamEntry first_beam = beam.top();
            beam.pop();
            if (!beam.empty()) {
                graph.prefetch_vertex(beam.top().id);
                graph.prefetch_vector(beam.top().id);
            }
            beam.push(first_beam);
        }

        visited.check_and_mark_visited(current.id, query_id);

        // Compute exact L2 distance to parent node for RaBitQ formula accuracy
        float exact_dist = l2_distance_simd<D>(raw_query, graph.get_vector(current.id));
        nn.push({current.id, exact_dist});

        const auto& nb = graph.get_neighbors(current.id);
        size_t n_neighbors = nb.size();
        if (n_neighbors == 0) continue;

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
                    lower_bounds + batch_start, dist_qp_sq);
            } else {
                // Two-stage progressive filtering:
                // Stage 1: MSB-only fast lower bounds
                fastscan::compute_msb_only_inner_products<D, BitWidth>(
                    query.lut, nb.code_blocks[batch], msb_sums + batch_start);
                fastscan::convert_msb_to_lower_bounds<D>(
                    query, msb_sums + batch_start, nb.aux + batch_start,
                    nb.popcounts + batch_start, batch_count,
                    lower_bounds + batch_start, dist_qp_sq);

                // Check if any candidate in this batch survives the MSB filter
                float threshold = nn.worst_distance();
                bool any_survivor = (nn.size() < k);
                if (!any_survivor) {
                    for (size_t j = 0; j < batch_count; ++j) {
                        if (lower_bounds[batch_start + j] < threshold) {
                            any_survivor = true;
                            break;
                        }
                    }
                }

                if (any_survivor) {
                    // Stage 2: Full N-bit computation for survivors
                    fastscan::compute_nbit_inner_products<D, BitWidth>(
                        query.lut, nb.code_blocks[batch],
                        fastscan_sums + batch_start, msb_sums + batch_start);
                    fastscan::convert_nbit_to_distances_with_bounds<D, BitWidth>(
                        query, fastscan_sums + batch_start,
                        msb_sums + batch_start, nb.aux + batch_start,
                        nb.popcounts + batch_start,
                        nb.weighted_popcounts + batch_start,
                        batch_count, est_distances + batch_start,
                        lower_bounds + batch_start, dist_qp_sq);
                } else {
                    // Entire batch filtered by MSB bounds
                    for (size_t j = 0; j < batch_count; ++j) {
                        est_distances[batch_start + j] = std::numeric_limits<float>::max();
                    }
                }
            }
        }

        for (size_t i = 0; i < n_neighbors; ++i) {
            NodeId neighbor_id = nb.neighbor_ids[i];
            if (neighbor_id == INVALID_NODE) continue;
            if (visited.check_and_mark_estimated(neighbor_id, query_id)) continue;

            float est_dist = est_distances[i];
            float lower = lower_bounds[i];
            if (nn.size() >= k && lower >= nn.worst_distance()) continue;

            beam.push({est_dist, lower, neighbor_id});
            graph.prefetch_vertex(neighbor_id);
        }

        size_t trim_trigger = static_cast<size_t>(ef_cap * adaptive_defaults::beam_trim_trigger_ratio());
        if (beam.size() > trim_trigger) {
            std::priority_queue<BeamEntry, std::vector<BeamEntry>,
                               std::greater<BeamEntry>> new_beam;
            size_t keep = static_cast<size_t>(ef_cap * adaptive_defaults::beam_trim_keep_ratio());
            size_t kept = 0;
            while (!beam.empty() && kept < keep) {
                new_beam.push(beam.top());
                beam.pop();
                kept++;
            }
            beam = std::move(new_beam);
        }
    }

    return nn.extract_sorted();
}

}  // namespace rabitq_search
}  // namespace cphnsw
