#pragma once

#include "../core/codes.hpp"
#include "../core/constants.hpp"
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

    size_t size() const { return data_.size(); }

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
    RaBitQQuery<D> query,
    const float* raw_query,
    const RaBitQGraph<D, R, BitWidth>& graph,
    size_t k,
    float gamma,
    TwoLevelVisitationTable& visited,
    NodeId entry,
    const float* slack_levels,
    int num_slack_levels,
    float gamma_max,
    float gamma_beta,
    size_t gamma_warmup)
{
    NodeId ep = entry;

    uint64_t query_id = visited.new_query();

    std::priority_queue<BeamEntry, std::vector<BeamEntry>,
                       std::greater<BeamEntry>> beam;
    BoundedMaxHeap<SearchResult> nn(k);


    float gamma_q = gamma;
    double ratio_sum = 0.0, ratio_sq_sum = 0.0;
    size_t ratio_count = 0;

    float query_norm_sq = dot_product_simd<D>(raw_query, raw_query);

    auto exact_l2 = [&](NodeId id) -> float {
        return std::max(query_norm_sq + graph.get_norm_sq(id)
               - 2.0f * dot_product_simd<D>(raw_query, graph.get_vector(id)), 0.0f);
    };

    float ep_est = exact_l2(ep);
    beam.push({ep_est, 0.0f, ep});
    visited.check_and_mark_estimated(ep, query_id);

    alignas(64) uint32_t fastscan_sums[R];
    alignas(64) uint32_t msb_sums[R];
    alignas(64) float est_distances[R];
    alignas(64) float lower_bounds[R];

    int slack_batch_count = 0;

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


        if (nn.size() >= k && current.est_distance >= gamma_q * nn.worst_distance()) break;

        if (nn.size() >= k && current.lower_bound > nn.worst_distance()) continue;

        if (!beam.empty()) {
            graph.prefetch_vertex(beam.top().id);
            graph.prefetch_vector(beam.top().id);
            graph.prefetch_norm(beam.top().id);
        }

        visited.check_and_mark_visited(current.id, query_id);

        float exact_dist = exact_l2(current.id);
        nn.push({current.id, exact_dist});

        const auto& nb = graph.get_neighbors(current.id);
        size_t n_neighbors = nb.size();
        if (n_neighbors == 0) continue;

        float dist_qp_sq = exact_dist;

        if (slack_levels && num_slack_levels > 0) {
            int level_idx = std::min(slack_batch_count, num_slack_levels - 1);
            query.dot_slack = slack_levels[level_idx];
            ++slack_batch_count;
        }

        constexpr size_t BATCH = constants::kFastScanBatch;
        size_t num_batches = (R + BATCH - 1) / BATCH;

        for (size_t batch = 0; batch < num_batches; ++batch) {
            size_t batch_start = batch * BATCH;
            if (batch_start >= n_neighbors) break;
            size_t batch_count = std::min(BATCH, n_neighbors - batch_start);

            if (batch + 1 < num_batches && (batch + 1) * BATCH < n_neighbors) {
                prefetch_t<0>(reinterpret_cast<const char*>(&nb.code_blocks[batch + 1]));
            }

            if constexpr (BitWidth == 1) {
                fastscan::compute_inner_products(
                    query.lut, nb.code_blocks[batch],
                    fastscan_sums + batch_start);
                fastscan::convert_to_distances_with_bounds(
                    query, fastscan_sums + batch_start,
                    nb.nop + batch_start, nb.ip_qo + batch_start,
                    nb.ip_cp + batch_start, nb.popcounts + batch_start,
                    batch_count, est_distances + batch_start,
                    lower_bounds + batch_start, dist_qp_sq);
            } else {
                fastscan::compute_msb_only_inner_products<D, BitWidth>(
                    query.lut, nb.code_blocks[batch], msb_sums + batch_start);
                fastscan::convert_msb_to_lower_bounds<D, BitWidth>(
                    query, msb_sums + batch_start,
                    nb.nop + batch_start, nb.ip_qo + batch_start,
                    nb.ip_cp + batch_start, nb.popcounts + batch_start,
                    batch_count, lower_bounds + batch_start, dist_qp_sq);

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
                    fastscan::compute_nbit_inner_products<D, BitWidth>(
                        query.lut, nb.code_blocks[batch],
                        fastscan_sums + batch_start, msb_sums + batch_start);
                    fastscan::convert_nbit_to_distances_with_bounds<D, BitWidth>(
                        query, fastscan_sums + batch_start,
                        msb_sums + batch_start,
                        nb.nop + batch_start, nb.ip_qo + batch_start,
                        nb.ip_cp + batch_start, nb.popcounts + batch_start,
                        nb.weighted_popcounts + batch_start,
                        batch_count, est_distances + batch_start,
                        lower_bounds + batch_start, dist_qp_sq);
                } else {
                    for (size_t j = 0; j < batch_count; ++j) {
                        est_distances[batch_start + j] = std::numeric_limits<float>::max();
                    }
                }
            }
        }


        bool warmup = (nn.size() < k);

        size_t prefetch_count = std::min(n_neighbors, constants::kPrefetchNeighbors);
        for (size_t i = 0; i < prefetch_count; ++i) {
            NodeId nid = nb.neighbor_ids[i];
            visited.prefetch_estimated(nid);
        }

        for (size_t i = 0; i < n_neighbors; ++i) {
            if (i + constants::kPrefetchStride < n_neighbors) {
                NodeId future_id = nb.neighbor_ids[i + constants::kPrefetchStride];
                if (future_id != INVALID_NODE) {
                    graph.prefetch_vertex(future_id);
                }
            }

            NodeId neighbor_id = nb.neighbor_ids[i];
            if (visited.check_and_mark_estimated(neighbor_id, query_id)) continue;

            // DABS candidate-queue filtering threshold
            float dabs_threshold = (nn.size() >= k)
                ? gamma_q * nn.worst_distance()
                : std::numeric_limits<float>::max();

            if (warmup) {
                float exact = exact_l2(neighbor_id);
                nn.push({neighbor_id, exact});
                if (exact < dabs_threshold) {
                    beam.push({exact, exact, neighbor_id});
                }
                graph.prefetch_vertex(neighbor_id);
                continue;
            }

            float est_dist = est_distances[i];
            float lower = lower_bounds[i];
            if (nn.size() >= k && lower >= nn.worst_distance()) continue;

            if (est_dist < nn.worst_distance()) {
                float exact = exact_l2(neighbor_id);
                nn.push({neighbor_id, exact});
                if (exact < dabs_threshold) {
                    beam.push({exact, lower, neighbor_id});
                }

                if (exact > constants::eps::kSmall) {
                    double r = est_dist / exact;
                    ratio_sum += r;
                    ratio_sq_sum += r * r;
                    ++ratio_count;
                    if (ratio_count >= gamma_warmup) {
                        double r_mean = ratio_sum / ratio_count;
                        double r_var = ratio_sq_sum / ratio_count - r_mean * r_mean;
                        double r_std = std::sqrt(std::max(r_var, 0.0));
                        gamma_q = std::clamp(
                            gamma * static_cast<float>(1.0 + gamma_beta * r_std),
                            gamma, gamma_max);
                    }
                }
            } else if (est_dist < dabs_threshold) {
                beam.push({est_dist, lower, neighbor_id});
            }
            graph.prefetch_vertex(neighbor_id);
        }
    }

    return nn.extract_sorted();
}

}
}
