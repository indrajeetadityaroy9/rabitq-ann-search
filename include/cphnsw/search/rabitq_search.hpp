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

    static std::vector<SearchResult> search(
        const QueryType& query,
        const float* raw_query,
        const Graph& graph,
        size_t k,
        float gamma,
        TwoLevelVisitationTable& visited,
        size_t ef_cap = 4096,
        NodeId entry = INVALID_NODE)
    {
        if (graph.empty()) return {};
        if (k == 0) k = 10;

        NodeId ep = (entry != INVALID_NODE) ? entry : graph.entry_point();
        if (ep == INVALID_NODE) return {};

        uint64_t query_id = visited.new_query();

        std::priority_queue<BeamEntry, std::vector<BeamEntry>,
                           std::greater<BeamEntry>> beam;
        BoundedMaxHeap<SearchResult> nn(k);

        float ep_est;
        if constexpr (BitWidth == 1) {
            ep_est = Policy::compute_distance(query, graph.get_code(ep));
        } else {
            const auto& nbit_code = graph.get_code(ep);
            float dist_o = nbit_code.dist_to_centroid;
            float ip_qo = nbit_code.ip_quantized_original;
            constexpr size_t NUM_SUB_SEGMENTS = (D + 3) / 4;
            uint32_t fastscan_sum = 0;
            auto msb_signs = nbit_code.codes.msb_as_binary();
            for (size_t j = 0; j < NUM_SUB_SEGMENTS; ++j) {
                size_t bit_base = j * 4;
                uint8_t pattern = 0;
                for (size_t b = 0; b < 4 && (bit_base + b) < D; ++b) {
                    if (msb_signs.get_bit(bit_base + b)) {
                        pattern |= (1 << b);
                    }
                }
                fastscan_sum += query.lut[j][pattern];
            }
            float ip_approx = query.coeff_fastscan * static_cast<float>(fastscan_sum)
                            + query.coeff_popcount * static_cast<float>(nbit_code.msb_popcount)
                            + query.coeff_constant;
            float ip_est = (ip_qo > 1e-10f) ? ip_approx / ip_qo : 0.0f;
            ep_est = dist_o * dist_o + query.query_norm_sq
                   - 2.0f * dist_o * query.query_norm * ip_est;
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

            if (nn.size() >= k && current.est_distance > (1.0f + gamma) * nn.worst_distance()) break;

            if (nn.size() >= k && current.lower_bound > nn.worst_distance()) continue;

            if (!beam.empty()) {
                graph.prefetch_vertex(beam.top().id);
                BeamEntry first_beam = beam.top();
                beam.pop();
                if (!beam.empty()) {
                    graph.prefetch_vertex(beam.top().id);
                }
                beam.push(first_beam);
            }

            visited.check_and_mark_visited(current.id, query_id);
            prefetch_t<0>(graph.get_vector(current.id));

            const float* vertex_vec = graph.get_vector(current.id);
            float exact_dist = l2_distance_simd<D>(raw_query, vertex_vec);

            nn.push({current.id, exact_dist});

            if (nn.size() >= k && exact_dist > (1.0f + gamma) * nn.worst_distance()) break;

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
                if (visited.check_and_mark_estimated(neighbor_id, query_id)) continue;

                float est_dist = est_distances[i];
                float lower = lower_bounds[i];
                if (nn.size() >= k && lower >= nn.worst_distance()) continue;

                beam.push({est_dist, lower, neighbor_id});
                graph.prefetch_vertex(neighbor_id);
            }

            if (beam.size() > ef_cap * 2) {
                std::priority_queue<BeamEntry, std::vector<BeamEntry>,
                                   std::greater<BeamEntry>> new_beam;
                size_t kept = 0;
                while (!beam.empty() && kept < ef_cap) {
                    new_beam.push(beam.top());
                    beam.pop();
                    kept++;
                }
                beam = std::move(new_beam);
            }
        }

        return nn.extract_sorted();
    }

    static std::vector<SearchResult> search(
        const QueryType& query,
        const float* raw_query,
        const Graph& graph,
        size_t k,
        float gamma)
    {
        thread_local TwoLevelVisitationTable visited(0);
        if (visited.capacity() < graph.size()) {
            visited.resize(graph.size() + 1024);
        }
        return search(query, raw_query, graph, k, gamma, visited);
    }

    static std::vector<SearchResult> search_from(
        const QueryType& query,
        const float* raw_query,
        const Graph& graph,
        NodeId entry,
        size_t k,
        float gamma)
    {
        thread_local TwoLevelVisitationTable visited(0);
        if (visited.capacity() < graph.size()) {
            visited.resize(graph.size() + 1024);
        }
        return search(query, raw_query, graph, k, gamma, visited, 4096, entry);
    }
};

struct ExactL2Policy {
    template <size_t D>
    static float compute(const float* a, const float* b) {
        return l2_distance_simd<D>(a, b);
    }
};

template <size_t D, size_t R = 32, size_t BitWidth = 1, typename DistancePolicy = ExactL2Policy>
class GraphSearchEngine {
public:
    using Graph = RaBitQGraph<D, R, BitWidth>;

    struct BeamEntry {
        float distance;
        NodeId id;
        bool operator>(const BeamEntry& o) const { return distance > o.distance; }
    };

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
        BoundedMaxHeap<SearchResult> nn(k);

        float ep_dist = DistancePolicy::template compute<D>(raw_query, graph.get_vector(ep));
        beam.push({ep_dist, ep});
        visited.check_and_mark(ep, query_id);

        while (!beam.empty()) {
            BeamEntry current = beam.top();
            beam.pop();

            if (nn.size() >= k && current.distance > nn.top().distance) break;

            if (!beam.empty()) {
                graph.prefetch_vertex(beam.top().id);
            }

            nn.push({current.id, current.distance});

            const auto& nb = graph.get_neighbors(current.id);
            size_t n_neighbors = nb.size();

            for (size_t i = 0; i < n_neighbors; ++i) {
                if (i + 2 < n_neighbors) {
                    NodeId prefetch_id = nb.neighbor_ids[i + 2];
                    if (prefetch_id != INVALID_NODE) {
                        prefetch_t<0>(graph.get_vector(prefetch_id));
                    }
                }

                NodeId neighbor_id = nb.neighbor_ids[i];
                if (neighbor_id == INVALID_NODE) continue;
                if (visited.check_and_mark(neighbor_id, query_id)) continue;

                float dist = DistancePolicy::template compute<D>(raw_query, graph.get_vector(neighbor_id));

                if (nn.size() >= k && dist >= nn.top().distance) continue;

                beam.push({dist, neighbor_id});
            }

            if (beam.size() > ef * 4) {
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

        return nn.extract_sorted();
    }
};

template <size_t D, size_t R = 32, size_t BitWidth = 1>
class QuantizedBuildSearchEngine {
public:
    using Graph = RaBitQGraph<D, R, BitWidth>;

    template <typename EncoderType>
    static std::vector<SearchResult> search(
        const float* raw_query,
        const Graph& graph,
        const EncoderType& encoder,
        size_t ef, size_t k,
        TwoLevelVisitationTable& visited,
        float error_epsilon,
        NodeId entry = INVALID_NODE)
    {
        if (graph.empty()) return {};
        auto query = encoder.encode_query(raw_query);
        query.error_epsilon = error_epsilon;
        float gamma = 0.5f;
        return RaBitQSearchEngine<D, R, BitWidth>::search(
            query, raw_query, graph, k, gamma, visited, ef, entry);
    }

    template <typename EncoderType>
    static std::vector<SearchResult> search(
        const float* raw_query,
        const Graph& graph,
        const EncoderType& encoder,
        size_t ef, size_t k,
        float error_epsilon)
    {
        if (graph.empty()) return {};
        auto query = encoder.encode_query(raw_query);
        query.error_epsilon = error_epsilon;
        float gamma = 0.5f;
        thread_local TwoLevelVisitationTable visited(0);
        if (visited.capacity() < graph.size()) {
            visited.resize(graph.size() + 1024);
        }
        return RaBitQSearchEngine<D, R, BitWidth>::search(
            query, raw_query, graph, k, gamma, visited, ef);
    }
};

}  // namespace cphnsw
