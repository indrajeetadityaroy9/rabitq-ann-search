#pragma once

#include "../core/types.hpp"
#include "../core/codes.hpp"
#include "../core/memory.hpp"
#include "../distance/fastscan_layout.hpp"
#include "neighbor_selection.hpp"
#include "visitation_table.hpp"
#include <vector>
#include <mutex>
#include <atomic>
#include <stdexcept>
#include <random>

namespace cphnsw {

template <size_t D, size_t R = 32>
class RaBitQGraph {
public:
    static_assert(R % 32 == 0, "R must be a multiple of 32 for AVX2 FastScan");

    using CodeType = RaBitQCode<D>;
    using NeighborBlockType = FastScanNeighborBlock<D, R, 32>;

    static constexpr size_t DIMS = D;
    static constexpr size_t DEGREE = R;

    explicit RaBitQGraph(size_t dim, size_t capacity = 1024)
        : dim_(dim) {
        entry_point_.store(INVALID_NODE, std::memory_order_relaxed);
        codes_.reserve(capacity);
        neighbors_.reserve(capacity);
        vectors_.reserve(capacity);
    }

    NodeId add_node(const CodeType& code, const float* vec) {
        std::lock_guard<std::mutex> lock(graph_mutex_);

        NodeId id = static_cast<NodeId>(codes_.size());
        codes_.push_back(code);
        neighbors_.emplace_back();

        AlignedVector<float> v(vec, vec + dim_);
        vectors_.push_back(std::move(v));

        NodeId expected = INVALID_NODE;
        entry_point_.compare_exchange_strong(
            expected, id,
            std::memory_order_release, std::memory_order_relaxed);

        return id;
    }

    void reserve(size_t capacity) {
        codes_.reserve(capacity);
        neighbors_.reserve(capacity);
        vectors_.reserve(capacity);
    }

    size_t size() const { return codes_.size(); }
    bool empty() const { return codes_.empty(); }
    size_t dim() const { return dim_; }

    NodeId entry_point() const {
        return entry_point_.load(std::memory_order_acquire);
    }

    void set_entry_point(NodeId id) {
        entry_point_.store(id, std::memory_order_release);
    }

    const CodeType& get_code(NodeId id) const { return codes_[id]; }
    CodeType& get_code(NodeId id) { return codes_[id]; }

    const float* get_vector(NodeId id) const { return vectors_[id].data(); }

    const NeighborBlockType& get_neighbors(NodeId id) const {
        return neighbors_[id];
    }

    NeighborBlockType& get_neighbors(NodeId id) {
        return neighbors_[id];
    }

    void set_neighbor(NodeId node, size_t slot, NodeId neighbor_id,
                      const BinaryCodeStorage<D>& neighbor_signs,
                      const VertexAuxData& aux) {
        neighbors_[node].set_neighbor(slot, neighbor_id, neighbor_signs, aux);
    }

    size_t neighbor_count(NodeId id) const {
        return (id < neighbors_.size()) ? neighbors_[id].size() : 0;
    }

    float average_degree() const {
        if (neighbors_.empty()) return 0.0f;
        size_t total = 0;
        for (const auto& nb : neighbors_) total += nb.size();
        return static_cast<float>(total) / neighbors_.size();
    }

    size_t max_degree() const {
        size_t m = 0;
        for (const auto& nb : neighbors_) {
            if (nb.size() > m) m = nb.size();
        }
        return m;
    }

    size_t count_isolated() const {
        size_t c = 0;
        for (const auto& nb : neighbors_) {
            if (nb.empty()) ++c;
        }
        return c;
    }

    NodeId find_medoid() const {
        if (empty()) return INVALID_NODE;

        std::vector<double> centroid(dim_, 0.0);
        for (size_t i = 0; i < vectors_.size(); ++i) {
            for (size_t j = 0; j < dim_; ++j) {
                centroid[j] += vectors_[i][j];
            }
        }
        double inv_n = 1.0 / vectors_.size();
        for (size_t j = 0; j < dim_; ++j) centroid[j] *= inv_n;

        NodeId best = 0;
        double best_dist = std::numeric_limits<double>::max();
        for (size_t i = 0; i < vectors_.size(); ++i) {
            double dist = 0.0;
            for (size_t j = 0; j < dim_; ++j) {
                double d = vectors_[i][j] - centroid[j];
                dist += d * d;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best = static_cast<NodeId>(i);
            }
        }

        return best;
    }

private:
    size_t dim_;
    std::vector<CodeType> codes_;
    std::vector<NeighborBlockType> neighbors_;
    std::vector<AlignedVector<float>> vectors_;
    std::atomic<NodeId> entry_point_;
    mutable std::mutex graph_mutex_;
};

using RaBitQGraph128 = RaBitQGraph<128, 32>;
using RaBitQGraph256 = RaBitQGraph<256, 32>;
using RaBitQGraph1024 = RaBitQGraph<1024, 32>;

}  // namespace cphnsw