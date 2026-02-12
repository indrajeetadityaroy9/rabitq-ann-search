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
#include <cstring>
#include <type_traits>

namespace cphnsw {

// Colocated per-vertex data for cache-friendly access during search.
// All data needed to visit a vertex (code, neighbors, raw vector) is stored
// contiguously, reducing cache misses from 3 (separate vectors) to 1.
// BitWidth=1 uses original 1-bit types; BitWidth>1 uses Extended RaBitQ types.
template <size_t D, size_t R, size_t BitWidth = 1>
struct alignas(64) VertexData {
    using CodeType = std::conditional_t<BitWidth == 1,
        RaBitQCode<D>, NbitRaBitQCode<D, BitWidth>>;
    using NeighborBlockType = std::conditional_t<BitWidth == 1,
        FastScanNeighborBlock<D, R, 32>,
        NbitFastScanNeighborBlock<D, R, BitWidth, 32>>;

    CodeType code;
    NeighborBlockType neighbors;
    alignas(64) float vector[D];
};

template <size_t D, size_t R = 32, size_t BitWidth = 1>
class RaBitQGraph {
public:
    static_assert(R % 32 == 0, "R must be a multiple of 32 for AVX2 FastScan");

    using VertexDataType = VertexData<D, R, BitWidth>;
    using CodeType = typename VertexDataType::CodeType;
    using NeighborBlockType = typename VertexDataType::NeighborBlockType;

    static constexpr size_t DIMS = D;
    static constexpr size_t DEGREE = R;

    explicit RaBitQGraph(size_t dim, size_t capacity = 1024)
        : dim_(dim) {
        entry_point_.store(INVALID_NODE, std::memory_order_relaxed);
        vertices_.reserve(capacity);
    }

    NodeId add_node(const CodeType& code, const float* vec) {
        std::lock_guard<std::mutex> lock(graph_mutex_);

        NodeId id = static_cast<NodeId>(vertices_.size());
        vertices_.emplace_back();
        auto& vd = vertices_.back();
        vd.code = code;
        std::memcpy(vd.vector, vec, dim_ * sizeof(float));
        // Zero-pad remaining dimensions if dim_ < D
        if (dim_ < D) {
            std::memset(vd.vector + dim_, 0, (D - dim_) * sizeof(float));
        }

        NodeId expected = INVALID_NODE;
        entry_point_.compare_exchange_strong(
            expected, id,
            std::memory_order_release, std::memory_order_relaxed);

        return id;
    }

    void reserve(size_t capacity) {
        vertices_.reserve(capacity);
    }

    size_t size() const { return vertices_.size(); }
    bool empty() const { return vertices_.empty(); }
    size_t dim() const { return dim_; }

    NodeId entry_point() const {
        return entry_point_.load(std::memory_order_acquire);
    }

    void set_entry_point(NodeId id) {
        entry_point_.store(id, std::memory_order_release);
    }

    const CodeType& get_code(NodeId id) const { return vertices_[id].code; }
    CodeType& get_code(NodeId id) { return vertices_[id].code; }

    const float* get_vector(NodeId id) const { return vertices_[id].vector; }

    const NeighborBlockType& get_neighbors(NodeId id) const {
        return vertices_[id].neighbors;
    }

    NeighborBlockType& get_neighbors(NodeId id) {
        return vertices_[id].neighbors;
    }

    template <typename CodeData>
    void set_neighbor(NodeId node, size_t slot, NodeId neighbor_id,
                      const CodeData& neighbor_code_data,
                      const VertexAuxData& aux) {
        vertices_[node].neighbors.set_neighbor(slot, neighbor_id, neighbor_code_data, aux);
    }

    size_t neighbor_count(NodeId id) const {
        return (id < vertices_.size()) ? vertices_[id].neighbors.size() : 0;
    }

    // Prefetch a vertex's data into cache. Call this 1-2 iterations ahead of access.
    // Dynamically prefetches cache lines based on actual VertexData size.
    static constexpr size_t PREFETCH_LINES =
        (sizeof(VertexDataType) / CACHE_LINE_SIZE < 12)
            ? (sizeof(VertexDataType) / CACHE_LINE_SIZE) : 12;

    void prefetch_vertex(NodeId id) const {
        if (id >= vertices_.size()) return;
        const char* base = reinterpret_cast<const char*>(&vertices_[id]);
        for (size_t line = 0; line < PREFETCH_LINES; ++line) {
            prefetch_t<1>(base + line * CACHE_LINE_SIZE);
        }
    }

    float average_degree() const {
        if (vertices_.empty()) return 0.0f;
        size_t total = 0;
        for (const auto& vd : vertices_) total += vd.neighbors.size();
        return static_cast<float>(total) / vertices_.size();
    }

    size_t max_degree() const {
        size_t m = 0;
        for (const auto& vd : vertices_) {
            if (vd.neighbors.size() > m) m = vd.neighbors.size();
        }
        return m;
    }

    size_t count_isolated() const {
        size_t c = 0;
        for (const auto& vd : vertices_) {
            if (vd.neighbors.empty()) ++c;
        }
        return c;
    }

    NodeId find_medoid() const {
        if (empty()) return INVALID_NODE;

        size_t n = vertices_.size();
        std::vector<double> centroid(dim_, 0.0);
        for (size_t i = 0; i < n; ++i) {
            const float* v = vertices_[i].vector;
            for (size_t j = 0; j < dim_; ++j) {
                centroid[j] += v[j];
            }
        }
        double inv_n = 1.0 / n;
        for (size_t j = 0; j < dim_; ++j) centroid[j] *= inv_n;

        NodeId best = 0;
        double best_dist = std::numeric_limits<double>::max();
        for (size_t i = 0; i < n; ++i) {
            const float* v = vertices_[i].vector;
            double dist = 0.0;
            for (size_t j = 0; j < dim_; ++j) {
                double d = v[j] - centroid[j];
                dist += d * d;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best = static_cast<NodeId>(i);
            }
        }

        return best;
    }

    // Hub entry point selection (inspired by GATE, arXiv:2506.15986).
    // Selects the highest-degree node among the top-sqrt(n) closest to centroid.
    // Called after graph construction when degree information is available.
    // Falls back to medoid if graph has no edges.
    NodeId find_hub_entry() const {
        if (empty()) return INVALID_NODE;

        size_t n = vertices_.size();

        // Compute centroid
        std::vector<double> centroid(dim_, 0.0);
        for (size_t i = 0; i < n; ++i) {
            const float* v = vertices_[i].vector;
            for (size_t j = 0; j < dim_; ++j) {
                centroid[j] += v[j];
            }
        }
        double inv_n = 1.0 / n;
        for (size_t j = 0; j < dim_; ++j) centroid[j] *= inv_n;

        // Compute distances to centroid
        struct CentroidDist {
            NodeId id;
            double dist;
            bool operator<(const CentroidDist& o) const { return dist < o.dist; }
        };

        size_t top_k = std::max<size_t>(1, static_cast<size_t>(std::sqrt(static_cast<double>(n))));
        std::vector<CentroidDist> dists(n);
        for (size_t i = 0; i < n; ++i) {
            const float* v = vertices_[i].vector;
            double dist = 0.0;
            for (size_t j = 0; j < dim_; ++j) {
                double d = v[j] - centroid[j];
                dist += d * d;
            }
            dists[i] = {static_cast<NodeId>(i), dist};
        }

        // Partial sort to get top-sqrt(n) closest
        if (top_k < n) {
            std::partial_sort(dists.begin(), dists.begin() + top_k, dists.end());
        }

        // Among top_k closest, select highest degree
        NodeId best = dists[0].id;
        size_t best_degree = vertices_[best].neighbors.size();

        for (size_t i = 1; i < top_k && i < n; ++i) {
            NodeId cand = dists[i].id;
            size_t deg = vertices_[cand].neighbors.size();
            if (deg > best_degree) {
                best_degree = deg;
                best = cand;
            }
        }

        return best;
    }

private:
    size_t dim_;
    std::vector<VertexDataType, AlignedAllocator<VertexDataType>> vertices_;
    std::atomic<NodeId> entry_point_;
    mutable std::mutex graph_mutex_;
};

// Multi-bit graph aliases
template <size_t D, size_t R = 32, size_t BitWidth = 2>
using NbitRaBitQGraph = RaBitQGraph<D, R, BitWidth>;

}  // namespace cphnsw
