#pragma once

#include "../core/types.hpp"
#include "../core/codes.hpp"
#include "../core/memory.hpp"
#include "../core/adaptive_defaults.hpp"
#include "../distance/fastscan_layout.hpp"
#include "neighbor_selection.hpp"
#include <vector>
#include <array>
#include <queue>
#include <mutex>
#include <atomic>
#include <cstring>
#include <type_traits>

namespace cphnsw {

// Keep search-hot code+neighbors separate from raw vectors.
template <size_t D, size_t R, size_t BitWidth = 1>
struct alignas(64) VertexSearchData {
    using CodeType = std::conditional_t<BitWidth == 1,
        RaBitQCode<D>, NbitRaBitQCode<D, BitWidth>>;
    using NeighborBlockType = std::conditional_t<BitWidth == 1,
        FastScanNeighborBlock<D, R, 32>,
        NbitFastScanNeighborBlock<D, R, BitWidth, 32>>;

    CodeType code;
    NeighborBlockType neighbors;
};

template <size_t D, size_t R = 32, size_t BitWidth = 1>
class RaBitQGraph {
public:
    static_assert(R % 32 == 0, "R must be a multiple of 32 for AVX2 FastScan");

    using SearchDataType = VertexSearchData<D, R, BitWidth>;
    using CodeType = typename SearchDataType::CodeType;
    using NeighborBlockType = typename SearchDataType::NeighborBlockType;
    using RawVector = std::array<float, D>;

    static constexpr size_t DIMS = D;
    static constexpr size_t DEGREE = R;

    explicit RaBitQGraph(size_t dim, size_t capacity = 1024)
        : dim_(dim) {
        entry_point_.store(INVALID_NODE, std::memory_order_relaxed);
        search_data_.reserve(capacity);
        raw_vectors_.reserve(capacity);
        norm_sq_.reserve(capacity);
    }

    RaBitQGraph(RaBitQGraph&& other) noexcept
        : dim_(other.dim_)
        , search_data_(std::move(other.search_data_))
        , raw_vectors_(std::move(other.raw_vectors_))
        , norm_sq_(std::move(other.norm_sq_)) {
        entry_point_.store(other.entry_point_.load(std::memory_order_relaxed),
                           std::memory_order_relaxed);
    }

    RaBitQGraph& operator=(RaBitQGraph&& other) noexcept {
        if (this != &other) {
            dim_ = other.dim_;
            search_data_ = std::move(other.search_data_);
            raw_vectors_ = std::move(other.raw_vectors_);
            norm_sq_ = std::move(other.norm_sq_);
            entry_point_.store(other.entry_point_.load(std::memory_order_relaxed),
                               std::memory_order_relaxed);
        }
        return *this;
    }

    RaBitQGraph(const RaBitQGraph&) = delete;
    RaBitQGraph& operator=(const RaBitQGraph&) = delete;

    NodeId add_node(const CodeType& code, const float* vec) {
        std::lock_guard<std::mutex> lock(graph_mutex_);

        NodeId id = static_cast<NodeId>(search_data_.size());
        search_data_.emplace_back();
        search_data_.back().code = code;

        raw_vectors_.emplace_back();
        std::memcpy(raw_vectors_.back().data(), vec, dim_ * sizeof(float));
        if (dim_ < D) {
            std::memset(raw_vectors_.back().data() + dim_, 0, (D - dim_) * sizeof(float));
        }

        float nsq = 0.0f;
        const float* stored = raw_vectors_.back().data();
        for (size_t j = 0; j < dim_; ++j) nsq += stored[j] * stored[j];
        norm_sq_.push_back(nsq);

        NodeId expected = INVALID_NODE;
        entry_point_.compare_exchange_strong(
            expected, id,
            std::memory_order_release, std::memory_order_relaxed);

        return id;
    }

    void reserve(size_t capacity) {
        search_data_.reserve(capacity);
        raw_vectors_.reserve(capacity);
        norm_sq_.reserve(capacity);
    }

    size_t size() const { return search_data_.size(); }
    bool empty() const { return search_data_.empty(); }
    size_t dim() const { return dim_; }

    NodeId entry_point() const {
        return entry_point_.load(std::memory_order_acquire);
    }

    void set_entry_point(NodeId id) {
        entry_point_.store(id, std::memory_order_release);
    }

    const CodeType& get_code(NodeId id) const { return search_data_[id].code; }
    CodeType& get_code(NodeId id) { return search_data_[id].code; }

    const float* get_vector(NodeId id) const { return raw_vectors_[id].data(); }

    float get_norm_sq(NodeId id) const { return norm_sq_[id]; }

    void prefetch_norm(NodeId id) const {
        if (id < norm_sq_.size()) {
            prefetch_t<1>(&norm_sq_[id]);
        }
    }

    void recompute_norms() {
        norm_sq_.resize(raw_vectors_.size());
        for (size_t i = 0; i < raw_vectors_.size(); ++i) {
            float nsq = 0.0f;
            const float* v = raw_vectors_[i].data();
            for (size_t j = 0; j < dim_; ++j) nsq += v[j] * v[j];
            norm_sq_[i] = nsq;
        }
    }

    const NeighborBlockType& get_neighbors(NodeId id) const {
        return search_data_[id].neighbors;
    }

    NeighborBlockType& get_neighbors(NodeId id) {
        return search_data_[id].neighbors;
    }

    size_t neighbor_count(NodeId id) const {
        return (id < search_data_.size()) ? search_data_[id].neighbors.size() : 0;
    }

    static constexpr size_t PREFETCH_LINES =
        (sizeof(SearchDataType) / CACHE_LINE_SIZE < adaptive_defaults::prefetch_line_cap())
            ? (sizeof(SearchDataType) / CACHE_LINE_SIZE) : adaptive_defaults::prefetch_line_cap();

    void prefetch_vertex(NodeId id) const {
        if (id >= search_data_.size()) return;
        const char* base = reinterpret_cast<const char*>(&search_data_[id]);
        for (size_t line = 0; line < PREFETCH_LINES; ++line) {
            prefetch_t<1>(base + line * CACHE_LINE_SIZE);
        }
    }

    void prefetch_vector(NodeId id) const {
        if (id >= raw_vectors_.size()) return;
        const char* base = reinterpret_cast<const char*>(raw_vectors_[id].data());
        constexpr size_t VEC_LINES = (D * sizeof(float) + CACHE_LINE_SIZE - 1) / CACHE_LINE_SIZE;
        constexpr size_t MAX_VEC_LINES = (VEC_LINES < 4) ? VEC_LINES : 4;
        for (size_t line = 0; line < MAX_VEC_LINES; ++line) {
            prefetch_t<1>(base + line * CACHE_LINE_SIZE);
        }
    }

    float average_degree() const {
        if (search_data_.empty()) return 0.0f;
        size_t total = 0;
        for (const auto& sd : search_data_) total += sd.neighbors.size();
        return static_cast<float>(total) / search_data_.size();
    }

    size_t max_degree() const {
        size_t m = 0;
        for (const auto& sd : search_data_) {
            if (sd.neighbors.size() > m) m = sd.neighbors.size();
        }
        return m;
    }

    NodeId find_medoid() const {
        if (empty()) return INVALID_NODE;

        size_t n = raw_vectors_.size();
        std::vector<double> centroid(dim_, 0.0);
        for (size_t i = 0; i < n; ++i) {
            const float* v = raw_vectors_[i].data();
            for (size_t j = 0; j < dim_; ++j) {
                centroid[j] += v[j];
            }
        }
        double inv_n = 1.0 / n;
        for (size_t j = 0; j < dim_; ++j) centroid[j] *= inv_n;

        NodeId best = 0;
        double best_dist = std::numeric_limits<double>::max();
        for (size_t i = 0; i < n; ++i) {
            const float* v = raw_vectors_[i].data();
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

    struct Permutation {
        std::vector<NodeId> old_to_new;
        std::vector<NodeId> new_to_old;
    };

    Permutation reorder_bfs(NodeId entry) {
        size_t n = search_data_.size();
        Permutation perm;
        perm.old_to_new.resize(n, INVALID_NODE);
        perm.new_to_old.resize(n);

        std::vector<bool> visited(n, false);
        std::queue<NodeId> bfs_queue;
        NodeId next_new_id = 0;

        auto run_bfs = [&](NodeId start) {
            if (start >= n || visited[start]) return;
            bfs_queue.push(start);
            visited[start] = true;
            while (!bfs_queue.empty()) {
                NodeId curr = bfs_queue.front();
                bfs_queue.pop();
                perm.old_to_new[curr] = next_new_id;
                perm.new_to_old[next_new_id] = curr;
                next_new_id++;

                const auto& nb = search_data_[curr].neighbors;
                for (size_t i = 0; i < nb.size(); ++i) {
                    NodeId neighbor = nb.neighbor_ids[i];
                    if (neighbor != INVALID_NODE && neighbor < n && !visited[neighbor]) {
                        visited[neighbor] = true;
                        bfs_queue.push(neighbor);
                    }
                }
            }
        };

        run_bfs(entry);

        for (size_t i = 0; i < n; ++i) {
            if (!visited[i]) {
                run_bfs(static_cast<NodeId>(i));
            }
        }

        std::vector<SearchDataType, AlignedAllocator<SearchDataType>> new_search(n);
        std::vector<RawVector> new_vectors(n);
        AlignedVector<float> new_norms(n);
        for (size_t new_id = 0; new_id < n; ++new_id) {
            NodeId old_id = perm.new_to_old[new_id];
            new_search[new_id] = std::move(search_data_[old_id]);
            new_vectors[new_id] = raw_vectors_[old_id];
            new_norms[new_id] = norm_sq_[old_id];
        }

        for (size_t i = 0; i < n; ++i) {
            auto& nb = new_search[i].neighbors;
            for (size_t j = 0; j < R; ++j) {
                NodeId old_nid = nb.neighbor_ids[j];
                if (old_nid != INVALID_NODE && old_nid < n) {
                    nb.neighbor_ids[j] = perm.old_to_new[old_nid];
                }
            }
        }

        search_data_ = std::move(new_search);
        raw_vectors_ = std::move(new_vectors);
        norm_sq_ = std::move(new_norms);

        NodeId old_ep = entry_point_.load(std::memory_order_relaxed);
        if (old_ep != INVALID_NODE && old_ep < n) {
            entry_point_.store(perm.old_to_new[old_ep], std::memory_order_release);
        }

        return perm;
    }

    NodeId find_hub_entry() const {
        if (empty()) return INVALID_NODE;

        size_t n = raw_vectors_.size();

        std::vector<double> centroid(dim_, 0.0);
        for (size_t i = 0; i < n; ++i) {
            const float* v = raw_vectors_[i].data();
            for (size_t j = 0; j < dim_; ++j) {
                centroid[j] += v[j];
            }
        }
        double inv_n = 1.0 / n;
        for (size_t j = 0; j < dim_; ++j) centroid[j] *= inv_n;

        struct CentroidDist {
            NodeId id;
            double dist;
            bool operator<(const CentroidDist& o) const { return dist < o.dist; }
        };

        size_t top_k = std::max<size_t>(1, static_cast<size_t>(std::sqrt(static_cast<double>(n))));
        std::vector<CentroidDist> dists(n);
        for (size_t i = 0; i < n; ++i) {
            const float* v = raw_vectors_[i].data();
            double dist = 0.0;
            for (size_t j = 0; j < dim_; ++j) {
                double d = v[j] - centroid[j];
                dist += d * d;
            }
            dists[i] = {static_cast<NodeId>(i), dist};
        }

        if (top_k < n) {
            std::partial_sort(dists.begin(), dists.begin() + top_k, dists.end());
        }

        NodeId best = dists[0].id;
        size_t best_degree = search_data_[best].neighbors.size();

        for (size_t i = 1; i < top_k && i < n; ++i) {
            NodeId cand = dists[i].id;
            size_t deg = search_data_[cand].neighbors.size();
            if (deg > best_degree) {
                best_degree = deg;
                best = cand;
            }
        }

        return best;
    }

private:
    size_t dim_;
    std::vector<SearchDataType, AlignedAllocator<SearchDataType>> search_data_;
    std::vector<RawVector> raw_vectors_;
    AlignedVector<float> norm_sq_;
    std::atomic<NodeId> entry_point_;
    mutable std::mutex graph_mutex_;
};

}  // namespace cphnsw
