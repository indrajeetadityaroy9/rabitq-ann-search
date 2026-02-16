#pragma once

#include "../core/types.hpp"
#include "../core/memory.hpp"
#include <memory>

namespace cphnsw {

// Thread-local visitation tables — no atomics needed since each thread
// owns its own table instance (used via thread_local in search paths).

class VisitationTable {
public:
    explicit VisitationTable(size_t capacity)
        : capacity_(capacity), current_epoch_(0) {
        epochs_ = std::make_unique<uint64_t[]>(capacity);
        std::memset(epochs_.get(), 0, capacity * sizeof(uint64_t));
    }

    VisitationTable(const VisitationTable&) = delete;
    VisitationTable& operator=(const VisitationTable&) = delete;

    VisitationTable(VisitationTable&& other) noexcept
        : epochs_(std::move(other.epochs_)),
          capacity_(other.capacity_),
          current_epoch_(other.current_epoch_) {
        other.capacity_ = 0;
    }

    uint64_t new_query() const {
        return ++current_epoch_;
    }

    bool check_and_mark(NodeId node_id, uint64_t query_id) const {
        if (node_id >= capacity_) return true;
        if (epochs_[node_id] == query_id) return true;
        epochs_[node_id] = query_id;
        return false;
    }

    void resize(size_t new_capacity) {
        if (new_capacity <= capacity_) return;
        auto new_epochs = std::make_unique<uint64_t[]>(new_capacity);
        std::memcpy(new_epochs.get(), epochs_.get(), capacity_ * sizeof(uint64_t));
        std::memset(new_epochs.get() + capacity_, 0, (new_capacity - capacity_) * sizeof(uint64_t));
        epochs_ = std::move(new_epochs);
        capacity_ = new_capacity;
    }

    size_t capacity() const { return capacity_; }

private:
    mutable std::unique_ptr<uint64_t[]> epochs_;
    size_t capacity_;
    mutable uint64_t current_epoch_;
};

class TwoLevelVisitationTable {
public:
    explicit TwoLevelVisitationTable(size_t capacity)
        : capacity_(capacity), current_epoch_(0) {
        estimated_ = std::make_unique<uint64_t[]>(capacity);
        visited_ = std::make_unique<uint64_t[]>(capacity);
        std::memset(estimated_.get(), 0, capacity * sizeof(uint64_t));
        std::memset(visited_.get(), 0, capacity * sizeof(uint64_t));
    }

    TwoLevelVisitationTable(const TwoLevelVisitationTable&) = delete;
    TwoLevelVisitationTable& operator=(const TwoLevelVisitationTable&) = delete;

    TwoLevelVisitationTable(TwoLevelVisitationTable&& other) noexcept
        : estimated_(std::move(other.estimated_)),
          visited_(std::move(other.visited_)),
          capacity_(other.capacity_),
          current_epoch_(other.current_epoch_) {
        other.capacity_ = 0;
    }

    uint64_t new_query() const {
        return ++current_epoch_;
    }

    bool check_and_mark_estimated(NodeId node_id, uint64_t query_id) const {
        if (node_id >= capacity_) return true;
        if (estimated_[node_id] == query_id) return true;
        estimated_[node_id] = query_id;
        return false;
    }

    bool check_and_mark_visited(NodeId node_id, uint64_t query_id) const {
        if (node_id >= capacity_) return true;
        if (visited_[node_id] == query_id) return true;
        visited_[node_id] = query_id;
        return false;
    }

    bool is_visited(NodeId node_id, uint64_t query_id) const {
        if (node_id >= capacity_) return true;
        return visited_[node_id] == query_id;
    }

    void resize(size_t new_capacity) {
        if (new_capacity <= capacity_) return;
        auto new_est = std::make_unique<uint64_t[]>(new_capacity);
        auto new_vis = std::make_unique<uint64_t[]>(new_capacity);
        std::memcpy(new_est.get(), estimated_.get(), capacity_ * sizeof(uint64_t));
        std::memcpy(new_vis.get(), visited_.get(), capacity_ * sizeof(uint64_t));
        std::memset(new_est.get() + capacity_, 0, (new_capacity - capacity_) * sizeof(uint64_t));
        std::memset(new_vis.get() + capacity_, 0, (new_capacity - capacity_) * sizeof(uint64_t));
        estimated_ = std::move(new_est);
        visited_ = std::move(new_vis);
        capacity_ = new_capacity;
    }

    size_t capacity() const { return capacity_; }

private:
    mutable std::unique_ptr<uint64_t[]> estimated_;
    mutable std::unique_ptr<uint64_t[]> visited_;
    size_t capacity_;
    mutable uint64_t current_epoch_;
};

}  // namespace cphnsw
