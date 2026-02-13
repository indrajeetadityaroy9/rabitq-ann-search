#pragma once

#include "../core/types.hpp"
#include "../core/memory.hpp"
#include <atomic>
#include <memory>

namespace cphnsw {

class VisitationTable {
public:
    explicit VisitationTable(size_t capacity)
        : capacity_(capacity), current_epoch_(0) {
        epochs_ = std::make_unique<std::atomic<uint64_t>[]>(capacity);
        for (size_t i = 0; i < capacity_; ++i) {
            epochs_[i].store(0, std::memory_order_relaxed);
        }
    }

    VisitationTable(const VisitationTable&) = delete;
    VisitationTable& operator=(const VisitationTable&) = delete;

    VisitationTable(VisitationTable&& other) noexcept
        : epochs_(std::move(other.epochs_)),
          capacity_(other.capacity_),
          current_epoch_(other.current_epoch_.load(std::memory_order_relaxed)) {
        other.capacity_ = 0;
    }

    uint64_t new_query() const {
        return current_epoch_.fetch_add(1, std::memory_order_relaxed) + 1;
    }

    bool check_and_mark(NodeId node_id, uint64_t query_id) const {
        if (node_id >= capacity_) return true;

        uint64_t expected = epochs_[node_id].load(std::memory_order_relaxed);

        if (expected == query_id) {
            return true;
        }

        return !epochs_[node_id].compare_exchange_strong(
            expected, query_id,
            std::memory_order_relaxed,
            std::memory_order_relaxed);
    }

    void resize(size_t new_capacity) {
        if (new_capacity <= capacity_) return;

        auto new_epochs = std::make_unique<std::atomic<uint64_t>[]>(new_capacity);

        for (size_t i = 0; i < capacity_; ++i) {
            new_epochs[i].store(epochs_[i].load(std::memory_order_relaxed),
                               std::memory_order_relaxed);
        }
        for (size_t i = capacity_; i < new_capacity; ++i) {
            new_epochs[i].store(0, std::memory_order_relaxed);
        }

        epochs_ = std::move(new_epochs);
        capacity_ = new_capacity;
    }

    size_t capacity() const { return capacity_; }

private:
    mutable std::unique_ptr<std::atomic<uint64_t>[]> epochs_;
    size_t capacity_;
    mutable std::atomic<uint64_t> current_epoch_;
};

class TwoLevelVisitationTable {
public:
    explicit TwoLevelVisitationTable(size_t capacity)
        : capacity_(capacity), current_epoch_(0) {
        estimated_ = std::make_unique<std::atomic<uint64_t>[]>(capacity);
        visited_ = std::make_unique<std::atomic<uint64_t>[]>(capacity);
        for (size_t i = 0; i < capacity_; ++i) {
            estimated_[i].store(0, std::memory_order_relaxed);
            visited_[i].store(0, std::memory_order_relaxed);
        }
    }

    TwoLevelVisitationTable(const TwoLevelVisitationTable&) = delete;
    TwoLevelVisitationTable& operator=(const TwoLevelVisitationTable&) = delete;

    TwoLevelVisitationTable(TwoLevelVisitationTable&& other) noexcept
        : estimated_(std::move(other.estimated_)),
          visited_(std::move(other.visited_)),
          capacity_(other.capacity_),
          current_epoch_(other.current_epoch_.load(std::memory_order_relaxed)) {
        other.capacity_ = 0;
    }

    uint64_t new_query() const {
        return current_epoch_.fetch_add(1, std::memory_order_relaxed) + 1;
    }

    bool check_and_mark_estimated(NodeId node_id, uint64_t query_id) const {
        if (node_id >= capacity_) return true;
        uint64_t expected = estimated_[node_id].load(std::memory_order_relaxed);
        if (expected == query_id) return true;
        return !estimated_[node_id].compare_exchange_strong(
            expected, query_id,
            std::memory_order_relaxed, std::memory_order_relaxed);
    }

    bool check_and_mark_visited(NodeId node_id, uint64_t query_id) const {
        if (node_id >= capacity_) return true;
        uint64_t expected = visited_[node_id].load(std::memory_order_relaxed);
        if (expected == query_id) return true;
        return !visited_[node_id].compare_exchange_strong(
            expected, query_id,
            std::memory_order_relaxed, std::memory_order_relaxed);
    }

    bool is_visited(NodeId node_id, uint64_t query_id) const {
        if (node_id >= capacity_) return true;
        return visited_[node_id].load(std::memory_order_relaxed) == query_id;
    }

    void resize(size_t new_capacity) {
        if (new_capacity <= capacity_) return;
        auto new_est = std::make_unique<std::atomic<uint64_t>[]>(new_capacity);
        auto new_vis = std::make_unique<std::atomic<uint64_t>[]>(new_capacity);
        for (size_t i = 0; i < capacity_; ++i) {
            new_est[i].store(estimated_[i].load(std::memory_order_relaxed),
                            std::memory_order_relaxed);
            new_vis[i].store(visited_[i].load(std::memory_order_relaxed),
                            std::memory_order_relaxed);
        }
        for (size_t i = capacity_; i < new_capacity; ++i) {
            new_est[i].store(0, std::memory_order_relaxed);
            new_vis[i].store(0, std::memory_order_relaxed);
        }
        estimated_ = std::move(new_est);
        visited_ = std::move(new_vis);
        capacity_ = new_capacity;
    }

    size_t capacity() const { return capacity_; }

private:
    mutable std::unique_ptr<std::atomic<uint64_t>[]> estimated_;
    mutable std::unique_ptr<std::atomic<uint64_t>[]> visited_;
    size_t capacity_;
    mutable std::atomic<uint64_t> current_epoch_;
};

}  // namespace cphnsw