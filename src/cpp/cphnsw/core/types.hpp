#pragma once

#include <cstdint>
#include <vector>
#include <queue>
#include <atomic>
#include <immintrin.h>

namespace cphnsw {

// ============================================================================
// Fundamental Types
// ============================================================================

/// Node ID type
using NodeId = uint32_t;

/// Invalid node ID sentinel
constexpr NodeId INVALID_NODE = 0xFFFFFFFF;

/// Distance type
using DistanceType = float;

// ============================================================================
// Spinlock: Per-node synchronization
// ============================================================================

/**
 * Spinlock: Lightweight lock for fine-grained per-node synchronization.
 *
 * Uses _mm_pause() hint for Hyperthreading friendliness.
 * Size: 1 byte (fits in struct padding).
 */
class Spinlock {
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;

public:
    void lock() noexcept {
        while (flag_.test_and_set(std::memory_order_acquire)) {
            _mm_pause();
        }
    }

    void unlock() noexcept {
        flag_.clear(std::memory_order_release);
    }

    // RAII guard
    class Guard {
        Spinlock& lock_;
    public:
        explicit Guard(Spinlock& lock) : lock_(lock) { lock_.lock(); }
        ~Guard() { lock_.unlock(); }
        Guard(const Guard&) = delete;
        Guard& operator=(const Guard&) = delete;
    };
};

static_assert(sizeof(Spinlock) == 1, "Spinlock must be 1 byte");

// ============================================================================
// Search Result
// ============================================================================

struct SearchResult {
    NodeId id;
    DistanceType distance;

    bool operator<(const SearchResult& other) const {
        return distance < other.distance;
    }

    bool operator>(const SearchResult& other) const {
        return distance > other.distance;
    }
};

// Min-heap: candidates ordered by distance (closest first)
using MinHeap = std::priority_queue<SearchResult, std::vector<SearchResult>,
                                     std::greater<SearchResult>>;

// Max-heap: results ordered by distance (furthest first for bounded set)
using MaxHeap = std::priority_queue<SearchResult, std::vector<SearchResult>,
                                     std::less<SearchResult>>;

}  // namespace cphnsw
