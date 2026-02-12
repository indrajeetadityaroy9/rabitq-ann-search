#pragma once

#include <cstdint>
#include <vector>
#include <queue>
#include <atomic>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace cphnsw {

using NodeId = uint32_t;
constexpr NodeId INVALID_NODE = 0xFFFFFFFF;
using DistanceType = float;

inline void cpu_relax() {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_pause();
#elif defined(__aarch64__)
    asm volatile("yield" ::: "memory");
#else
#endif
}

class Spinlock {
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;

public:
    void lock() noexcept {
        while (flag_.test_and_set(std::memory_order_acquire)) {
            cpu_relax();
        }
    }

    void unlock() noexcept {
        flag_.clear(std::memory_order_release);
    }

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

using MinHeap = std::priority_queue<SearchResult, std::vector<SearchResult>,
                                     std::greater<SearchResult>>;

using MaxHeap = std::priority_queue<SearchResult, std::vector<SearchResult>,
                                     std::less<SearchResult>>;

}  // namespace cphnsw