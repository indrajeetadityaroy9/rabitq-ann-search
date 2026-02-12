#pragma once

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <new>
#include <vector>
#include <type_traits>
#include <immintrin.h>

namespace cphnsw {

#ifndef CPHNSW_CACHE_LINE_SIZE_DEFINED
#define CPHNSW_CACHE_LINE_SIZE_DEFINED
constexpr size_t CACHE_LINE_SIZE = 64;
#endif
constexpr size_t SIMD_ALIGNMENT = 64;

template <typename T, size_t Alignment = SIMD_ALIGNMENT>
class AlignedAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    static_assert((Alignment & (Alignment - 1)) == 0, "Alignment must be power of 2");
    static_assert(Alignment >= alignof(T), "Alignment must be >= alignof(T)");

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    AlignedAllocator() noexcept = default;

    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    pointer allocate(size_type n) {
        if (n == 0) return nullptr;
        size_t bytes = n * sizeof(T);
        bytes = ((bytes + Alignment - 1) / Alignment) * Alignment;
        void* ptr = std::aligned_alloc(Alignment, bytes);
        if (!ptr) throw std::bad_alloc();
        return static_cast<pointer>(ptr);
    }

    void deallocate(pointer p, size_type) noexcept {
        if (p) std::free(p);
    }

    template <typename U, size_t A>
    bool operator==(const AlignedAllocator<U, A>&) const noexcept {
        return Alignment == A;
    }

    template <typename U, size_t A>
    bool operator!=(const AlignedAllocator<U, A>&) const noexcept {
        return Alignment != A;
    }
};

template <typename T, size_t Alignment = SIMD_ALIGNMENT>
using AlignedVector = std::vector<T, AlignedAllocator<T, Alignment>>;

template <typename T>
struct MemoryTraits {
    static constexpr size_t alignment = alignof(T);
    static constexpr size_t size = sizeof(T);
    static constexpr bool is_trivially_copyable = std::is_trivially_copyable_v<T>;
    static constexpr bool is_cache_aligned = (alignment >= CACHE_LINE_SIZE);
    static constexpr bool is_simd_aligned = (alignment >= SIMD_ALIGNMENT);
};

inline void prefetch(const void* addr) {
    __builtin_prefetch(addr, 0, 3);
}

inline void prefetch_write(void* addr) {
    __builtin_prefetch(addr, 1, 3);
}

template <int Locality = 3>
inline void prefetch_t(const void* addr) {
    __builtin_prefetch(addr, 0, Locality);
}

struct alignas(CACHE_LINE_SIZE) CacheLinePad {
    char padding[CACHE_LINE_SIZE];
};

template <typename T>
struct alignas(CACHE_LINE_SIZE) CacheLineIsolated {
    T value;
    static constexpr size_t REMAINDER = sizeof(T) % CACHE_LINE_SIZE;
    static constexpr size_t PAD_SIZE = (REMAINDER == 0) ? 1 : (CACHE_LINE_SIZE - REMAINDER);
    char padding[PAD_SIZE];

    CacheLineIsolated() = default;
    explicit CacheLineIsolated(const T& v) : value(v) {}
    explicit CacheLineIsolated(T&& v) : value(std::move(v)) {}

    operator T&() { return value; }
    operator const T&() const { return value; }
};

struct AlignedDeleter {
    void operator()(void* ptr) const noexcept {
        if (ptr) std::free(ptr);
    }
};

template <typename T>
using AlignedUniquePtr = std::unique_ptr<T, AlignedDeleter>;

template <typename T, size_t Alignment = SIMD_ALIGNMENT>
AlignedUniquePtr<T> make_aligned(size_t count = 1) {
    static_assert((Alignment & (Alignment - 1)) == 0, "Alignment must be power of 2");
    size_t bytes = count * sizeof(T);
    bytes = ((bytes + Alignment - 1) / Alignment) * Alignment;
    void* ptr = std::aligned_alloc(Alignment, bytes);
    if (!ptr) throw std::bad_alloc();
    return AlignedUniquePtr<T>(static_cast<T*>(ptr));
}

template <size_t D>
inline float l2_distance_simd(const float* a, const float* b) {
    static_assert(D % 8 == 0, "D must be a multiple of 8 for AVX2");
    __m256 sum = _mm256_setzero_ps();
    for (size_t i = 0; i < D; i += 8) {
        __m256 diff = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

inline float l2_distance_simd_runtime(const float* a, const float* b, size_t dim) {
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 diff = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    float result = _mm_cvtss_f32(s);
    for (; i < dim; ++i) {
        float d = a[i] - b[i];
        result += d * d;
    }
    return result;
}

#ifdef __AVX512F__
template <size_t D>
inline float l2_distance_avx512(const float* a, const float* b) {
    static_assert(D % 16 == 0, "D must be a multiple of 16 for AVX-512");
    __m512 sum = _mm512_setzero_ps();
    for (size_t i = 0; i < D; i += 16) {
        __m512 diff = _mm512_sub_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i));
        sum = _mm512_fmadd_ps(diff, diff, sum);
    }
    return _mm512_reduce_add_ps(sum);
}
#endif

template <size_t D>
inline float l2_distance_best(const float* a, const float* b) {
#ifdef __AVX512F__
    if constexpr (D % 16 == 0) {
        static const bool has_avx512 = __builtin_cpu_supports("avx512f");
        if (has_avx512) {
            return l2_distance_avx512<D>(a, b);
        }
    }
#endif
    return l2_distance_simd<D>(a, b);
}

}  // namespace cphnsw
