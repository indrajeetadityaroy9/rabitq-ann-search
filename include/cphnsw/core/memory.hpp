#pragma once

#include <cstddef>
#include <cstdlib>
#include <new>
#include <vector>
#include <immintrin.h>

namespace cphnsw {

constexpr size_t CACHE_LINE_SIZE = 64;
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

template <int Locality = 3>
inline void prefetch_t(const void* addr) {
    __builtin_prefetch(addr, 0, Locality);
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

}  // namespace cphnsw
