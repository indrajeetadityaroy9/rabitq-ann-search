#pragma once

#include <cstddef>
#include <cassert>
#include <immintrin.h>

namespace cphnsw {

namespace detail {
inline bool is_power_of_two(size_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}
}  // namespace detail

inline void fht_scalar(float* vec, size_t len) {
    assert(detail::is_power_of_two(len));
    
    for (size_t h = 1; h < len; h *= 2) {
        for (size_t i = 0; i < len; i += h * 2) {
            for (size_t j = i; j < i + h; ++j) {
                float x = vec[j];
                float y = vec[j + h];
                vec[j] = x + y;
                vec[j + h] = x - y;
            }
        }
    }
}

inline void fht_avx2(float* vec, size_t len) {
    assert(detail::is_power_of_two(len));

    constexpr size_t SIMD_WIDTH = 8;
    
    if (len < SIMD_WIDTH) {
        fht_scalar(vec, len);
        return;
    }

    for (size_t i = 0; i < len; i += SIMD_WIDTH) {
        __m256 v = _mm256_loadu_ps(&vec[i]);

        __m256 v_swap1 = _mm256_permute_ps(v, 0b10110001);
        __m256 v_add1 = _mm256_add_ps(v, v_swap1);
        __m256 v_sub1 = _mm256_sub_ps(v, v_swap1);
        v = _mm256_blend_ps(v_add1, v_sub1, 0b10101010);

        __m256 v_swap2 = _mm256_permute_ps(v, 0b01001110);
        __m256 v_add2 = _mm256_add_ps(v, v_swap2);
        __m256 v_sub2 = _mm256_sub_ps(v, v_swap2);
        v = _mm256_blend_ps(v_add2, v_sub2, 0b11001100);

        __m256 v_swap4 = _mm256_permute2f128_ps(v, v, 0x01);
        __m256 v_add4 = _mm256_add_ps(v, v_swap4);
        __m256 v_sub4 = _mm256_sub_ps(v, v_swap4);
        v = _mm256_blend_ps(v_add4, v_sub4, 0b11110000);

        _mm256_storeu_ps(&vec[i], v);
    }

    for (size_t h = SIMD_WIDTH; h < len; h *= 2) {
        for (size_t i = 0; i < len; i += h * 2) {
            for (size_t j = i; j < i + h; j += SIMD_WIDTH) {
                __m256 x = _mm256_loadu_ps(&vec[j]);
                __m256 y = _mm256_loadu_ps(&vec[j + h]);
                _mm256_storeu_ps(&vec[j], _mm256_add_ps(x, y));
                _mm256_storeu_ps(&vec[j + h], _mm256_sub_ps(x, y));
            }
        }
    }
}

#ifdef __AVX512F__
inline void fht_avx512(float* vec, size_t len) {
    assert(detail::is_power_of_two(len));

    if (len < 16) {
        fht_avx2(vec, len);
        return;
    }

    constexpr size_t SIMD_WIDTH = 16;

    for (size_t i = 0; i < len; i += SIMD_WIDTH) {
        __m512 v = _mm512_loadu_ps(&vec[i]);

        __m512i idx1 = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
        __m512 v_swap1 = _mm512_permutexvar_ps(idx1, v);
        __m512 v_add1 = _mm512_add_ps(v, v_swap1);
        __m512 v_sub1 = _mm512_sub_ps(v, v_swap1);
        v = _mm512_mask_blend_ps(0xAAAA, v_add1, v_sub1);

        __m512i idx2 = _mm512_set_epi32(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2);
        __m512 v_swap2 = _mm512_permutexvar_ps(idx2, v);
        __m512 v_add2 = _mm512_add_ps(v, v_swap2);
        __m512 v_sub2 = _mm512_sub_ps(v, v_swap2);
        v = _mm512_mask_blend_ps(0xCCCC, v_add2, v_sub2);

        __m512i idx4 = _mm512_set_epi32(11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4);
        __m512 v_swap4 = _mm512_permutexvar_ps(idx4, v);
        __m512 v_add4 = _mm512_add_ps(v, v_swap4);
        __m512 v_sub4 = _mm512_sub_ps(v, v_swap4);
        v = _mm512_mask_blend_ps(0xF0F0, v_add4, v_sub4);

        __m512 v_swap8 = _mm512_shuffle_f32x4(v, v, 0x4E);
        __m512 v_add8 = _mm512_add_ps(v, v_swap8);
        __m512 v_sub8 = _mm512_sub_ps(v, v_swap8);
        v = _mm512_mask_blend_ps(0xFF00, v_add8, v_sub8);

        _mm512_storeu_ps(&vec[i], v);
    }

    for (size_t h = SIMD_WIDTH; h < len; h *= 2) {
        for (size_t i = 0; i < len; i += h * 2) {
            for (size_t j = i; j < i + h; j += SIMD_WIDTH) {
                __m512 x = _mm512_loadu_ps(&vec[j]);
                __m512 y = _mm512_loadu_ps(&vec[j + h]);
                _mm512_storeu_ps(&vec[j], _mm512_add_ps(x, y));
                _mm512_storeu_ps(&vec[j + h], _mm512_sub_ps(x, y));
            }
        }
    }
}
#endif

inline void fht(float* vec, size_t len) {
    assert(detail::is_power_of_two(len));

#ifdef __AVX512F__
    if (len >= 16) {
        fht_avx512(vec, len);
        return;
    }
#endif
    fht_avx2(vec, len);
}

}  // namespace cphnsw