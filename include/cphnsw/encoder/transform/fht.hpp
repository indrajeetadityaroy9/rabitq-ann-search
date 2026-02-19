#pragma once

#include <cstddef>

#include <immintrin.h>

namespace cphnsw {

// Unnormalized Walsh-Hadamard Transform (WHT) butterfly.
//
// This transform is NOT unitary: a single call on input of length `len`
// scales the L2-norm by sqrt(len), because for the unnormalized WHT
// ||WHT(x)||_2 = sqrt(len) * ||x||_2 (Parseval identity).
//
// Normalization is intentionally deferred: RaBitQEncoderBase applies three
// consecutive passes via RandomHadamardRotation::apply(), accumulating a
// total scale factor of sqrt(D)^3 = D * sqrt(D). The combined normalization
// norm_factor_ = 1 / (D * sqrt(D)) is applied once after all three passes,
// avoiding three separate per-pass division loops.
//
// Do NOT add per-call normalization here without updating norm_factor_ in
// rabitq_encoder.hpp.
inline void fht(float* vec, size_t len) {
    constexpr size_t SIMD_WIDTH = 8;

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

}
