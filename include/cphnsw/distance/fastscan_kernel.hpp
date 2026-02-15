#pragma once

#include "../core/codes.hpp"
#include "../core/adaptive_defaults.hpp"
#include "fastscan_layout.hpp"
#include <cstdint>
#include <cmath>
#include <limits>
#include <algorithm>
#include <immintrin.h>

namespace cphnsw {

namespace fastscan {

// Each sub-pair contributes up to 2 LUT lookups (lo + hi nibble), max LUT value = 15.
// After N pairs: accumulated max per byte = N * 2 * 15 = 30N.
// uint8 overflows at 255, so theoretical max N = 8.
// Flush at N=2 (max accumulation = 60) for conservative safety margin.
constexpr size_t FLUSH_INTERVAL = 2;

template <size_t D>
inline void compute_inner_products(
    const uint8_t lut[][16],
    const FastScanCodeBlock<D, 32>& block,
    uint32_t* out)
{
    constexpr size_t NUM_SUB_PAIRS = FastScanCodeBlock<D, 32>::NUM_SUB_PAIRS;
    constexpr size_t NUM_SUB_SEGMENTS = FastScanCodeBlock<D, 32>::NUM_SUB_SEGMENTS;

    const __m256i low_mask = _mm256_set1_epi8(0x0F);

    __m256i acc_lo = _mm256_setzero_si256();
    __m256i acc_hi = _mm256_setzero_si256();

    __m256i tmp_acc = _mm256_setzero_si256();
    size_t pairs_since_flush = 0;

    for (size_t sp = 0; sp < NUM_SUB_PAIRS; ++sp) {
        size_t seg_lo = 2 * sp;
        size_t seg_hi = 2 * sp + 1;

        __m256i codes = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(block.packed[sp]));

        __m256i lo_nibbles = _mm256_and_si256(codes, low_mask);
        __m256i hi_nibbles = _mm256_and_si256(_mm256_srli_epi16(codes, 4), low_mask);

        __m128i lut_lo_128 = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(lut[seg_lo]));
        __m256i lut_lo = _mm256_broadcastsi128_si256(lut_lo_128);

        __m256i result_lo = _mm256_shuffle_epi8(lut_lo, lo_nibbles);
        tmp_acc = _mm256_add_epi8(tmp_acc, result_lo);

        if (seg_hi < NUM_SUB_SEGMENTS) {
            __m128i lut_hi_128 = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(lut[seg_hi]));
            __m256i lut_hi = _mm256_broadcastsi128_si256(lut_hi_128);

            __m256i result_hi = _mm256_shuffle_epi8(lut_hi, hi_nibbles);
            tmp_acc = _mm256_add_epi8(tmp_acc, result_hi);
        }

        pairs_since_flush++;

        if (pairs_since_flush >= FLUSH_INTERVAL || sp == NUM_SUB_PAIRS - 1) {
            __m256i widened_lo = _mm256_cvtepu8_epi16(
                _mm256_castsi256_si128(tmp_acc));
            acc_lo = _mm256_add_epi16(acc_lo, widened_lo);

            __m256i widened_hi = _mm256_cvtepu8_epi16(
                _mm256_extracti128_si256(tmp_acc, 1));
            acc_hi = _mm256_add_epi16(acc_hi, widened_hi);

            tmp_acc = _mm256_setzero_si256();
            pairs_since_flush = 0;
        }
    }

    __m256i r0 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(acc_lo));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out), r0);

    __m256i r1 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(acc_lo, 1));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + 8), r1);

    __m256i r2 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(acc_hi));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + 16), r2);

    __m256i r3 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(acc_hi, 1));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + 24), r3);
}

template <size_t D>
inline void convert_to_distances_with_bounds(
    const RaBitQQuery<D>& query,
    const uint32_t* fastscan_sums,
    const VertexAuxData* aux,
    const uint16_t* popcounts,
    size_t count,
    float* out_dist,
    float* out_lower,
    float dist_qp_sq)
{
    float A = query.coeff_fastscan;
    float B = query.coeff_popcount;
    float C = query.coeff_constant;
    float epsilon = query.error_epsilon;

    float sqrt_dqp = std::sqrt(dist_qp_sq);

    for (size_t i = 0; i < count; ++i) {
        float ip_approx = A * static_cast<float>(fastscan_sums[i])
                        + B * static_cast<float>(popcounts[i])
                        + C;

        float ip_qo_p = aux[i].ip_quantized_original;

        float ip_corrected = ip_approx - aux[i].ip_code_parent;
        float ip_est = (ip_qo_p > adaptive_defaults::ip_quality_epsilon()) ? ip_corrected / ip_qo_p : 0.0f;

        float nop = aux[i].dist_to_centroid;

        out_dist[i] = nop * nop + dist_qp_sq - 2.0f * nop * ip_est;

        float ip_qo_p_sq = ip_qo_p * ip_qo_p;
        if (ip_qo_p_sq < adaptive_defaults::ip_quality_epsilon()) {
            out_dist[i] = std::max(out_dist[i], 0.0f);
            out_lower[i] = std::max(out_dist[i], 0.0f);
            continue;
        }
        float Df = static_cast<float>(D);
        float bound_on_cos = epsilon * std::sqrt(
            (1.0f - ip_qo_p_sq) * (Df - 1.0f) / (ip_qo_p_sq * Df));

        float cos_upper = (sqrt_dqp > adaptive_defaults::ip_quality_epsilon())
            ? std::min(ip_est / sqrt_dqp + bound_on_cos, 1.0f)
            : 1.0f;

        out_dist[i] = std::max(out_dist[i], 0.0f);
        out_lower[i] = nop * nop + dist_qp_sq
                      - 2.0f * nop * sqrt_dqp * cos_upper;
        if (out_lower[i] < 0.0f) out_lower[i] = 0.0f;
    }
}


template <size_t D, size_t BitWidth>
inline void compute_nbit_inner_products(
    const uint8_t lut[][16],
    const NbitFastScanCodeBlock<D, BitWidth, 32>& block,
    uint32_t* out_nbit,
    uint32_t* out_msb)
{
    std::memset(out_nbit, 0, 32 * sizeof(uint32_t));
    alignas(64) uint32_t plane_sums[32];

    for (size_t b = 0; b < BitWidth; ++b) {
        compute_inner_products<D>(lut, block.planes[b], plane_sums);
        if (b == 0) {
            std::memcpy(out_msb, plane_sums, 32 * sizeof(uint32_t));
        }
        uint32_t weight = 1u << (BitWidth - 1 - b);
        for (size_t i = 0; i < 32; ++i) {
            out_nbit[i] += weight * plane_sums[i];
        }
    }
}


template <size_t D, size_t BitWidth>
inline void convert_nbit_to_distances_with_bounds(
    const RaBitQQuery<D>& query,
    const uint32_t* nbit_fastscan_sums,
    const uint32_t* msb_fastscan_sums,
    const VertexAuxData* aux,
    const uint16_t* msb_popcounts,
    const uint16_t* weighted_popcounts,
    size_t count,
    float* out_dist,
    float* out_lower,
    float dist_qp_sq)
{
    constexpr float K = static_cast<float>((1u << BitWidth) - 1);
    constexpr float inv_K = 1.0f / K;

    float A_nbit = query.coeff_fastscan * inv_K;
    float B_nbit = query.coeff_popcount * inv_K;
    float C = query.coeff_constant;
    float epsilon = query.error_epsilon;

    float A_msb = query.coeff_fastscan;
    float B_msb = query.coeff_popcount;

    float sqrt_dqp = std::sqrt(dist_qp_sq);

    for (size_t i = 0; i < count; ++i) {
        float ip_approx_nbit = A_nbit * static_cast<float>(nbit_fastscan_sums[i])
                             + B_nbit * static_cast<float>(weighted_popcounts[i])
                             + C;

        float ip_qo_p = aux[i].ip_quantized_original;

        float ip_corrected_nbit = ip_approx_nbit - aux[i].ip_code_parent;
        float ip_est_nbit = (ip_qo_p > adaptive_defaults::ip_quality_epsilon()) ? ip_corrected_nbit / ip_qo_p : 0.0f;

        float nop = aux[i].dist_to_centroid;

        out_dist[i] = nop * nop + dist_qp_sq
                    - 2.0f * nop * ip_est_nbit;

        float ip_approx_msb = A_msb * static_cast<float>(msb_fastscan_sums[i])
                            + B_msb * static_cast<float>(msb_popcounts[i])
                            + C;

        float ip_qo_p_sq = ip_qo_p * ip_qo_p;
        if (ip_qo_p_sq < adaptive_defaults::ip_quality_epsilon()) {
            out_dist[i] = std::max(out_dist[i], 0.0f);
            out_lower[i] = std::max(out_dist[i], 0.0f);
            continue;
        }
        float Df = static_cast<float>(D);
        float bound_on_cos = epsilon * std::sqrt(
            (1.0f - ip_qo_p_sq) * (Df - 1.0f) / (ip_qo_p_sq * Df));

        float ip_corrected_msb = ip_approx_msb - aux[i].ip_code_parent;
        float ip_est_msb = ip_corrected_msb / ip_qo_p;
        float cos_upper = (sqrt_dqp > adaptive_defaults::ip_quality_epsilon())
            ? std::min(ip_est_msb / sqrt_dqp + bound_on_cos, 1.0f)
            : 1.0f;

        out_dist[i] = std::max(out_dist[i], 0.0f);
        out_lower[i] = nop * nop + dist_qp_sq
                      - 2.0f * nop * sqrt_dqp * cos_upper;
        if (out_lower[i] < 0.0f) out_lower[i] = 0.0f;
    }
}

} // namespace fastscan

}  // namespace cphnsw
