#pragma once

#include "../core/codes.hpp"
#include "fastscan_layout.hpp"
#include <cstdint>
#include <cmath>
#include <limits>
#include <algorithm>
#include <immintrin.h>

namespace cphnsw {

namespace fastscan {

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

        // Flush uint8 accumulator to uint16 every 2 pairs (max 4*60=240 < 255)
        if (pairs_since_flush >= 2 || sp == NUM_SUB_PAIRS - 1) {
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
inline void convert_to_distances(
    const RaBitQQuery<D>& query,
    const uint32_t* fastscan_sums,
    const VertexAuxData* aux,
    const uint16_t* popcounts,
    size_t count,
    float* out,
    float /*parent_norm*/,
    float /*dist_qp_sq*/)
{
    float A = query.coeff_fastscan;
    float B = query.coeff_popcount;
    float C = query.coeff_constant;
    float query_norm = query.query_norm;
    float query_norm_sq = query.query_norm_sq;

    for (size_t i = 0; i < count; ++i) {
        float ip_approx = A * static_cast<float>(fastscan_sums[i])
                        + B * static_cast<float>(popcounts[i])
                        + C;

        float ip_qo = aux[i].ip_quantized_original;

        // Standard RaBitQ distance estimate (Eq. 19 of SIGMOD'24 paper):
        //   ||q - o||^2 = ||o-c||^2 + ||q-c||^2 - 2*||o-c||*||q-c||*<unit(q-c), unit(o-c)>
        // where <unit(q-c), unit(o-c)> ≈ ip_approx / ip_qo
        float ip_est = (ip_qo > 1e-10f) ? ip_approx / ip_qo : 0.0f;

        float dist_o = aux[i].dist_to_centroid;
        out[i] = dist_o * dist_o + query_norm_sq
               - 2.0f * dist_o * query_norm * ip_est;
    }
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
    float /*parent_norm*/,
    float /*dist_qp_sq*/)
{
    float A = query.coeff_fastscan;
    float B = query.coeff_popcount;
    float C = query.coeff_constant;
    float query_norm = query.query_norm;
    float query_norm_sq = query.query_norm_sq;
    float epsilon = query.error_epsilon;

    for (size_t i = 0; i < count; ++i) {
        float ip_approx = A * static_cast<float>(fastscan_sums[i])
                        + B * static_cast<float>(popcounts[i])
                        + C;

        float ip_qo = aux[i].ip_quantized_original;

        // Standard RaBitQ distance estimate
        float ip_est = (ip_qo > 1e-10f) ? ip_approx / ip_qo : 0.0f;

        float dist_o = aux[i].dist_to_centroid;

        out_dist[i] = dist_o * dist_o + query_norm_sq
                    - 2.0f * dist_o * query_norm * ip_est;

        // Error bound on ip_approx (Theorem 3.2)
        float ip_qo_sq = ip_qo * ip_qo;
        if (ip_qo_sq < 1e-10f) {
            out_lower[i] = std::numeric_limits<float>::max();
            continue;
        }
        float bound_on_ip = epsilon * std::sqrt(
            (1.0f - ip_qo_sq) / (ip_qo_sq * static_cast<float>(D)));

        // Upper bound on cosine → lower bound on distance
        // Clamp cosine to 1.0 (mathematical maximum) to ensure
        // lower >= (dist_o - query_norm)^2 (triangle inequality)
        float ip_est_upper = std::min(ip_est + bound_on_ip, 1.0f);

        out_dist[i] = std::max(out_dist[i], 0.0f);
        out_lower[i] = dist_o * dist_o + query_norm_sq
                      - 2.0f * dist_o * query_norm * ip_est_upper;
        if (out_lower[i] < 0.0f) out_lower[i] = 0.0f;
    }
}

// === Multi-bit FastScan (Extended RaBitQ) ===
// Weighted sum of per-plane inner products. Also outputs MSB-only sums
// for coarse lower bounds.

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

// Multi-bit distance conversion with bounds.
// Uses full B-bit weighted sums for distance estimates, MSB (1-bit) sums
// for conservative lower bounds.

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
    float /*parent_norm*/,
    float /*dist_qp_sq*/)
{
    constexpr float K = static_cast<float>((1u << BitWidth) - 1);
    constexpr float inv_K = 1.0f / K;

    float A_nbit = query.coeff_fastscan * inv_K;
    float B_nbit = query.coeff_popcount * inv_K;
    float C = query.coeff_constant;
    float query_norm = query.query_norm;
    float query_norm_sq = query.query_norm_sq;
    float epsilon = query.error_epsilon;

    float A_msb = query.coeff_fastscan;
    float B_msb = query.coeff_popcount;

    for (size_t i = 0; i < count; ++i) {
        // Full B-bit distance estimate (standard RaBitQ formula)
        float ip_approx = A_nbit * static_cast<float>(nbit_fastscan_sums[i])
                        + B_nbit * static_cast<float>(weighted_popcounts[i])
                        + C;

        float ip_qo = aux[i].ip_quantized_original;
        float ip_est = (ip_qo > 1e-10f) ? ip_approx / ip_qo : 0.0f;

        float dist_o = aux[i].dist_to_centroid;
        out_dist[i] = dist_o * dist_o + query_norm_sq
                    - 2.0f * dist_o * query_norm * ip_est;

        // Lower bound using MSB (1-bit) estimate + error bound
        float ip_approx_msb = A_msb * static_cast<float>(msb_fastscan_sums[i])
                            + B_msb * static_cast<float>(msb_popcounts[i])
                            + C;

        float ip_qo_sq = ip_qo * ip_qo;
        if (ip_qo_sq < 1e-10f) {
            out_lower[i] = std::numeric_limits<float>::max();
            continue;
        }
        float bound_on_ip = epsilon * std::sqrt(
            (1.0f - ip_qo_sq) / (ip_qo_sq * static_cast<float>(D)));

        float ip_est_msb = ip_approx_msb / ip_qo;
        float ip_est_upper = std::min(ip_est_msb + bound_on_ip, 1.0f);

        out_dist[i] = std::max(out_dist[i], 0.0f);
        out_lower[i] = dist_o * dist_o + query_norm_sq
                     - 2.0f * dist_o * query_norm * ip_est_upper;
        if (out_lower[i] < 0.0f) out_lower[i] = 0.0f;
    }
}

} // namespace fastscan

template <size_t D>
struct RaBitQMetricPolicy {
    using CodeType = RaBitQCode<D>;
    using QueryType = RaBitQQuery<D>;

    static constexpr size_t DIMS = D;

    // Entry-point distance estimation using global-centroid RaBitQ formula.
    //
    // This method uses per-vector metadata (code.dist_to_centroid = ||o_r - centroid||,
    // code.ip_quantized_original = <o_bar, o>) which are relative to the dataset centroid.
    // It does NOT apply the SymphonyQG parent-relative correction because at the search
    // entry point there is no parent vertex.
    //
    // For neighbor distance estimation during graph traversal, use the SIMD batch path
    // (fastscan::convert_to_distances / convert_to_distances_with_bounds) which applies
    // the SymphonyQG correction via per-edge VertexAuxData.
    static inline float compute_distance(
        const QueryType& query, const CodeType& code)
    {
        constexpr size_t NUM_SUB_SEGMENTS = (D + 3) / 4;
        uint32_t fastscan_sum = 0;

        for (size_t j = 0; j < NUM_SUB_SEGMENTS; ++j) {
            size_t bit_base = j * 4;
            uint8_t pattern = 0;
            for (size_t b = 0; b < 4 && (bit_base + b) < D; ++b) {
                if (code.signs.get_bit(bit_base + b)) {
                    pattern |= (1 << b);
                }
            }
            fastscan_sum += query.lut[j][pattern];
        }

        float ip_approx = query.coeff_fastscan * static_cast<float>(fastscan_sum)
                        + query.coeff_popcount * static_cast<float>(code.code_popcount)
                        + query.coeff_constant;

        float ip_est = (code.ip_quantized_original > 1e-10f)
                     ? ip_approx / code.ip_quantized_original
                     : 0.0f;

        float dist_o = code.dist_to_centroid;
        return dist_o * dist_o + query.query_norm_sq
             - 2.0f * dist_o * query.query_norm * ip_est;
    }

    // Scalar distance estimation using per-edge VertexAuxData.
    // Uses the same standard RaBitQ formula as the SIMD batch path.
    static inline float compute_distance_with_aux(
        const QueryType& query, const CodeType& code,
        const VertexAuxData& aux,
        float /*parent_norm*/, float /*dist_qp_sq*/)
    {
        constexpr size_t NUM_SUB_SEGMENTS = (D + 3) / 4;
        uint32_t fastscan_sum = 0;

        for (size_t j = 0; j < NUM_SUB_SEGMENTS; ++j) {
            size_t bit_base = j * 4;
            uint8_t pattern = 0;
            for (size_t b = 0; b < 4 && (bit_base + b) < D; ++b) {
                if (code.signs.get_bit(bit_base + b)) {
                    pattern |= (1 << b);
                }
            }
            fastscan_sum += query.lut[j][pattern];
        }

        float ip_approx = query.coeff_fastscan * static_cast<float>(fastscan_sum)
                        + query.coeff_popcount * static_cast<float>(code.code_popcount)
                        + query.coeff_constant;

        float ip_qo = aux.ip_quantized_original;
        float ip_est = (ip_qo > 1e-10f) ? ip_approx / ip_qo : 0.0f;

        float dist_o = aux.dist_to_centroid;
        return dist_o * dist_o + query.query_norm_sq
             - 2.0f * dist_o * query.query_norm * ip_est;
    }

};

using RaBitQPolicy128 = RaBitQMetricPolicy<128>;
using RaBitQPolicy256 = RaBitQMetricPolicy<256>;
using RaBitQPolicy512 = RaBitQMetricPolicy<512>;
using RaBitQPolicy1024 = RaBitQMetricPolicy<1024>;

}  // namespace cphnsw