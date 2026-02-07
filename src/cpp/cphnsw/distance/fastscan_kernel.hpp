#pragma once

#include "../core/codes.hpp"
#include "fastscan_layout.hpp"
#include <cstdint>
#include <cmath>
#include <immintrin.h>

namespace cphnsw {

// ============================================================================
// FastScan Kernel: vpshufb-based SIMD distance computation
// ============================================================================

/**
 * FastScan computes <x_bar_b, q_bar_u> for 32 codes simultaneously using
 * the vpshufb (PSHUFB) instruction as a 16-entry lookup table.
 *
 * For each sub-segment pair:
 *   1. Load 32 nibble-packed bytes from the code block
 *   2. Split into lo/hi nibbles (4-bit sub-segment indices)
 *   3. vpshufb with LUTs → 32 parallel lookups per sub-segment
 *   4. Accumulate into running sum (uint16 to prevent overflow)
 *
 * Result: 32 values of <x_bar_b, q_bar_u>, one per code in the batch.
 * These are then converted to distance estimates via linear coefficients.
 *
 * OVERFLOW PREVENTION: Each LUT entry is 0-60 (4 dims * 15 max each).
 * After 4 sub-segment pairs (accumulating 8 sub-segments), max value is
 * 8*60 = 480 which exceeds uint8 (255). We widen to uint16 every 2 pairs
 * (4 sub-segments, max 240 < 255).
 */

namespace fastscan {

// ============================================================================
// Scalar Fallback
// ============================================================================

/**
 * Scalar FastScan: compute <x_bar_b, q_bar_u> for a batch of codes.
 */
template <size_t D, size_t BatchSize>
inline void compute_inner_products_scalar(
    const uint8_t lut[][16],
    const FastScanCodeBlock<D, BatchSize>& block,
    size_t count,
    uint32_t* out)
{
    constexpr size_t NUM_SUB_PAIRS = FastScanCodeBlock<D, BatchSize>::NUM_SUB_PAIRS;
    constexpr size_t NUM_SUB_SEGMENTS = FastScanCodeBlock<D, BatchSize>::NUM_SUB_SEGMENTS;

    for (size_t c = 0; c < count; ++c) {
        uint32_t sum = 0;
        for (size_t sp = 0; sp < NUM_SUB_PAIRS; ++sp) {
            uint8_t byte = block.packed[sp][c];
            uint8_t lo = byte & 0x0F;
            uint8_t hi = (byte >> 4) & 0x0F;

            size_t seg_lo = 2 * sp;
            size_t seg_hi = 2 * sp + 1;

            sum += lut[seg_lo][lo];
            if (seg_hi < NUM_SUB_SEGMENTS) {
                sum += lut[seg_hi][hi];
            }
        }
        out[c] = sum;
    }
}

// ============================================================================
// AVX2 Kernel (32 codes per batch)
// ============================================================================

#if defined(__AVX2__)

/**
 * AVX2 FastScan: compute <x_bar_b, q_bar_u> for 32 codes simultaneously.
 *
 * Uses vpshufb (_mm256_shuffle_epi8) as a 16-entry LUT applied in parallel.
 * Each vpshufb processes one 128-bit lane independently (two identical LUTs).
 *
 * @param lut The LUT array: lut[sub_segment][0..15]
 * @param block The nibble-interleaved code block (32 codes)
 * @param out Output: 32 uint32 values of <x_bar_b, q_bar_u>
 */
template <size_t D>
inline void compute_inner_products_avx2(
    const uint8_t lut[][16],
    const FastScanCodeBlock<D, 32>& block,
    uint32_t* out)
{
    constexpr size_t NUM_SUB_PAIRS = FastScanCodeBlock<D, 32>::NUM_SUB_PAIRS;
    constexpr size_t NUM_SUB_SEGMENTS = FastScanCodeBlock<D, 32>::NUM_SUB_SEGMENTS;

    const __m256i low_mask = _mm256_set1_epi8(0x0F);

    // Accumulator in uint16 (to prevent uint8 overflow)
    __m256i acc_lo = _mm256_setzero_si256();  // codes 0-15 (low 16 bytes each lane)
    __m256i acc_hi = _mm256_setzero_si256();  // codes 16-31

    // Temporary uint8 accumulator (flushed to uint16 periodically)
    __m256i tmp_acc = _mm256_setzero_si256();
    size_t pairs_since_flush = 0;

    for (size_t sp = 0; sp < NUM_SUB_PAIRS; ++sp) {
        size_t seg_lo = 2 * sp;
        size_t seg_hi = 2 * sp + 1;

        // Load 32 nibble-packed bytes
        __m256i codes = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(block.packed[sp]));

        // Extract lo nibbles (sub-segment 2*sp)
        __m256i lo_nibbles = _mm256_and_si256(codes, low_mask);
        // Extract hi nibbles (sub-segment 2*sp+1)
        __m256i hi_nibbles = _mm256_and_si256(_mm256_srli_epi16(codes, 4), low_mask);

        // Load LUT for lo sub-segment: broadcast 16 bytes to both 128-bit lanes
        __m128i lut_lo_128 = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(lut[seg_lo]));
        __m256i lut_lo = _mm256_broadcastsi128_si256(lut_lo_128);

        // Lookup: vpshufb does parallel 16-entry table lookup
        __m256i result_lo = _mm256_shuffle_epi8(lut_lo, lo_nibbles);
        tmp_acc = _mm256_add_epi8(tmp_acc, result_lo);

        // Hi sub-segment (if valid)
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
            // Widen lower 16 bytes to uint16
            __m256i widened_lo = _mm256_cvtepu8_epi16(
                _mm256_castsi256_si128(tmp_acc));
            acc_lo = _mm256_add_epi16(acc_lo, widened_lo);

            // Widen upper 16 bytes to uint16
            __m256i widened_hi = _mm256_cvtepu8_epi16(
                _mm256_extracti128_si256(tmp_acc, 1));
            acc_hi = _mm256_add_epi16(acc_hi, widened_hi);

            tmp_acc = _mm256_setzero_si256();
            pairs_since_flush = 0;
        }
    }

    // Extract results: widen uint16 to uint32 and store
    // acc_lo has 16 uint16 values for codes 0-15
    // acc_hi has 16 uint16 values for codes 16-31

    // Widen acc_lo: lower 8 uint16 → 8 uint32
    __m256i r0 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(acc_lo));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out), r0);

    // Upper 8 uint16 → 8 uint32
    __m256i r1 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(acc_lo, 1));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + 8), r1);

    // Same for acc_hi (codes 16-31)
    __m256i r2 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(acc_hi));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + 16), r2);

    __m256i r3 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(acc_hi, 1));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + 24), r3);
}

#endif // __AVX2__

// ============================================================================
// Distance Conversion: FastScan sum → estimated L2 distance
// ============================================================================

/**
 * Convert FastScan inner product sums to L2 distance estimates.
 *
 * For each code i:
 *   <x_bar, q_bar>_i = A * fastscan_sum[i] + B * popcount[i] + C
 *   <o, q>_est_i = <x_bar, q_bar>_i / <o_bar, o>_i
 *   dist_i = ||o_r-c||^2 + ||q_r-c||^2 - 2*||o_r-c||*||q_r-c||*<o,q>_est_i
 *
 * @param query RaBitQ query with precomputed coefficients
 * @param fastscan_sums Array of <x_bar_b, q_bar_u> values (from SIMD kernel)
 * @param aux Array of per-neighbor auxiliary data
 * @param popcounts Array of per-neighbor popcounts
 * @param count Number of neighbors to process
 * @param out Output distance estimates
 */
template <size_t D>
inline void convert_to_distances(
    const RaBitQQuery<D>& query,
    const uint32_t* fastscan_sums,
    const VertexAuxData* aux,
    const uint16_t* popcounts,
    size_t count,
    float* out)
{
    float A = query.coeff_fastscan;
    float B = query.coeff_popcount;
    float C = query.coeff_constant;
    float query_norm = query.query_norm;
    float query_norm_sq = query.query_norm_sq;

    for (size_t i = 0; i < count; ++i) {
        // <x_bar, q_bar> from linear coefficients
        float ip_approx = A * static_cast<float>(fastscan_sums[i])
                        + B * static_cast<float>(popcounts[i])
                        + C;

        // Unbiased estimator: <o, q>_est = <x_bar, q_bar> / <o_bar, o>
        float ip_qo = aux[i].ip_quantized_original;
        float ip_est = (ip_qo > 1e-10f) ? ip_approx / ip_qo : 0.0f;

        // L2 distance estimate
        float dist_o = aux[i].dist_to_centroid;
        out[i] = dist_o * dist_o + query_norm_sq
               - 2.0f * dist_o * query_norm * ip_est;
    }
}

/**
 * Convert FastScan sums to distances with error bounds (for pruning).
 *
 * @param out_dist Output distance estimates
 * @param out_lower Output lower bounds on distance
 */
template <size_t D>
inline void convert_to_distances_with_bounds(
    const RaBitQQuery<D>& query,
    const uint32_t* fastscan_sums,
    const VertexAuxData* aux,
    const uint16_t* popcounts,
    size_t count,
    float* out_dist,
    float* out_lower)
{
    float A = query.coeff_fastscan;
    float B = query.coeff_popcount;
    float C = query.coeff_constant;
    float query_norm = query.query_norm;
    float query_norm_sq = query.query_norm_sq;
    float epsilon = query.error_epsilon;
    float inv_sqrt_d = query.inv_sqrt_d;

    for (size_t i = 0; i < count; ++i) {
        float ip_approx = A * static_cast<float>(fastscan_sums[i])
                        + B * static_cast<float>(popcounts[i])
                        + C;

        float ip_qo = aux[i].ip_quantized_original;
        float ip_est = (ip_qo > 1e-10f) ? ip_approx / ip_qo : 0.0f;

        float dist_o = aux[i].dist_to_centroid;

        // Central distance estimate
        out_dist[i] = dist_o * dist_o + query_norm_sq
                    - 2.0f * dist_o * query_norm * ip_est;

        // Error bound on <o, q>
        float ip_qo_sq = ip_qo * ip_qo;
        float bound = epsilon * std::sqrt(
            (1.0f - ip_qo_sq) / (ip_qo_sq * static_cast<float>(D)));

        // Upper bound on <o, q> → lower bound on distance
        float ip_upper = ip_est + bound;
        out_lower[i] = dist_o * dist_o + query_norm_sq
                      - 2.0f * dist_o * query_norm * ip_upper;
    }
}

} // namespace fastscan

// ============================================================================
// RaBitQMetricPolicy: Policy for use with SearchEngine
// ============================================================================

/**
 * RaBitQMetricPolicy: Distance computation policy using RaBitQ quantization.
 *
 * Provides the same interface as UnifiedMetricPolicy but uses:
 *   - Asymmetric distance estimation (float query vs. binary DB)
 *   - FastScan vpshufb SIMD for batch computation
 *   - Unbiased estimator with provable error bounds
 *
 * @tparam D Padded dimension
 */
template <size_t D>
struct RaBitQMetricPolicy {
    using CodeType = RaBitQCode<D>;
    using QueryType = RaBitQQuery<D>;

    static constexpr size_t DIMS = D;

    /**
     * Compute distance for a single code (scalar path).
     */
    static inline float compute_distance(
        const QueryType& query, const CodeType& code)
    {
        // Scalar LUT evaluation
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

        // Linear coefficients → distance
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

    /**
     * Compute distance for a batch of codes using FastScan SIMD.
     *
     * @param query The encoded query
     * @param block FastScan code block (32 codes)
     * @param aux Per-neighbor auxiliary data
     * @param popcounts Per-neighbor precomputed popcounts
     * @param count Number of valid codes in the block
     * @param out Output distances
     */
    static void compute_distance_batch(
        const QueryType& query,
        const FastScanCodeBlock<D, 32>& block,
        const VertexAuxData* aux,
        const uint16_t* popcounts,
        size_t count,
        float* out)
    {
        alignas(64) uint32_t fastscan_sums[32];

#if defined(__AVX2__)
        fastscan::compute_inner_products_avx2(query.lut, block, fastscan_sums);
#else
        fastscan::compute_inner_products_scalar(query.lut, block, count, fastscan_sums);
#endif

        fastscan::convert_to_distances(
            query, fastscan_sums, aux, popcounts, count, out);
    }
};

// Type aliases
using RaBitQPolicy128 = RaBitQMetricPolicy<128>;
using RaBitQPolicy256 = RaBitQMetricPolicy<256>;
using RaBitQPolicy512 = RaBitQMetricPolicy<512>;
using RaBitQPolicy1024 = RaBitQMetricPolicy<1024>;

}  // namespace cphnsw
