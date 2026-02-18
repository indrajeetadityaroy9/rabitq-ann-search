#pragma once

#include "../core/codes.hpp"
#include "../core/constants.hpp"
#include "fastscan_layout.hpp"
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <immintrin.h>

namespace cphnsw {

namespace fastscan {

constexpr size_t FLUSH_INTERVAL = constants::kFlushInterval;

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
    const float* nop_arr,
    const float* ip_qo_p_arr,
    const float* ip_cp_arr,
    const uint16_t* popcounts,
    size_t count,
    float* out_dist,
    float* out_lower,
    float dist_qp_sq)
{
    float A = query.coeff_fastscan;
    float B = query.coeff_popcount;
    float C = query.coeff_constant;
    float affine_a = query.affine_a;
    float affine_b = query.affine_b;
    float ip_qo_floor = query.ip_qo_floor;
    float dot_slack = query.dot_slack;
    float sqrt_dqp = std::sqrt(dist_qp_sq);

    // Near-zero parent distance disables lower-bound pruning.
    if (dist_qp_sq < constants::kNearZeroSq) {
        for (size_t i = 0; i < count; ++i) {
            float nop = nop_arr[i];
            out_dist[i] = nop * nop + dist_qp_sq;
            out_lower[i] = 0.0f;
        }
        return;
    }

    const __m256 vA = _mm256_set1_ps(A);
    const __m256 vB = _mm256_set1_ps(B);
    const __m256 vC = _mm256_set1_ps(C);
    const __m256 vaffine_a = _mm256_set1_ps(affine_a);
    const __m256 vaffine_b = _mm256_set1_ps(affine_b);
    const __m256 vfloor = _mm256_set1_ps(ip_qo_floor);
    const __m256 vdot_slack = _mm256_set1_ps(dot_slack);
    const __m256 vsqrt_dqp = _mm256_set1_ps(sqrt_dqp);
    const __m256 vdist_qp_sq = _mm256_set1_ps(dist_qp_sq);
    const __m256 vip_thresh = _mm256_set1_ps(constants::kIpQualityEps);
    const __m256 vzero = _mm256_setzero_ps();
    const __m256 vone = _mm256_set1_ps(1.0f);
    const __m256 vneg_one = _mm256_set1_ps(-1.0f);
    const __m256 vtwo = _mm256_set1_ps(2.0f);

    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m256 fs = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(fastscan_sums + i)));
        __m128i pc16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(popcounts + i));
        __m256 vpc = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(pc16));
        __m256 ip_approx = _mm256_fmadd_ps(vA, fs, _mm256_fmadd_ps(vB, vpc, vC));

        __m256 ip_qo_p = _mm256_loadu_ps(ip_qo_p_arr + i);
        __m256 ip_cp = _mm256_loadu_ps(ip_cp_arr + i);
        __m256 nop = _mm256_loadu_ps(nop_arr + i);

        ip_qo_p = _mm256_max_ps(ip_qo_p, vfloor);

        __m256 ip_corrected = _mm256_sub_ps(ip_approx, ip_cp);
        __m256 mask_good = _mm256_cmp_ps(ip_qo_p, vip_thresh, _CMP_GT_OQ);
        __m256 ip_est = _mm256_and_ps(mask_good, _mm256_div_ps(ip_corrected, ip_qo_p));

        ip_est = _mm256_fmadd_ps(vaffine_a, ip_est, vaffine_b);

        __m256 dist = _mm256_fmadd_ps(nop, nop, vdist_qp_sq);
        dist = _mm256_fnmadd_ps(_mm256_mul_ps(vtwo, nop), ip_est, dist);
        dist = _mm256_max_ps(dist, vzero);

        __m256 cos_upper = _mm256_div_ps(
            _mm256_add_ps(ip_est, vdot_slack),
            _mm256_max_ps(vsqrt_dqp, vip_thresh));
        cos_upper = _mm256_min_ps(_mm256_max_ps(cos_upper, vneg_one), vone);

        __m256 lower = _mm256_fmadd_ps(nop, nop, vdist_qp_sq);
        lower = _mm256_fnmadd_ps(_mm256_mul_ps(_mm256_mul_ps(vtwo, nop), vsqrt_dqp), cos_upper, lower);
        lower = _mm256_max_ps(lower, vzero);

        lower = _mm256_and_ps(mask_good, lower);

        _mm256_storeu_ps(out_dist + i, dist);
        _mm256_storeu_ps(out_lower + i, lower);
    }

    for (; i < count; ++i) {
        float ip_approx = A * static_cast<float>(fastscan_sums[i])
                        + B * static_cast<float>(popcounts[i]) + C;
        float ip_qo_p = std::max(ip_qo_p_arr[i], ip_qo_floor);
        float ip_corrected = ip_approx - ip_cp_arr[i];
        float ip_est = (ip_qo_p > constants::kIpQualityEps) ? ip_corrected / ip_qo_p : 0.0f;
        ip_est = affine_a * ip_est + affine_b;
        float nop = nop_arr[i];
        out_dist[i] = std::max(nop * nop + dist_qp_sq - 2.0f * nop * ip_est, 0.0f);

        if (ip_qo_p <= constants::kIpQualityEps) {
            out_lower[i] = 0.0f;
            continue;
        }

        float cos_upper = std::clamp((ip_est + dot_slack) / sqrt_dqp, -1.0f, 1.0f);
        out_lower[i] = std::max(nop * nop + dist_qp_sq - 2.0f * nop * sqrt_dqp * cos_upper, 0.0f);
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
    const float* nop_arr,
    const float* ip_qo_p_arr,
    const float* ip_cp_arr,
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
    float affine_a = query.affine_a;
    float affine_b = query.affine_b;
    float ip_qo_floor = query.ip_qo_floor;
    float dot_slack = query.dot_slack;
    float A_msb = query.coeff_fastscan;
    float B_msb = query.coeff_popcount;
    float sqrt_dqp = std::sqrt(dist_qp_sq);

    if (dist_qp_sq < constants::kNearZeroSq) {
        for (size_t i = 0; i < count; ++i) {
            float nop = nop_arr[i];
            out_dist[i] = nop * nop + dist_qp_sq;
            out_lower[i] = 0.0f;
        }
        return;
    }

    const __m256 vA_nbit = _mm256_set1_ps(A_nbit);
    const __m256 vB_nbit = _mm256_set1_ps(B_nbit);
    const __m256 vA_msb = _mm256_set1_ps(A_msb);
    const __m256 vB_msb = _mm256_set1_ps(B_msb);
    const __m256 vC = _mm256_set1_ps(C);
    const __m256 vaffine_a = _mm256_set1_ps(affine_a);
    const __m256 vaffine_b = _mm256_set1_ps(affine_b);
    const __m256 vfloor = _mm256_set1_ps(ip_qo_floor);
    const __m256 vdot_slack = _mm256_set1_ps(dot_slack);
    const __m256 vsqrt_dqp = _mm256_set1_ps(sqrt_dqp);
    const __m256 vdist_qp_sq = _mm256_set1_ps(dist_qp_sq);
    const __m256 vip_thresh = _mm256_set1_ps(constants::kIpQualityEps);
    const __m256 vzero = _mm256_setzero_ps();
    const __m256 vone = _mm256_set1_ps(1.0f);
    const __m256 vneg_one = _mm256_set1_ps(-1.0f);
    const __m256 vtwo = _mm256_set1_ps(2.0f);

    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m256 nbit_fs = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(nbit_fastscan_sums + i)));
        __m128i wpc16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(weighted_popcounts + i));
        __m256 vwpc = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(wpc16));
        __m256 ip_approx_nbit = _mm256_fmadd_ps(vA_nbit, nbit_fs, _mm256_fmadd_ps(vB_nbit, vwpc, vC));

        __m256 ip_qo_p = _mm256_loadu_ps(ip_qo_p_arr + i);
        __m256 ip_cp = _mm256_loadu_ps(ip_cp_arr + i);
        __m256 nop = _mm256_loadu_ps(nop_arr + i);

        ip_qo_p = _mm256_max_ps(ip_qo_p, vfloor);

        __m256 ip_corrected_nbit = _mm256_sub_ps(ip_approx_nbit, ip_cp);
        __m256 mask_good = _mm256_cmp_ps(ip_qo_p, vip_thresh, _CMP_GT_OQ);
        __m256 ip_est_nbit = _mm256_and_ps(mask_good, _mm256_div_ps(ip_corrected_nbit, ip_qo_p));

        ip_est_nbit = _mm256_fmadd_ps(vaffine_a, ip_est_nbit, vaffine_b);

        __m256 dist = _mm256_fmadd_ps(nop, nop, vdist_qp_sq);
        dist = _mm256_fnmadd_ps(_mm256_mul_ps(vtwo, nop), ip_est_nbit, dist);
        dist = _mm256_max_ps(dist, vzero);

        __m256 msb_fs = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(msb_fastscan_sums + i)));
        __m128i mpc16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(msb_popcounts + i));
        __m256 vmpc = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(mpc16));
        __m256 ip_approx_msb = _mm256_fmadd_ps(vA_msb, msb_fs, _mm256_fmadd_ps(vB_msb, vmpc, vC));

        __m256 ip_corrected_msb = _mm256_sub_ps(ip_approx_msb, ip_cp);
        __m256 ip_est_msb = _mm256_and_ps(mask_good, _mm256_div_ps(ip_corrected_msb, ip_qo_p));
        ip_est_msb = _mm256_fmadd_ps(vaffine_a, ip_est_msb, vaffine_b);

        __m256 cos_upper = _mm256_div_ps(
            _mm256_add_ps(ip_est_msb, vdot_slack),
            _mm256_max_ps(vsqrt_dqp, vip_thresh));
        cos_upper = _mm256_min_ps(_mm256_max_ps(cos_upper, vneg_one), vone);

        __m256 lower = _mm256_fmadd_ps(nop, nop, vdist_qp_sq);
        lower = _mm256_fnmadd_ps(_mm256_mul_ps(_mm256_mul_ps(vtwo, nop), vsqrt_dqp), cos_upper, lower);
        lower = _mm256_max_ps(lower, vzero);

        lower = _mm256_and_ps(mask_good, lower);

        _mm256_storeu_ps(out_dist + i, dist);
        _mm256_storeu_ps(out_lower + i, lower);
    }

    for (; i < count; ++i) {
        float ip_approx_nbit = A_nbit * static_cast<float>(nbit_fastscan_sums[i])
                             + B_nbit * static_cast<float>(weighted_popcounts[i]) + C;
        float ip_qo_p = std::max(ip_qo_p_arr[i], ip_qo_floor);
        float ip_corrected_nbit = ip_approx_nbit - ip_cp_arr[i];
        float ip_est_nbit = (ip_qo_p > constants::kIpQualityEps) ? ip_corrected_nbit / ip_qo_p : 0.0f;
        ip_est_nbit = affine_a * ip_est_nbit + affine_b;
        float nop = nop_arr[i];
        out_dist[i] = std::max(nop * nop + dist_qp_sq - 2.0f * nop * ip_est_nbit, 0.0f);

        if (ip_qo_p <= constants::kIpQualityEps) {
            out_lower[i] = 0.0f;
            continue;
        }

        float ip_approx_msb = A_msb * static_cast<float>(msb_fastscan_sums[i])
                            + B_msb * static_cast<float>(msb_popcounts[i]) + C;
        float ip_corrected_msb = ip_approx_msb - ip_cp_arr[i];
        float ip_est_msb = affine_a * (ip_corrected_msb / ip_qo_p) + affine_b;
        float cos_upper = std::clamp((ip_est_msb + dot_slack) / sqrt_dqp, -1.0f, 1.0f);
        out_lower[i] = std::max(nop * nop + dist_qp_sq - 2.0f * nop * sqrt_dqp * cos_upper, 0.0f);
    }
}

// Multi-plane lower-bound pass: uses top min(2, BitWidth) planes for tighter
// bounds than MSB-only. For 2-bit this uses full information; for 4-bit it
// captures ~60% of variance vs ~25% for MSB-only.
template <size_t D, size_t BitWidth>
inline void compute_msb_only_inner_products(
    const uint8_t lut[][16],
    const NbitFastScanCodeBlock<D, BitWidth, 32>& block,
    uint32_t* out_msb)
{
    constexpr size_t N_BOUND_PLANES = (BitWidth < 2) ? 1 : 2;

    // Compute MSB plane (always needed).
    compute_inner_products<D>(lut, block.planes[0], out_msb);

    if constexpr (N_BOUND_PLANES >= 2 && BitWidth >= 2) {
        // Add second-most-significant plane contribution.
        alignas(64) uint32_t plane1_sums[32];
        compute_inner_products<D>(lut, block.planes[1], plane1_sums);

        // Weight MSB by 2^(B-1) and plane 1 by 2^(B-2).
        // Scale out_msb: multiply existing by 2 (shift), add plane1.
        // Result: out_msb[i] = 2 * msb_sum[i] + plane1_sum[i]
        // This corresponds to top-2 weighted bits.
        for (size_t i = 0; i < 32; ++i) {
            out_msb[i] = 2 * out_msb[i] + plane1_sums[i];
        }
    }
}

// Lower bound conversion for multi-plane pre-filter sums.
// When BitWidth >= 2, msb_fastscan_sums contain weighted top-2 plane sums:
//   sum = 2 * plane0_sum + plane1_sum
// The coefficients are scaled by 1/K_partial where K_partial = 3 (for top-2).
// For 1-bit codes, this reduces to the original MSB-only formula.
template <size_t D, size_t BitWidth = 1>
inline void convert_msb_to_lower_bounds(
    const RaBitQQuery<D>& query,
    const uint32_t* msb_fastscan_sums,
    const float* nop_arr,
    const float* ip_qo_p_arr,
    const float* ip_cp_arr,
    const uint16_t* msb_popcounts,
    size_t count,
    float* out_lower,
    float dist_qp_sq)
{
    // For top-2 planes: K_partial = 2^1 + 2^0 = 3. Scale = 1/3.
    // For MSB-only (1 plane): no scaling needed.
    constexpr size_t N_BOUND_PLANES = (BitWidth < 2) ? 1 : 2;
    constexpr float K_PARTIAL = (N_BOUND_PLANES >= 2) ? 3.0f : 1.0f;
    constexpr float INV_K_PARTIAL = 1.0f / K_PARTIAL;

    float A = query.coeff_fastscan * INV_K_PARTIAL;
    float B = query.coeff_popcount * INV_K_PARTIAL;
    float C = query.coeff_constant;
    float affine_a = query.affine_a;
    float affine_b = query.affine_b;
    float ip_qo_floor = query.ip_qo_floor;
    float dot_slack = query.dot_slack;
    float sqrt_dqp = std::sqrt(dist_qp_sq);

    if (dist_qp_sq < constants::kNearZeroSq) {
        for (size_t i = 0; i < count; ++i) {
            out_lower[i] = 0.0f;
        }
        return;
    }

    for (size_t i = 0; i < count; ++i) {
        float ip_approx = A * static_cast<float>(msb_fastscan_sums[i])
                        + B * static_cast<float>(msb_popcounts[i])
                        + C;

        float ip_qo_p = std::max(ip_qo_p_arr[i], ip_qo_floor);
        float nop = nop_arr[i];

        if (ip_qo_p <= constants::kIpQualityEps) {
            out_lower[i] = 0.0f;
            continue;
        }

        float ip_corrected = ip_approx - ip_cp_arr[i];
        float ip_est = ip_corrected / ip_qo_p;
        ip_est = affine_a * ip_est + affine_b;

        float cos_upper = std::clamp((ip_est + dot_slack) / sqrt_dqp, -1.0f, 1.0f);

        out_lower[i] = nop * nop + dist_qp_sq - 2.0f * nop * sqrt_dqp * cos_upper;
        if (out_lower[i] < 0.0f) out_lower[i] = 0.0f;
    }
}

} // namespace fastscan

}  // namespace cphnsw
