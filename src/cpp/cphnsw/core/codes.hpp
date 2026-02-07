#pragma once

#include <cstdint>
#include <cstring>

namespace cphnsw {

// ============================================================================
// Binary Code Storage
// ============================================================================

/**
 * BinaryCodeStorage: Low-level storage for packed sign bits.
 *
 * Stores Bits sign bits packed into 64-bit words, aligned for SIMD access.
 *
 * @tparam Bits Number of sign bits to store
 */
template <size_t Bits>
struct alignas(64) BinaryCodeStorage {
    static constexpr size_t NUM_BITS = Bits;
    static constexpr size_t NUM_WORDS = (Bits + 63) / 64;

    uint64_t signs[NUM_WORDS];

    void clear() {
        for (size_t i = 0; i < NUM_WORDS; ++i) {
            signs[i] = 0;
        }
    }

    void set_bit(size_t idx, bool value) {
        if (value) {
            signs[idx / 64] |= (1ULL << (idx % 64));
        } else {
            signs[idx / 64] &= ~(1ULL << (idx % 64));
        }
    }

    bool get_bit(size_t idx) const {
        return (signs[idx / 64] >> (idx % 64)) & 1;
    }

    bool operator==(const BinaryCodeStorage& other) const {
        for (size_t i = 0; i < NUM_WORDS; ++i) {
            if (signs[i] != other.signs[i]) return false;
        }
        return true;
    }

    /**
     * Count the number of set bits (population count).
     */
    uint32_t popcount() const {
        uint32_t count = 0;
        for (size_t i = 0; i < NUM_WORDS; ++i) {
            count += static_cast<uint32_t>(__builtin_popcountll(signs[i]));
        }
        return count;
    }
};

// ============================================================================
// RaBitQ Data Structures (SIGMOD 2024)
// ============================================================================

/**
 * RaBitQCode: Per-vector quantization code from RaBitQ.
 *
 * Stores the D-bit sign quantization code x_bar_b along with auxiliary
 * scalars needed for the unbiased distance estimator.
 *
 * The quantized vector is: o_bar = P * x_bar, where
 *   x_bar[i] = +1/sqrt(D) if x_bar_b[i] = 1
 *   x_bar[i] = -1/sqrt(D) if x_bar_b[i] = 0
 *
 * @tparam D Padded dimension (next power of 2 of input dim)
 */
template <size_t D>
struct RaBitQCode {
    static constexpr size_t DIMS = D;
    static constexpr size_t NUM_WORDS = (D + 63) / 64;

    BinaryCodeStorage<D> signs;       // D-bit quantization code x_bar_b
    float dist_to_centroid;           // ||o_r - c|| (L2 norm of raw vector minus centroid)
    float ip_quantized_original;      // <o_bar, o> = ||P^{-1} * o||_{L1} / sqrt(D)
    uint16_t code_popcount;           // popcount(x_bar_b) = sum of set bits

    void clear() {
        signs.clear();
        dist_to_centroid = 0.0f;
        ip_quantized_original = 0.0f;
        code_popcount = 0;
    }
};

/**
 * VertexAuxData: Per-neighbor auxiliary data stored in graph edges.
 *
 * In SymphonyQG's design, each neighbor stores precomputed scalars
 * relative to the parent vertex (used as normalization center c).
 *
 * The key SymphonyQG optimization: <x_bar, P^{-1} * q> is decomposed as
 *   (<x_bar, P^{-1} * q_r> - <x_bar, P^{-1} * c>) / ||q_r - c||
 * where <x_bar, P^{-1} * c> is precomputed per-neighbor at index time.
 */
struct VertexAuxData {
    float dist_to_centroid;           // ||o_r - c|| where c is parent vertex
    float ip_quantized_original;      // <o_bar, o> for unbiased estimator
    float ip_xbar_Pinv_c;            // <x_bar, P^{-1} * c> precomputed term
};

/**
 * RaBitQQuery: Per-query precomputed data for RaBitQ distance estimation.
 *
 * Contains FastScan lookup tables and linear coefficients for the
 * distance formula. The LUTs enable vpshufb-based SIMD computation
 * of <x_bar_b, q_bar_u> for 32 codes simultaneously.
 *
 * Distance formula (per data vector):
 *   <x_bar, q_bar> = coeff_fastscan * fastscan_sum
 *                   + coeff_popcount * popcount(x_bar_b)
 *                   + coeff_constant
 *
 * Then: <o,q>_est = <x_bar, q_bar> / <o_bar, o>
 * And:  ||o_r - q_r||^2 = ||o_r-c||^2 + ||q_r-c||^2
 *                        - 2 * ||o_r-c|| * ||q_r-c|| * <o,q>_est
 *
 * @tparam D Padded dimension
 */
template <size_t D>
struct RaBitQQuery {
    static constexpr size_t DIMS = D;
    static constexpr size_t NUM_SUB_SEGMENTS = (D + 3) / 4;

    // FastScan LUTs: D/4 sub-tables, each with 16 uint8 entries
    alignas(64) uint8_t lut[NUM_SUB_SEGMENTS][16];

    // Query quantization parameters (4-bit uniform scalar quantization)
    float vl;                         // min of P^{-1} * q' (quantization lower bound)
    float delta;                      // (vmax - vl) / 15 (quantization step)
    float sum_qu;                     // sum of all quantized query values

    // Distance computation context
    float query_norm;                 // ||q_r - c|| (norm of query minus centroid)
    float query_norm_sq;              // ||q_r - c||^2

    // Linear coefficients for: <x_bar, q_bar> = A * fastscan_sum + B * popcount + C
    float coeff_fastscan;             // 2 * delta / sqrt(D)
    float coeff_popcount;             // 2 * vl / sqrt(D)
    float coeff_constant;             // -(delta/sqrt(D)) * sum_qu - vl * sqrt(D)

    // Error bound parameter (Theorem 3.2)
    float error_epsilon;              // epsilon_0 (typically 1.9)
    float inv_sqrt_d;                 // 1 / sqrt(D)
};

// Type aliases for common configurations
using RaBitQCode128 = RaBitQCode<128>;     // SIFT, GloVe-100
using RaBitQCode256 = RaBitQCode<256>;     // GloVe-200
using RaBitQCode512 = RaBitQCode<512>;     // Mid-range embeddings
using RaBitQCode1024 = RaBitQCode<1024>;   // Text embeddings (768 padded)

using RaBitQQuery128 = RaBitQQuery<128>;
using RaBitQQuery256 = RaBitQQuery<256>;
using RaBitQQuery512 = RaBitQQuery<512>;
using RaBitQQuery1024 = RaBitQQuery<1024>;

}  // namespace cphnsw
