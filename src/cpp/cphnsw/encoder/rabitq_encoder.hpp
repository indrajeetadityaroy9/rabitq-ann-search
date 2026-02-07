#pragma once

#include "../core/codes.hpp"
#include "../core/memory.hpp"
#include "rotation.hpp"
#include <cmath>
#include <algorithm>
#include <vector>
#include <limits>
#include <stdexcept>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cphnsw {

// ============================================================================
// RaBitQEncoder: Theoretically-grounded binary quantization (SIGMOD 2024)
// ============================================================================

/**
 * RaBitQEncoder: Implements RaBitQ encoding for database vectors and queries.
 *
 * KEY DIFFERENCES FROM CPEncoder:
 *   - Uses a SINGLE random rotation (SRHT) instead of K independent rotations
 *   - Sign quantization of ALL D dimensions (not K argmax operations)
 *   - Asymmetric: database is 1-bit quantized, query is 4-bit quantized
 *   - Produces unbiased distance estimates with provable error bounds
 *
 * DATABASE ENCODING (per vector):
 *   1. Subtract centroid: v = vec - c
 *   2. Normalize: o = v / ||v||, store dist_to_centroid = ||v||
 *   3. Apply SRHT: o' = P^{-1} * o (using RandomHadamardRotation)
 *   4. Sign quantization: bit[i] = (o'[i] >= 0) for all D dimensions
 *   5. Compute <o_bar, o> = ||o'||_{L1} / sqrt(D) (for unbiased estimator)
 *
 * QUERY ENCODING (per query, asymmetric):
 *   1. Subtract centroid, normalize → q, store ||q_r - c||
 *   2. Apply same SRHT → q'
 *   3. 4-bit uniform scalar quantization of q'
 *   4. Build D/4 FastScan lookup tables (16 entries each)
 *   5. Precompute linear coefficients for distance formula
 *
 * NORMALIZATION: The unnormalized SRHT T satisfies ||T*x|| = D^{3/2} * ||x||.
 * We define P^{-1} = T / D^{3/2} (orthogonal) so all inner products use
 * properly normalized values. The D^{3/2} factor is folded into norm_factor_.
 *
 * @tparam D Padded dimension (must equal next_power_of_two(input_dim))
 */
template <size_t D>
class RaBitQEncoder {
public:
    using CodeType = RaBitQCode<D>;
    using QueryType = RaBitQQuery<D>;

    static constexpr size_t DIMS = D;
    static constexpr size_t NUM_SUB_SEGMENTS = (D + 3) / 4;

    /**
     * Construct encoder.
     *
     * @param dim Original vector dimension (will be padded to D)
     * @param seed Random seed for the SRHT rotation matrix
     */
    RaBitQEncoder(size_t dim, uint64_t seed = 42)
        : dim_(dim)
        , rotation_(dim, seed)
        , padded_dim_(rotation_.padded_dim())
        , centroid_(dim, 0.0f)
        , has_centroid_(false) {

        // D must match the padded dimension from the SRHT
        if (padded_dim_ != D) {
            throw std::invalid_argument(
                "Template parameter D (" + std::to_string(D) +
                ") must equal padded dimension (" + std::to_string(padded_dim_) + ")");
        }

        // Precompute normalization constants
        float d_float = static_cast<float>(D);
        norm_factor_ = 1.0f / (d_float * std::sqrt(d_float)); // 1 / D^{3/2}
        inv_sqrt_d_ = 1.0f / std::sqrt(d_float);
        sqrt_d_ = std::sqrt(d_float);
    }

    // ========================================================================
    // Centroid Management
    // ========================================================================

    /**
     * Set centroid from external source.
     */
    void set_centroid(const float* c) {
        centroid_.assign(c, c + dim_);
        has_centroid_ = true;
    }

    /**
     * Compute centroid from a batch of vectors.
     */
    void compute_centroid(const float* vecs, size_t num_vecs) {
        centroid_.assign(dim_, 0.0f);
        for (size_t i = 0; i < num_vecs; ++i) {
            const float* v = vecs + i * dim_;
            for (size_t j = 0; j < dim_; ++j) {
                centroid_[j] += v[j];
            }
        }
        float inv_n = 1.0f / static_cast<float>(num_vecs);
        for (size_t j = 0; j < dim_; ++j) {
            centroid_[j] *= inv_n;
        }
        has_centroid_ = true;
    }

    bool has_centroid() const { return has_centroid_; }
    const std::vector<float>& centroid() const { return centroid_; }

    // ========================================================================
    // Database Vector Encoding
    // ========================================================================

    /**
     * Encode a single database vector to RaBitQ code.
     * Uses thread_local buffers for thread safety.
     */
    CodeType encode(const float* vec) const {
        thread_local AlignedVector<float> buf;
        thread_local std::vector<float> centered;
        buf.resize(padded_dim_);
        centered.resize(dim_);
        return encode_impl(vec, buf.data(), centered.data());
    }

    /**
     * Encode using caller-provided buffers (for explicit thread management).
     *
     * @param vec Input vector (dim_ floats)
     * @param buf SRHT work buffer (padded_dim_ floats, aligned)
     * @param centered_buf Centered vector buffer (dim_ floats)
     */
    CodeType encode_impl(const float* vec, float* buf, float* centered_buf) const {
        CodeType code;
        code.clear();

        // Step 1: Subtract centroid and compute ||v||
        float norm_sq = 0.0f;
        for (size_t i = 0; i < dim_; ++i) {
            float v = vec[i] - centroid_[i];
            centered_buf[i] = v;
            norm_sq += v * v;
        }
        float norm = std::sqrt(norm_sq);
        code.dist_to_centroid = norm;

        // Handle zero vector
        if (norm < 1e-10f) {
            code.ip_quantized_original = 0.0f;
            code.code_popcount = 0;
            return code;
        }

        // Step 2: Normalize to unit vector
        float inv_norm = 1.0f / norm;
        for (size_t i = 0; i < dim_; ++i) {
            centered_buf[i] *= inv_norm;
        }

        // Step 3: Apply SRHT and normalize by D^{3/2}
        // o' = T(o) / D^{3/2} = P^{-1} * o
        rotation_.apply_copy(centered_buf, buf);
        for (size_t i = 0; i < padded_dim_; ++i) {
            buf[i] *= norm_factor_;
        }

        // Step 4: Sign quantization + compute L1 norm
        float l1_norm = 0.0f;
        for (size_t i = 0; i < padded_dim_; ++i) {
            code.signs.set_bit(i, buf[i] >= 0.0f);
            l1_norm += std::abs(buf[i]);
        }

        // Step 5: <o_bar, o> = ||P^{-1}*o||_{L1} / sqrt(D)
        code.ip_quantized_original = l1_norm * inv_sqrt_d_;

        // Step 6: Precomputed popcount
        code.code_popcount = static_cast<uint16_t>(code.signs.popcount());

        return code;
    }

    /**
     * Encode a batch of database vectors (parallelized with OpenMP).
     * Computes centroid from the batch if not already set.
     */
    void encode_batch(const float* vecs, size_t num_vecs, CodeType* codes) {
        if (!has_centroid_ && num_vecs > 0) {
            compute_centroid(vecs, num_vecs);
        }

#ifdef _OPENMP
        #pragma omp parallel
        {
            AlignedVector<float> buf(padded_dim_);
            std::vector<float> centered(dim_);

            #pragma omp for schedule(static)
            for (size_t i = 0; i < num_vecs; ++i) {
                codes[i] = encode_impl(vecs + i * dim_, buf.data(), centered.data());
            }
        }
#else
        AlignedVector<float> buf(padded_dim_);
        std::vector<float> centered(dim_);
        for (size_t i = 0; i < num_vecs; ++i) {
            codes[i] = encode_impl(vecs + i * dim_, buf.data(), centered.data());
        }
#endif
    }

    // ========================================================================
    // Query Encoding (Asymmetric, 4-bit quantization + FastScan LUTs)
    // ========================================================================

    /**
     * Encode a query vector for RaBitQ distance estimation.
     *
     * Produces a RaBitQQuery containing:
     *   - FastScan LUTs for vpshufb-based batch distance computation
     *   - Linear coefficients for converting FastScan results to distances
     *   - Query norm and error bound parameters
     *
     * @param vec Query vector (dim_ floats)
     * @return Fully populated RaBitQQuery<D>
     */
    QueryType encode_query(const float* vec) const {
        thread_local AlignedVector<float> buf;
        thread_local std::vector<float> centered;
        buf.resize(padded_dim_);
        centered.resize(dim_);
        return encode_query_impl(vec, buf.data(), centered.data());
    }

    QueryType encode_query_impl(const float* vec, float* buf, float* centered_buf) const {
        QueryType query;

        // Step 1: Subtract centroid and compute ||q_r - c||
        float norm_sq = 0.0f;
        for (size_t i = 0; i < dim_; ++i) {
            float v = vec[i] - centroid_[i];
            centered_buf[i] = v;
            norm_sq += v * v;
        }
        float norm = std::sqrt(norm_sq);
        query.query_norm = norm;
        query.query_norm_sq = norm_sq;
        query.inv_sqrt_d = inv_sqrt_d_;
        query.error_epsilon = 1.9f;  // Default from paper

        // Handle zero query
        if (norm < 1e-10f) {
            std::memset(query.lut, 0, sizeof(query.lut));
            query.vl = 0.0f;
            query.delta = 0.0f;
            query.sum_qu = 0.0f;
            query.coeff_fastscan = 0.0f;
            query.coeff_popcount = 0.0f;
            query.coeff_constant = 0.0f;
            return query;
        }

        // Step 2: Normalize to unit vector
        float inv_norm = 1.0f / norm;
        for (size_t i = 0; i < dim_; ++i) {
            centered_buf[i] *= inv_norm;
        }

        // Step 3: Apply SRHT and normalize: q' = P^{-1} * q
        rotation_.apply_copy(centered_buf, buf);
        for (size_t i = 0; i < padded_dim_; ++i) {
            buf[i] *= norm_factor_;
        }

        // Step 4: Find min/max of q' for 4-bit quantization
        float vl = std::numeric_limits<float>::max();
        float vmax = std::numeric_limits<float>::lowest();
        for (size_t i = 0; i < padded_dim_; ++i) {
            vl = std::min(vl, buf[i]);
            vmax = std::max(vmax, buf[i]);
        }
        query.vl = vl;

        // Compute quantization step
        float range = vmax - vl;
        float delta = (range > 1e-10f) ? range / 15.0f : 1e-10f;
        query.delta = delta;
        float inv_delta = 1.0f / delta;

        // Step 5: 4-bit quantize q' → q_bar_u[i] in [0, 15]
        // and compute sum_qu
        // Use a temporary array for the quantized values
        uint8_t q_bar_u[D];
        float sum_qu = 0.0f;
        for (size_t i = 0; i < padded_dim_; ++i) {
            float val = (buf[i] - vl) * inv_delta;
            int ival = static_cast<int>(val + 0.5f);  // Round to nearest
            if (ival < 0) ival = 0;
            if (ival > 15) ival = 15;
            q_bar_u[i] = static_cast<uint8_t>(ival);
            sum_qu += static_cast<float>(ival);
        }
        query.sum_qu = sum_qu;

        // Step 6: Build FastScan LUTs
        // For each sub-segment j (4 consecutive dimensions):
        //   LUT[j][pattern] = sum of q_bar_u[4*j+b] for each bit b set in pattern
        for (size_t j = 0; j < NUM_SUB_SEGMENTS; ++j) {
            size_t base = j * 4;
            // Get the 4 quantized values for this sub-segment
            uint8_t vals[4] = {0, 0, 0, 0};
            for (size_t b = 0; b < 4 && (base + b) < padded_dim_; ++b) {
                vals[b] = q_bar_u[base + b];
            }

            // Enumerate all 16 patterns
            for (uint8_t p = 0; p < 16; ++p) {
                uint8_t sum = 0;
                if (p & 1) sum += vals[0];
                if (p & 2) sum += vals[1];
                if (p & 4) sum += vals[2];
                if (p & 8) sum += vals[3];
                query.lut[j][p] = sum;
            }
        }

        // Step 7: Precompute linear coefficients
        // From RaBitQ Equation (19):
        //   <x_bar, q_bar> = (2*delta/sqrt(D)) * <x_bar_b, q_bar_u>
        //                  + (2*vl/sqrt(D)) * popcount(x_bar_b)
        //                  - (delta/sqrt(D)) * sum_qu
        //                  - vl * sqrt(D)
        //
        // = coeff_fastscan * fastscan_result + coeff_popcount * popcount + coeff_constant
        query.coeff_fastscan = 2.0f * delta * inv_sqrt_d_;
        query.coeff_popcount = 2.0f * vl * inv_sqrt_d_;
        query.coeff_constant = -(delta * inv_sqrt_d_) * sum_qu - vl * sqrt_d_;

        return query;
    }

    // ========================================================================
    // Distance Computation Helpers
    // ========================================================================

    /**
     * Compute the full distance estimate from precomputed terms.
     *
     * @param query The encoded query
     * @param code The database vector's RaBitQ code
     * @param fastscan_sum Result of FastScan LUT accumulation (<x_bar_b, q_bar_u>)
     * @return Estimated ||o_r - q_r||^2
     */
    static float compute_distance_estimate(
        const QueryType& query,
        const CodeType& code,
        uint32_t fastscan_sum)
    {
        // Compute <x_bar, q_bar> from linear coefficients
        float ip_approx = query.coeff_fastscan * static_cast<float>(fastscan_sum)
                        + query.coeff_popcount * static_cast<float>(code.code_popcount)
                        + query.coeff_constant;

        // Unbiased estimator: <o, q>_est = <x_bar, q_bar> / <o_bar, o>
        float ip_est = (code.ip_quantized_original > 1e-10f)
                     ? ip_approx / code.ip_quantized_original
                     : 0.0f;

        // Full distance: ||o_r - q_r||^2 = ||o_r-c||^2 + ||q_r-c||^2
        //                                 - 2*||o_r-c||*||q_r-c||*<o,q>_est
        float dist_o = code.dist_to_centroid;
        float dist_q = query.query_norm;
        return dist_o * dist_o + dist_q * dist_q - 2.0f * dist_o * dist_q * ip_est;
    }

    /**
     * Compute the RaBitQ error bound for a given code.
     *
     * The bound (Theorem 3.2): with high probability,
     *   |<o,q>_est - <o,q>| <= epsilon * sqrt((1 - <o_bar,o>^2) / (<o_bar,o>^2 * D))
     *
     * @return The inner product error bound (add/subtract from ip_est)
     */
    static float compute_error_bound(const QueryType& query, const CodeType& code) {
        float ip_qo = code.ip_quantized_original;
        if (ip_qo < 1e-10f) return std::numeric_limits<float>::max();

        float ip_qo_sq = ip_qo * ip_qo;
        float variance = (1.0f - ip_qo_sq) / (ip_qo_sq * static_cast<float>(D));
        return query.error_epsilon * std::sqrt(variance);
    }

    // ========================================================================
    // Scalar Distance (non-SIMD path for single vector)
    // ========================================================================

    /**
     * Compute distance estimate for a single vector using scalar popcount.
     * Useful for verification and non-batch operations.
     */
    static float compute_distance_scalar(
        const QueryType& query,
        const CodeType& code)
    {
        // Compute <x_bar_b, q_bar_u> via LUT
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

        return compute_distance_estimate(query, code, fastscan_sum);
    }

    // ========================================================================
    // SymphonyQG Helper: Precompute per-neighbor auxiliary data
    // ========================================================================

    /**
     * Compute VertexAuxData for a neighbor relative to a parent vertex.
     *
     * In SymphonyQG, each neighbor stores:
     *   - dist_to_centroid: ||o_r - c|| where c is parent vertex
     *   - ip_quantized_original: <o_bar, o> (from the code)
     *   - ip_xbar_Pinv_c: <x_bar, P^{-1} * c> precomputed
     *
     * The SymphonyQG decomposition enables LUT reuse across vertices:
     *   <x_bar, P^{-1}*q> = (<x_bar, P^{-1}*q_r> - <x_bar, P^{-1}*c>) / ||q_r - c||
     *
     * @param neighbor_code The neighbor's RaBitQ code
     * @param parent_vec The parent vertex's raw vector (used as center c)
     * @param neighbor_vec The neighbor's raw vector
     * @return VertexAuxData for this edge
     */
    VertexAuxData compute_neighbor_aux(
        const CodeType& neighbor_code,
        const float* parent_vec,
        const float* neighbor_vec) const
    {
        VertexAuxData aux;

        // ||o_r - c|| where c = parent_vec
        float norm_sq = 0.0f;
        for (size_t i = 0; i < dim_; ++i) {
            float d = neighbor_vec[i] - parent_vec[i];
            norm_sq += d * d;
        }
        aux.dist_to_centroid = std::sqrt(norm_sq);
        aux.ip_quantized_original = neighbor_code.ip_quantized_original;

        // Compute <x_bar, P^{-1} * c> where c = parent_vec
        // P^{-1} * c = T(c) / D^{3/2} (using the same SRHT)
        thread_local AlignedVector<float> buf;
        buf.resize(padded_dim_);

        // Center parent_vec around the global centroid for SRHT
        // Actually, for SymphonyQG, c IS the parent vertex's raw vector
        // P^{-1} * c = SRHT(c_normalized) / D^{3/2}
        // But c here is the raw parent vector, not normalized.
        // We need <x_bar, P^{-1} * c> where c is used as-is after centering:
        // P^{-1} * (c - centroid) / ||c - centroid||...
        // This gets complex. For simplicity, store the precomputed scalar.

        // Apply SRHT to parent vector (centered, normalized)
        thread_local std::vector<float> centered;
        centered.resize(dim_);
        float c_norm_sq = 0.0f;
        for (size_t i = 0; i < dim_; ++i) {
            centered[i] = parent_vec[i] - centroid_[i];
            c_norm_sq += centered[i] * centered[i];
        }
        float c_norm = std::sqrt(c_norm_sq);
        if (c_norm > 1e-10f) {
            float inv_c_norm = 1.0f / c_norm;
            for (size_t i = 0; i < dim_; ++i) {
                centered[i] *= inv_c_norm;
            }
        }

        rotation_.apply_copy(centered.data(), buf.data());
        for (size_t i = 0; i < padded_dim_; ++i) {
            buf[i] *= norm_factor_;
        }

        // <x_bar, P^{-1}*c> = (1/sqrt(D)) * sum_i (2*x_bar_b[i]-1) * buf[i]
        float ip = 0.0f;
        for (size_t i = 0; i < padded_dim_; ++i) {
            float sign = neighbor_code.signs.get_bit(i) ? 1.0f : -1.0f;
            ip += sign * buf[i];
        }
        aux.ip_xbar_Pinv_c = ip * inv_sqrt_d_;

        return aux;
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    size_t dim() const { return dim_; }
    size_t padded_dim() const { return padded_dim_; }

private:
    size_t dim_;
    RandomHadamardRotation rotation_;
    size_t padded_dim_;
    float norm_factor_;   // 1 / D^{3/2}
    float inv_sqrt_d_;    // 1 / sqrt(D)
    float sqrt_d_;        // sqrt(D)
    std::vector<float> centroid_;
    bool has_centroid_;
};

// ============================================================================
// Type Aliases for Common Configurations
// ============================================================================

using RaBitQEncoder128 = RaBitQEncoder<128>;     // SIFT, GloVe-100
using RaBitQEncoder256 = RaBitQEncoder<256>;     // GloVe-200
using RaBitQEncoder512 = RaBitQEncoder<512>;     // Mid-range
using RaBitQEncoder1024 = RaBitQEncoder<1024>;   // Text embeddings (768→1024)

}  // namespace cphnsw
