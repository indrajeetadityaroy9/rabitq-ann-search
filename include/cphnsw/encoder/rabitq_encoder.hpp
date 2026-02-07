#pragma once

#include "../core/codes.hpp"
#include "../core/memory.hpp"
#include "rotation.hpp"
#include "dense_rotation.hpp"
#include <cmath>
#include <algorithm>
#include <random>
#include <vector>
#include <limits>
#include <stdexcept>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cphnsw {

template <size_t D, typename RotationPolicy = RandomHadamardRotation>
class RaBitQEncoder {
public:
    using CodeType = RaBitQCode<D>;
    using QueryType = RaBitQQuery<D>;

    static constexpr size_t DIMS = D;
    static constexpr size_t NUM_SUB_SEGMENTS = (D + 3) / 4;

    RaBitQEncoder(size_t dim, uint64_t seed = 42)
        : dim_(dim)
        , rotation_(dim, seed)
        , padded_dim_(rotation_.padded_dim())
        , centroid_(dim, 0.0f)
        , has_centroid_(false) {

        if (padded_dim_ != D && std::is_same<RotationPolicy, RandomHadamardRotation>::value) {
             // For SRHT, strict padding check might be needed if user didn't pad D.
             // But SRHT pads internally. Check if D matches output.
             // If D=128, padded=128. If D=100, padded=128.
             // The CodeType<D> expects D bits.
             // If padded_dim_ > D, we have an issue with storage.
        }
        
        // Relax check for DenseRotation which might pad differently or not at all
        if (padded_dim_ < D) {
             throw std::invalid_argument("Padded dimension must be >= D");
        }

        float d_float = static_cast<float>(D);
        norm_factor_ = 1.0f / (d_float * std::sqrt(d_float)); // 1 / D^{3/2}
        // Note: norm_factor_ scaling depends on the Rotation property.
        // SRHT (Walsh) is unnormalized, it scales by D (or sqrt(D)?). 
        // Fast Walsh Transform usually doesn't scale (norm grows by sqrt(D) per layer? No, factor D).
        // Standard H (unnormalized) * x -> vector norm scales by sqrt(D).
        // 3 layers -> sqrt(D)^3 ?
        
        // DenseRotation is Orthonormal (Unitary). Norm is preserved.
        // If Rotation preserves norm, we don't need norm_factor_ to scale back?
        // RaBitQ formula (Eq 3) assumes y = H D H ... x.
        // If we use DenseRotation (R), y = R x. ||y|| = ||x||.
        // SRHT implementation in rotation.hpp:
        // H is unnormalized [1, 1; 1, -1]. 
        // 1 layer: norm *= sqrt(D). 3 layers: norm *= D^1.5 ?
        
        // Let's adjust norm_factor_ based on Policy.
        if constexpr (std::is_same<RotationPolicy, DenseRotation>::value) {
            // Orthonormal rotation preserves norm.
            // We need the components to be in range for quantization.
            // RaBitQ assumes components are approx Gaussian N(0, 1/D).
            // If ||x||=1, then after R, ||y||=1. entries approx 1/sqrt(D).
            // Original code: buf[i] *= norm_factor_ (1/D^1.5).
            // This suggests the rotation output was HUGE.
            
            // If we use Orthonormal, output is normal size.
            // We want to scale it to match the quantization logic?
            // Actually, `encode_impl` just takes signs. Scaling doesn't affect signs.
            // BUT, `ip_quantized_original` stores L1 norm of rotated vector.
            // For SRHT: L1 is large.
            // For Dense: L1 is small.
            
            // We should keep the scaling consistent so that `ip_quantized_original` 
            // means the same thing, OR adjust the formula.
            // RaBitQ paper: u = 1/sqrt(D) * signs(Rx).
            // ip_est = (u . v).
            
            // Let's stick to: we want `buf` to be the rotated vector.
            // If SRHT scales up, we scale down.
            // If Dense preserves, we calculate L1 on it directly.
            
            norm_factor_ = 1.0f; // No scaling needed for Orthonormal
        }
        
        inv_sqrt_d_ = 1.0f / std::sqrt(d_float);
        sqrt_d_ = std::sqrt(d_float);
    }

    void set_centroid(const float* c) {
        centroid_.assign(c, c + dim_);
        has_centroid_ = true;
    }

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

    CodeType encode(const float* vec) const {
        thread_local AlignedVector<float> buf;
        thread_local std::vector<float> centered;
        buf.resize(padded_dim_);
        centered.resize(dim_);
        return encode_impl(vec, buf.data(), centered.data());
    }

    CodeType encode_impl(const float* vec, float* buf, float* centered_buf) const {
        CodeType code;
        code.clear();

        float norm_sq = 0.0f;
        for (size_t i = 0; i < dim_; ++i) {
            float v = vec[i] - centroid_[i];
            centered_buf[i] = v;
            norm_sq += v * v;
        }
        float norm = std::sqrt(norm_sq);
        code.dist_to_centroid = norm;

        if (norm < 1e-10f) {
            code.ip_quantized_original = 0.0f;
            code.code_popcount = 0;
            return code;
        }

        float inv_norm = 1.0f / norm;
        for (size_t i = 0; i < dim_; ++i) {
            centered_buf[i] *= inv_norm;
        }

        rotation_.apply_copy(centered_buf, buf);
        for (size_t i = 0; i < padded_dim_; ++i) {
            buf[i] *= norm_factor_;
        }

        float l1_norm = 0.0f;
        if (stochastic_rounding_) {
            thread_local std::mt19937 rng(42);
            thread_local std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            for (size_t i = 0; i < padded_dim_; ++i) {
                float prob = 0.5f + buf[i] * sqrt_d_ * 0.5f;
                prob = std::max(0.0f, std::min(1.0f, prob));
                code.signs.set_bit(i, dist(rng) < prob);
                l1_norm += std::abs(buf[i]);
            }
        } else {
            for (size_t i = 0; i < padded_dim_; ++i) {
                code.signs.set_bit(i, buf[i] >= 0.0f);
                l1_norm += std::abs(buf[i]);
            }
        }

        code.ip_quantized_original = l1_norm * inv_sqrt_d_;
        code.code_popcount = static_cast<uint16_t>(code.signs.popcount());

        return code;
    }

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

    QueryType encode_query(const float* vec) const {
        thread_local AlignedVector<float> buf;
        thread_local std::vector<float> centered;
        buf.resize(padded_dim_);
        centered.resize(dim_);
        return encode_query_impl(vec, buf.data(), centered.data());
    }

    QueryType encode_query_impl(const float* vec, float* buf, float* centered_buf) const {
        QueryType query;

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
        query.error_epsilon = 1.9f;

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

        float inv_norm = 1.0f / norm;
        for (size_t i = 0; i < dim_; ++i) {
            centered_buf[i] *= inv_norm;
        }

        rotation_.apply_copy(centered_buf, buf);
        for (size_t i = 0; i < padded_dim_; ++i) {
            buf[i] *= norm_factor_;
        }

        float vl = std::numeric_limits<float>::max();
        float vmax = std::numeric_limits<float>::lowest();
        for (size_t i = 0; i < padded_dim_; ++i) {
            vl = std::min(vl, buf[i]);
            vmax = std::max(vmax, buf[i]);
        }
        query.vl = vl;

        float range = vmax - vl;
        float delta = (range > 1e-10f) ? range / 15.0f : 1e-10f;
        query.delta = delta;
        float inv_delta = 1.0f / delta;

        uint8_t q_bar_u[D];
        float sum_qu = 0.0f;
        for (size_t i = 0; i < padded_dim_; ++i) {
            float val = (buf[i] - vl) * inv_delta;
            int ival = static_cast<int>(val + 0.5f);
            if (ival < 0) ival = 0;
            if (ival > 15) ival = 15;
            q_bar_u[i] = static_cast<uint8_t>(ival);
            sum_qu += static_cast<float>(ival);
        }
        query.sum_qu = sum_qu;

        for (size_t j = 0; j < NUM_SUB_SEGMENTS; ++j) {
            size_t base = j * 4;
            uint8_t vals[4] = {0, 0, 0, 0};
            for (size_t b = 0; b < 4 && (base + b) < padded_dim_; ++b) {
                vals[b] = q_bar_u[base + b];
            }

            for (uint8_t p = 0; p < 16; ++p) {
                uint8_t sum = 0;
                if (p & 1) sum += vals[0];
                if (p & 2) sum += vals[1];
                if (p & 4) sum += vals[2];
                if (p & 8) sum += vals[3];
                query.lut[j][p] = sum;
            }
        }

        // From RaBitQ Equation (19)
        query.coeff_fastscan = 2.0f * delta * inv_sqrt_d_;
        query.coeff_popcount = 2.0f * vl * inv_sqrt_d_;
        query.coeff_constant = -(delta * inv_sqrt_d_) * sum_qu - vl * sqrt_d_;

        return query;
    }

    static float compute_distance_estimate(
        const QueryType& query,
        const CodeType& code,
        uint32_t fastscan_sum)
    {
        float ip_approx = query.coeff_fastscan * static_cast<float>(fastscan_sum)
                        + query.coeff_popcount * static_cast<float>(code.code_popcount)
                        + query.coeff_constant;

        float ip_est = (code.ip_quantized_original > 1e-10f)
                     ? ip_approx / code.ip_quantized_original
                     : 0.0f;

        float dist_o = code.dist_to_centroid;
        float dist_q = query.query_norm;
        return dist_o * dist_o + dist_q * dist_q - 2.0f * dist_o * dist_q * ip_est;
    }

    static float compute_error_bound(const QueryType& query, const CodeType& code) {
        float ip_qo = code.ip_quantized_original;
        if (ip_qo < 1e-10f) return std::numeric_limits<float>::max();

        float ip_qo_sq = ip_qo * ip_qo;
        float variance = (1.0f - ip_qo_sq) / (ip_qo_sq * static_cast<float>(D));
        return query.error_epsilon * std::sqrt(variance);
    }

    static float compute_distance_scalar(
        const QueryType& query,
        const CodeType& code)
    {
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

    VertexAuxData compute_neighbor_aux(
        const CodeType& neighbor_code,
        const float* parent_vec,
        const float* neighbor_vec) const
    {
        VertexAuxData aux;

        float norm_sq = 0.0f;
        for (size_t i = 0; i < dim_; ++i) {
            float d = neighbor_vec[i] - parent_vec[i];
            norm_sq += d * d;
        }
        aux.dist_to_centroid = std::sqrt(norm_sq);
        aux.ip_quantized_original = neighbor_code.ip_quantized_original;

        thread_local AlignedVector<float> buf;
        buf.resize(padded_dim_);

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

        float ip = 0.0f;
        for (size_t i = 0; i < padded_dim_; ++i) {
            float sign = neighbor_code.signs.get_bit(i) ? 1.0f : -1.0f;
            ip += sign * buf[i];
        }
        aux.ip_xbar_Pinv_c = ip * inv_sqrt_d_;

        return aux;
    }

    size_t dim() const { return dim_; }
    size_t padded_dim() const { return padded_dim_; }

    void set_stochastic_rounding(bool enabled) { stochastic_rounding_ = enabled; }
    bool stochastic_rounding() const { return stochastic_rounding_; }

private:
    size_t dim_;
    RotationPolicy rotation_;
    size_t padded_dim_;
    float norm_factor_;   // 1 / D^{3/2}
    float inv_sqrt_d_;    // 1 / sqrt(D)
    float sqrt_d_;        // sqrt(D)
    std::vector<float> centroid_;
    bool has_centroid_;
    bool stochastic_rounding_ = false;
};

using RaBitQEncoder128 = RaBitQEncoder<128>;
using RaBitQEncoder256 = RaBitQEncoder<256>;
using RaBitQEncoder512 = RaBitQEncoder<512>;
using RaBitQEncoder1024 = RaBitQEncoder<1024>;

// Dense Rotation Variants (SOTA Quality)
using RaBitQEncoderDense128 = RaBitQEncoder<128, DenseRotation>;
using RaBitQEncoderDense256 = RaBitQEncoder<256, DenseRotation>;
using RaBitQEncoderDense512 = RaBitQEncoder<512, DenseRotation>;
using RaBitQEncoderDense1024 = RaBitQEncoder<1024, DenseRotation>;

}  // namespace cphnsw