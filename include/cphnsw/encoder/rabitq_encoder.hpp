#pragma once

#include "../core/codes.hpp"
#include "../core/memory.hpp"
#include "../core/adaptive_defaults.hpp"
#include "rotation.hpp"
#include <cmath>
#include <algorithm>
#include <vector>
#include <limits>
#include <stdexcept>

#include <omp.h>

namespace cphnsw {

template <size_t D>
struct EncoderWorkspace {
    AlignedVector<float> buf;
    std::vector<float> centered;
    AlignedVector<uint8_t> q_bar_u;

    explicit EncoderWorkspace(size_t padded_dim, size_t dim)
        : buf(padded_dim), centered(dim), q_bar_u(padded_dim) {}
};

template <typename Derived, size_t D>
class RaBitQEncoderBase {
public:
    using QueryType = RaBitQQuery<D>;

    static constexpr size_t NUM_SUB_SEGMENTS = (D + 3) / 4;

    RaBitQEncoderBase(size_t dim, uint64_t seed = 42)
        : dim_(dim)
        , rotation_(dim, seed)
        , padded_dim_(rotation_.padded_dim())
        , centroid_(dim, 0.0f)
        , has_centroid_(false) {

        if (padded_dim_ < D) {
            throw std::invalid_argument("Padded dimension must be >= D");
        }

        float d_float = static_cast<float>(D);
        norm_factor_ = 1.0f / (d_float * std::sqrt(d_float));
        inv_sqrt_d_ = 1.0f / std::sqrt(d_float);
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

    template <typename CodeType>
    void encode_batch(const float* vecs, size_t num_vecs, CodeType* codes) {
        if (!has_centroid_ && num_vecs > 0) {
            compute_centroid(vecs, num_vecs);
        }

        #pragma omp parallel
        {
            AlignedVector<float> buf(padded_dim_);
            std::vector<float> centered(dim_);

            #pragma omp for schedule(static)
            for (size_t i = 0; i < num_vecs; ++i) {
                codes[i] = static_cast<const Derived*>(this)->encode_impl(
                    vecs + i * dim_, buf.data(), centered.data());
            }
        }
    }

    QueryType encode_query_raw(const float* vec) const {
        thread_local EncoderWorkspace<D> ws(padded_dim_, dim_);
        ws.buf.resize(padded_dim_);
        ws.q_bar_u.resize(padded_dim_);
        return encode_query_raw_impl(vec, ws.buf.data(), ws.q_bar_u.data());
    }

    void rotate_raw_vector(const float* vec, float* out) const {
        rotation_.apply_copy(vec, out);
        for (size_t i = 0; i < padded_dim_; ++i) {
            out[i] *= norm_factor_;
        }
    }

    float compute_ip_code_parent(const BinaryCodeStorage<D>& code,
                                  const float* rotated_parent) const {
        float ip = 0.0f;
        for (size_t i = 0; i < padded_dim_; ++i) {
            float sign = code.get_bit(i) ? 1.0f : -1.0f;
            ip += sign * rotated_parent[i];
        }
        return ip * inv_sqrt_d_;
    }

    void build_lut(const float* buf, uint8_t* q_bar_u, QueryType& query) const {
        constexpr size_t NUM_SUB_SEGMENTS = (D + 3) / 4;

        float vl = buf[0], vmax = buf[0];
        for (size_t i = 1; i < padded_dim_; ++i) {
            if (buf[i] < vl) vl = buf[i];
            if (buf[i] > vmax) vmax = buf[i];
        }

        float delta = (vmax - vl) / 15.0f;
        if (delta < adaptive_defaults::division_epsilon()) delta = adaptive_defaults::division_epsilon();
        float inv_delta = 1.0f / delta;

        float sum_qu = 0.0f;
        for (size_t i = 0; i < padded_dim_; ++i) {
            float val = (buf[i] - vl) * inv_delta;
            int u = static_cast<int>(val + 0.5f);
            if (u < 0) u = 0;
            if (u > 15) u = 15;
            q_bar_u[i] = static_cast<uint8_t>(u);
            sum_qu += static_cast<float>(u);
        }

        for (size_t j = 0; j < NUM_SUB_SEGMENTS; ++j) {
            for (uint8_t p = 0; p < 16; ++p) {
                uint8_t sum = 0;
                for (size_t b = 0; b < 4; ++b) {
                    size_t idx = j * 4 + b;
                    if (idx < D && (p & (1u << b))) {
                        sum += q_bar_u[idx];
                    }
                }
                query.lut[j][p] = sum;
            }
        }

        float Df = static_cast<float>(D);
        query.coeff_fastscan = 2.0f * delta * inv_sqrt_d_;
        query.coeff_popcount = 2.0f * vl * inv_sqrt_d_;
        query.coeff_constant = -(Df * vl + delta * sum_qu) * inv_sqrt_d_;
    }

    template <typename CodeType>
    VertexAuxData compute_neighbor_aux(
        const CodeType& neighbor_code,
        const float* parent_vec,
        const float* neighbor_vec,
        const float* rotated_parent,
        BinaryCodeStorage<D>* out_code = nullptr) const
    {
        VertexAuxData aux;

        if (out_code) {
            out_code->clear();

            alignas(32) float diff_op[D];
            float nop_sq = 0.0f;
            for (size_t i = 0; i < dim_; ++i) {
                diff_op[i] = neighbor_vec[i] - parent_vec[i];
                nop_sq += diff_op[i] * diff_op[i];
            }
            for (size_t i = dim_; i < D; ++i) diff_op[i] = 0.0f;

            float nop = std::sqrt(nop_sq);
            aux.dist_to_centroid = nop;

            if (nop < adaptive_defaults::norm_epsilon(D)) {
                aux.ip_quantized_original = 0.0f;
                aux.ip_code_parent = 0.0f;
                return aux;
            }

            float inv_nop = 1.0f / nop;
            for (size_t i = 0; i < D; ++i) diff_op[i] *= inv_nop;

            alignas(32) float rotated[D];
            rotation_.apply_copy(diff_op, rotated);
            for (size_t i = 0; i < padded_dim_; ++i) rotated[i] *= norm_factor_;

            float l1_norm = 0.0f;
            for (size_t i = 0; i < padded_dim_; ++i) {
                out_code->set_bit(i, rotated[i] >= 0.0f);
                l1_norm += std::abs(rotated[i]);
            }

            aux.ip_quantized_original = l1_norm * inv_sqrt_d_;

            // SymphonyQG Eq 6: precomputed parent-relative inner product
            aux.ip_code_parent = compute_ip_code_parent(*out_code, rotated_parent);

            return aux;
        }

        aux.dist_to_centroid = neighbor_code.dist_to_centroid;
        aux.ip_quantized_original = neighbor_code.ip_quantized_original;
        aux.ip_code_parent = 0.0f;
        return aux;
    }

protected:
    size_t dim_;
    RandomHadamardRotation rotation_;
    size_t padded_dim_;
    float norm_factor_;
    float inv_sqrt_d_;
    std::vector<float> centroid_;
    bool has_centroid_;

private:
    QueryType encode_query_raw_impl(const float* vec, float* buf,
                                     uint8_t* q_bar_u) const {
        QueryType query;

        query.error_epsilon = 0.0f;

        rotation_.apply_copy(vec, buf);
        for (size_t i = 0; i < padded_dim_; ++i) {
            buf[i] *= norm_factor_;
        }

        build_lut(buf, q_bar_u, query);

        return query;
    }
};


template <size_t D>
class RaBitQEncoder : public RaBitQEncoderBase<RaBitQEncoder<D>, D> {
    using Base = RaBitQEncoderBase<RaBitQEncoder<D>, D>;
    friend Base;

public:
    using CodeType = RaBitQCode<D>;
    using QueryType = RaBitQQuery<D>;
    static constexpr size_t DIMS = D;
    static constexpr size_t NUM_SUB_SEGMENTS = (D + 3) / 4;

    using Base::Base;

    CodeType encode_impl(const float* vec, float* buf, float* centered_buf) const {
        CodeType code;
        code.clear();

        float norm_sq = 0.0f;
        for (size_t i = 0; i < this->dim_; ++i) {
            float v = vec[i] - this->centroid_[i];
            centered_buf[i] = v;
            norm_sq += v * v;
        }
        float norm = std::sqrt(norm_sq);
        code.dist_to_centroid = norm;

        if (norm < adaptive_defaults::norm_epsilon(D)) {
            code.ip_quantized_original = 0.0f;
            return code;
        }

        float inv_norm = 1.0f / norm;
        for (size_t i = 0; i < this->dim_; ++i) {
            centered_buf[i] *= inv_norm;
        }

        this->rotation_.apply_copy(centered_buf, buf);
        for (size_t i = 0; i < this->padded_dim_; ++i) {
            buf[i] *= this->norm_factor_;
        }

        float l1_norm = 0.0f;
        for (size_t i = 0; i < this->padded_dim_; ++i) {
            code.signs.set_bit(i, buf[i] >= 0.0f);
            l1_norm += std::abs(buf[i]);
        }

        code.ip_quantized_original = l1_norm * this->inv_sqrt_d_;

        return code;
    }
};


template <size_t D, size_t BitWidth>
class NbitRaBitQEncoder : public RaBitQEncoderBase<NbitRaBitQEncoder<D, BitWidth>, D> {
    using Base = RaBitQEncoderBase<NbitRaBitQEncoder<D, BitWidth>, D>;
    friend Base;

public:
    using CodeType = NbitRaBitQCode<D, BitWidth>;
    using QueryType = RaBitQQuery<D>;
    static constexpr size_t DIMS = D;
    static constexpr size_t BIT_WIDTH = BitWidth;
    static constexpr size_t NUM_SUB_SEGMENTS = (D + 3) / 4;
    static constexpr int K_INT = (1 << BitWidth) - 1;
    static constexpr float K = static_cast<float>(K_INT);

    struct NbitNeighborResult {
        NbitCodeStorage<D, BitWidth> code;
        VertexAuxData aux;
    };

    using Base::Base;

    // Encode parent-relative direction with full N-bit CAQ precision.
    // All planes are parent-relative, making the distance formula consistent.
    NbitNeighborResult compute_neighbor_aux_nbit(
        const float* parent_vec, const float* neighbor_vec,
        const float* rotated_parent) const
    {
        NbitNeighborResult result;
        result.code.clear();

        alignas(32) float diff[D];
        float nop_sq = 0.0f;
        for (size_t i = 0; i < this->dim_; ++i) {
            diff[i] = neighbor_vec[i] - parent_vec[i];
            nop_sq += diff[i] * diff[i];
        }
        for (size_t i = this->dim_; i < D; ++i) diff[i] = 0.0f;

        float nop = std::sqrt(nop_sq);
        result.aux.dist_to_centroid = nop;

        if (nop < adaptive_defaults::norm_epsilon(D)) {
            result.aux.ip_quantized_original = 0.0f;
            result.aux.ip_code_parent = 0.0f;
            return result;
        }

        float inv_nop = 1.0f / nop;
        for (size_t i = 0; i < D; ++i) diff[i] *= inv_nop;

        alignas(32) float rotated[D];
        this->rotation_.apply_copy(diff, rotated);
        for (size_t i = 0; i < this->padded_dim_; ++i) rotated[i] *= this->norm_factor_;

        const size_t pd = this->padded_dim_;

        // Delta-based initialization
        float buf_min = rotated[0], buf_max = rotated[0];
        for (size_t i = 1; i < pd; ++i) {
            if (rotated[i] < buf_min) buf_min = rotated[i];
            if (rotated[i] > buf_max) buf_max = rotated[i];
        }
        float delta = (buf_max - buf_min) / K;
        if (delta < adaptive_defaults::coordinate_epsilon(D))
            delta = adaptive_defaults::coordinate_epsilon(D);
        float inv_delta = 1.0f / delta;

        thread_local std::vector<int> pr_codes;
        pr_codes.resize(pd);

        float dot_co = 0.0f, norm_c_sq = 0.0f;
        for (size_t i = 0; i < pd; ++i) {
            float val = (rotated[i] - buf_min) * inv_delta;
            int u = static_cast<int>(val + 0.5f);
            if (u < 0) u = 0;
            if (u > K_INT) u = K_INT;
            pr_codes[i] = u;
            float c = (2.0f * u - K) / K;
            dot_co += c * rotated[i];
            norm_c_sq += c * c;
        }

        // CAQ coordinate descent
        constexpr size_t max_iters = adaptive_defaults::caq_max_iterations();
        for (size_t iter = 0; iter < max_iters; ++iter) {
            bool changed = false;
            for (size_t i = 0; i < pd; ++i) {
                int old_u = pr_codes[i];
                float old_c = (2.0f * old_u - K) / K;
                float dot_without = dot_co - old_c * rotated[i];
                float norm_without = norm_c_sq - old_c * old_c;
                int best_u = old_u;
                float best_dot = dot_co, best_norm = norm_c_sq;
                for (int u = 0; u <= K_INT; ++u) {
                    if (u == old_u) continue;
                    float c = (2.0f * u - K) / K;
                    float new_dot = dot_without + c * rotated[i];
                    float new_norm = norm_without + c * c;
                    if (new_dot * new_dot * best_norm > best_dot * best_dot * new_norm) {
                        best_u = u;
                        best_dot = new_dot;
                        best_norm = new_norm;
                    }
                }
                if (best_u != old_u) {
                    float new_c = (2.0f * best_u - K) / K;
                    dot_co = dot_without + new_c * rotated[i];
                    norm_c_sq = norm_without + new_c * new_c;
                    pr_codes[i] = best_u;
                    changed = true;
                }
            }
            if (!changed) break;
        }

        // Write codes and compute aux data
        float ip_qo = 0.0f, ip_cp = 0.0f;
        for (size_t i = 0; i < pd; ++i) {
            result.code.set_value(i, static_cast<uint8_t>(pr_codes[i]));
            float c = (2.0f * pr_codes[i] - K) / K;
            ip_qo += c * rotated[i];
            ip_cp += c * rotated_parent[i];
        }
        result.aux.ip_quantized_original = ip_qo * this->inv_sqrt_d_;
        result.aux.ip_code_parent = ip_cp * this->inv_sqrt_d_;
        return result;
    }

    // CAQ (Code Adjustment Quantization) encoding via coordinate descent.
    // For each dimension, greedily selects the codeword maximizing cos(quantized, original).
    // O(D * 2^B) per vector vs O(2^B * D * log D) for the old critical-point search.
    CodeType encode_impl(const float* vec, float* buf, float* centered_buf) const {
        CodeType code;
        code.clear();

        float norm_sq = 0.0f;
        for (size_t i = 0; i < this->dim_; ++i) {
            float v = vec[i] - this->centroid_[i];
            centered_buf[i] = v;
            norm_sq += v * v;
        }
        float norm = std::sqrt(norm_sq);
        code.dist_to_centroid = norm;

        if (norm < adaptive_defaults::norm_epsilon(D)) {
            code.ip_quantized_original = 0.0f;
            code.msb_popcount = 0;
            return code;
        }

        float inv_norm = 1.0f / norm;
        for (size_t i = 0; i < this->dim_; ++i) centered_buf[i] *= inv_norm;

        this->rotation_.apply_copy(centered_buf, buf);
        for (size_t i = 0; i < this->padded_dim_; ++i) buf[i] *= this->norm_factor_;

        const size_t pd = this->padded_dim_;

        // Initialize codes to nearest-rounding assignment
        thread_local std::vector<int> codes_arr;
        codes_arr.resize(pd);

        float dot_co = 0.0f;   // <code_vector, original>
        float norm_c_sq = 0.0f; // ||code_vector||^2

        // Delta-based initialization: find min/max of buf, scale to [0, K]
        float buf_min = buf[0], buf_max = buf[0];
        for (size_t i = 1; i < pd; ++i) {
            if (buf[i] < buf_min) buf_min = buf[i];
            if (buf[i] > buf_max) buf_max = buf[i];
        }
        float delta = (buf_max - buf_min) / K;
        if (delta < adaptive_defaults::coordinate_epsilon(D))
            delta = adaptive_defaults::coordinate_epsilon(D);
        float inv_delta = 1.0f / delta;

        for (size_t i = 0; i < pd; ++i) {
            float val = (buf[i] - buf_min) * inv_delta;
            int u = static_cast<int>(val + 0.5f);
            if (u < 0) u = 0;
            if (u > K_INT) u = K_INT;
            codes_arr[i] = u;
            float c = (2.0f * u - K) / K;
            dot_co += c * buf[i];
            norm_c_sq += c * c;
        }

        // Coordinate descent: sweep each dimension, try all 2^B codewords,
        // keep the one that maximizes cos^2 = dot_co^2 / norm_c_sq
        // (avoiding sqrt by comparing dot^2 * other_norm vs other_dot^2 * norm)
        constexpr size_t max_iters = adaptive_defaults::caq_max_iterations();
        for (size_t iter = 0; iter < max_iters; ++iter) {
            bool changed = false;
            for (size_t i = 0; i < pd; ++i) {
                int old_u = codes_arr[i];
                float old_c = (2.0f * old_u - K) / K;

                // Remove contribution of dimension i
                float dot_without = dot_co - old_c * buf[i];
                float norm_without = norm_c_sq - old_c * old_c;

                int best_u = old_u;
                // Current objective: dot_co^2 * 1 (comparing against norm_c_sq)
                // We want: (dot_without + new_c * buf[i])^2 / (norm_without + new_c^2) > best
                // Avoid division: compare cross-multiplied
                float best_dot = dot_co;
                float best_norm = norm_c_sq;

                for (int u = 0; u <= K_INT; ++u) {
                    if (u == old_u) continue;
                    float c = (2.0f * u - K) / K;
                    float new_dot = dot_without + c * buf[i];
                    float new_norm = norm_without + c * c;

                    // Compare new_dot^2 * best_norm vs best_dot^2 * new_norm
                    if (new_dot * new_dot * best_norm > best_dot * best_dot * new_norm) {
                        best_u = u;
                        best_dot = new_dot;
                        best_norm = new_norm;
                    }
                }

                if (best_u != old_u) {
                    float new_c = (2.0f * best_u - K) / K;
                    dot_co = dot_without + new_c * buf[i];
                    norm_c_sq = norm_without + new_c * new_c;
                    codes_arr[i] = best_u;
                    changed = true;
                }
            }
            if (!changed) break;
        }

        // Write final codes
        float ip_qo = 0.0f;
        for (size_t i = 0; i < pd; ++i) {
            code.codes.set_value(i, static_cast<uint8_t>(codes_arr[i]));
            float c = (2.0f * codes_arr[i] - K) / K;
            ip_qo += c * buf[i];
        }

        code.ip_quantized_original = ip_qo * this->inv_sqrt_d_;
        code.msb_popcount = static_cast<uint16_t>(code.codes.msb_popcount());
        return code;
    }
};

}  // namespace cphnsw
