#pragma once

#include "../core/codes.hpp"
#include "../core/memory.hpp"
#include "../core/adaptive_defaults.hpp"
#include "rotation.hpp"
#include <cmath>
#include <algorithm>
#include <vector>

#include <omp.h>

namespace cphnsw {

template <typename Derived, size_t D>
class RaBitQEncoderBase {
public:
    using QueryType = RaBitQQuery<D>;

    static constexpr size_t NUM_SUB_SEGMENTS = num_sub_segments<D>;

    explicit RaBitQEncoderBase(size_t dim,
        uint64_t rotation_seed = constants::kDefaultRotationSeed)
        : dim_(dim)
        , rotation_seed_(rotation_seed)
        , rotation_(dim, rotation_seed)
        , padded_dim_(rotation_.padded_dim())
        , centroid_(dim, 0.0f) {

        // norm_factor_ compensates for three consecutive unnormalized WHT
        // passes applied in RandomHadamardRotation::apply(). Each unnormalized
        // WHT call on a vector of length D scales the L2-norm by sqrt(D) (see
        // fht.hpp). Three passes accumulate: sqrt(D)^3 = D * sqrt(D).
        // The diagonal sign-flip layers are unitary (norm-preserving).
        // This single post-rotation multiply is equivalent to dividing by
        // sqrt(D) after each of the three individual passes.
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
    }

    template <typename CodeType>
    void encode_batch(const float* vecs, size_t num_vecs, CodeType* codes) {
        compute_centroid(vecs, num_vecs);

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
        thread_local AlignedVector<float> buf(padded_dim_);
        thread_local AlignedVector<uint8_t> q_bar_u(padded_dim_);
        buf.resize(padded_dim_);
        q_bar_u.resize(padded_dim_);
        return encode_query_raw_impl(vec, buf.data(), q_bar_u.data());
    }

    void rotate_raw_vector(const float* vec, float* out) const {
        rotation_.apply_copy(vec, out);
        for (size_t i = 0; i < padded_dim_; ++i) {
            out[i] *= norm_factor_;
        }
    }

    float compute_ip_cp(const BinaryCodeStorage<D>& code,
                                  const float* rotated_parent) const {
        float ip = 0.0f;
        for (size_t i = 0; i < padded_dim_; ++i) {
            float sign = code.get_bit(i) ? 1.0f : -1.0f;
            ip += sign * rotated_parent[i];
        }
        return ip * inv_sqrt_d_;
    }

    void build_lut(const float* buf, uint8_t* q_bar_u, QueryType& query) const {
        float vl = buf[0], vmax = buf[0];
        for (size_t i = 1; i < padded_dim_; ++i) {
            if (buf[i] < vl) vl = buf[i];
            if (buf[i] > vmax) vmax = buf[i];
        }

        float delta = (vmax - vl) / constants::kLutLevels;
        if (delta < constants::eps::kTiny) delta = constants::eps::kTiny;
        float inv_delta = 1.0f / delta;

        float sum_qu = 0.0f;
        for (size_t i = 0; i < padded_dim_; ++i) {
            float val = (buf[i] - vl) * inv_delta;
            int u = static_cast<int>(val + 0.5f);
            if (u < 0) u = 0;
            if (u > static_cast<int>(constants::kLutLevels)) u = static_cast<int>(constants::kLutLevels);
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

    VertexAuxData compute_neighbor_aux(
        const float* parent_vec,
        const float* neighbor_vec,
        const float* rotated_parent,
        BinaryCodeStorage<D>& out_code) const
    {
        VertexAuxData aux;
        out_code.clear();

        alignas(32) float diff_op[D];
        float nop_sq = 0.0f;
        for (size_t i = 0; i < dim_; ++i) {
            diff_op[i] = neighbor_vec[i] - parent_vec[i];
            nop_sq += diff_op[i] * diff_op[i];
        }
        for (size_t i = dim_; i < D; ++i) diff_op[i] = 0.0f;

        float nop = std::sqrt(nop_sq);
        aux.nop = nop;

        if (nop < constants::norm_epsilon(D)) {
            aux.ip_qo = 0.0f;
            aux.ip_cp = 0.0f;
            return aux;
        }

        float inv_nop = 1.0f / nop;
        for (size_t i = 0; i < D; ++i) diff_op[i] *= inv_nop;

        alignas(32) float rotated[D];
        rotation_.apply_copy(diff_op, rotated);
        for (size_t i = 0; i < padded_dim_; ++i) rotated[i] *= norm_factor_;

        float l1_norm = 0.0f;
        for (size_t i = 0; i < padded_dim_; ++i) {
            out_code.set_bit(i, rotated[i] >= 0.0f);
            l1_norm += std::abs(rotated[i]);
        }

        aux.ip_qo = l1_norm * inv_sqrt_d_;
        aux.ip_cp = compute_ip_cp(out_code, rotated_parent);

        return aux;
    }

    uint64_t get_rotation_seed() const { return rotation_seed_; }
    const std::vector<float>& get_centroid() const { return centroid_; }
    void set_centroid(std::vector<float> c) { centroid_ = std::move(c); }

protected:
    size_t dim_;
    uint64_t rotation_seed_;
    RandomHadamardRotation rotation_;
    size_t padded_dim_;
    float norm_factor_;
    float inv_sqrt_d_;
    std::vector<float> centroid_;

private:
    QueryType encode_query_raw_impl(const float* vec, float* buf,
                                     uint8_t* q_bar_u) const {
        QueryType query;

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
        code.nop = norm;

        if (norm < constants::norm_epsilon(D)) {
            code.ip_qo = 0.0f;
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

        code.ip_qo = l1_norm * this->inv_sqrt_d_;

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
    static constexpr int K_INT = (1 << BitWidth) - 1;
    static constexpr float K = static_cast<float>(K_INT);

    struct NbitNeighborResult {
        NbitCodeStorage<D, BitWidth> code;
        VertexAuxData aux;
    };

    using Base::Base;


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
        result.aux.nop = nop;

        if (nop < constants::norm_epsilon(D)) {
            result.aux.ip_qo = 0.0f;
            result.aux.ip_cp = 0.0f;
            return result;
        }

        float inv_nop = 1.0f / nop;
        for (size_t i = 0; i < D; ++i) diff[i] *= inv_nop;

        alignas(32) float rotated[D];
        this->rotation_.apply_copy(diff, rotated);
        for (size_t i = 0; i < this->padded_dim_; ++i) rotated[i] *= this->norm_factor_;

        float ip_cp = 0.0f;
        result.aux.ip_qo = caq_quantize(rotated, result.code,
                                         rotated_parent, &ip_cp);
        result.aux.ip_cp = ip_cp;
        return result;
    }


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
        code.nop = norm;

        if (norm < constants::norm_epsilon(D)) {
            code.ip_qo = 0.0f;
            return code;
        }

        float inv_norm = 1.0f / norm;
        for (size_t i = 0; i < this->dim_; ++i) centered_buf[i] *= inv_norm;

        this->rotation_.apply_copy(centered_buf, buf);
        for (size_t i = 0; i < this->padded_dim_; ++i) buf[i] *= this->norm_factor_;

        code.ip_qo = caq_quantize(buf, code.codes);
        return code;
    }

private:
    // CAQ (Cosine-Aligned Quantization): uniform initialization followed by
    // coordinate-wise refinement maximizing cosine similarity between the
    // N-bit quantized code and the rotated input vector.
    //
    // Returns ip_qo = <c_bar, rotated_buf> * inv_sqrt_d_, where
    // c_bar[i] = (2*u[i] - K) / K in [-1, 1].
    //
    // IMPORTANT: ip_qo is NOT the true cosine <c_bar/||c_bar||, rotated_buf>
    // because c_bar is not L2-normalized. The implicit ||c_bar|| factor
    // cancels in the distance estimator ratio ip_corrected/ip_qo, since
    // ip_corrected = <c_bar, query_lut_approx> * inv_sqrt_d_ - ip_cp is
    // computed with the same unnormalized c_bar via the LUT fastscan.
    //
    // If rotated_parent != nullptr, also computes ip_cp into *out_ip_cp.
    // ip_cp carries the same implicit ||c_bar|| and participates in the
    // same cancellation.
    float caq_quantize(const float* rotated_buf,
                       NbitCodeStorage<D, BitWidth>& out_code,
                       const float* rotated_parent = nullptr,
                       float* out_ip_cp = nullptr) const {
        const size_t pd = this->padded_dim_;

        float buf_min = rotated_buf[0], buf_max = rotated_buf[0];
        for (size_t i = 1; i < pd; ++i) {
            if (rotated_buf[i] < buf_min) buf_min = rotated_buf[i];
            if (rotated_buf[i] > buf_max) buf_max = rotated_buf[i];
        }
        float delta = (buf_max - buf_min) / K;
        if (delta < constants::coordinate_epsilon(D))
            delta = constants::coordinate_epsilon(D);
        float inv_delta = 1.0f / delta;

        thread_local std::vector<int> codes_arr;
        codes_arr.resize(pd);

        float dot_co = 0.0f, norm_c_sq = 0.0f;
        for (size_t i = 0; i < pd; ++i) {
            float val = (rotated_buf[i] - buf_min) * inv_delta;
            int u = static_cast<int>(val + 0.5f);
            if (u < 0) u = 0;
            if (u > K_INT) u = K_INT;
            codes_arr[i] = u;
            float c = (2.0f * u - K) / K;
            dot_co += c * rotated_buf[i];
            norm_c_sq += c * c;
        }

        constexpr size_t max_iters = 10;
        float prev_cos_sq = 0.0f;
        for (size_t iter = 0; iter < max_iters; ++iter) {
            bool changed = false;
            for (size_t i = 0; i < pd; ++i) {
                int old_u = codes_arr[i];
                float old_c = (2.0f * old_u - K) / K;
                float dot_without = dot_co - old_c * rotated_buf[i];
                float norm_without = norm_c_sq - old_c * old_c;
                int best_u = old_u;
                float best_dot = dot_co, best_norm = norm_c_sq;
                if constexpr (BitWidth >= 4) {
                    // SAQ ±1 refinement: O(2) per dimension
                    for (int delta : {-1, +1}) {
                        int u_try = old_u + delta;
                        if (u_try < 0 || u_try > K_INT) continue;
                        float c = (2.0f * u_try - K) / K;
                        float new_dot = dot_without + c * rotated_buf[i];
                        float new_norm = norm_without + c * c;
                        if (new_dot * new_dot * best_norm > best_dot * best_dot * new_norm) {
                            best_u = u_try;
                            best_dot = new_dot;
                            best_norm = new_norm;
                        }
                    }
                } else {
                    // Exhaustive: O(K+1) per dimension — optimal for small K
                    for (int u = 0; u <= K_INT; ++u) {
                        if (u == old_u) continue;
                        float c = (2.0f * u - K) / K;
                        float new_dot = dot_without + c * rotated_buf[i];
                        float new_norm = norm_without + c * c;
                        if (new_dot * new_dot * best_norm > best_dot * best_dot * new_norm) {
                            best_u = u;
                            best_dot = new_dot;
                            best_norm = new_norm;
                        }
                    }
                }
                if (best_u != old_u) {
                    float new_c = (2.0f * best_u - K) / K;
                    dot_co = dot_without + new_c * rotated_buf[i];
                    norm_c_sq = norm_without + new_c * new_c;
                    codes_arr[i] = best_u;
                    changed = true;
                }
            }
            if (!changed) break;
            float cos_sq = (norm_c_sq > 0.0f) ? (dot_co * dot_co / norm_c_sq) : 0.0f;
            if (iter > 0 && (cos_sq - prev_cos_sq) < constants::kCaqEarlyExitTol) break;
            prev_cos_sq = cos_sq;
        }

        float ip_qo = 0.0f;
        float ip_cp = 0.0f;
        for (size_t i = 0; i < pd; ++i) {
            out_code.set_value(i, static_cast<uint8_t>(codes_arr[i]));
            float c = (2.0f * codes_arr[i] - K) / K;
            ip_qo += c * rotated_buf[i];
            if (rotated_parent) ip_cp += c * rotated_parent[i];
        }
        if (out_ip_cp) *out_ip_cp = ip_cp * this->inv_sqrt_d_;
        // Unnormalized dot product scaled by inv_sqrt_d_. The omitted
        // ||c_bar|| normalization cancels in ip_corrected/ip_qo.
        return ip_qo * this->inv_sqrt_d_;
    }
};

}
