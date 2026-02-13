#pragma once

#include "../core/codes.hpp"
#include "../core/memory.hpp"
#include "rotation.hpp"
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

template <size_t D>
struct EncoderWorkspace {
    AlignedVector<float> buf;
    std::vector<float> centered;
    AlignedVector<uint8_t> q_bar_u;

    explicit EncoderWorkspace(size_t padded_dim, size_t dim)
        : buf(padded_dim), centered(dim), q_bar_u(padded_dim) {}
};

template <typename Derived, size_t D, typename RotationPolicy>
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
        // Normalization factor for RandomHadamardRotation
        norm_factor_ = 1.0f / (d_float * std::sqrt(d_float));

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

    auto encode(const float* vec) const {
        thread_local EncoderWorkspace<D> ws(padded_dim_, dim_);
        ws.buf.resize(padded_dim_);
        ws.centered.resize(dim_);
        return static_cast<const Derived*>(this)->encode_impl(vec, ws.buf.data(), ws.centered.data());
    }

    template <typename CodeType>
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
                codes[i] = static_cast<const Derived*>(this)->encode_impl(
                    vecs + i * dim_, buf.data(), centered.data());
            }
        }
#else
        AlignedVector<float> buf(padded_dim_);
        std::vector<float> centered(dim_);
        for (size_t i = 0; i < num_vecs; ++i) {
            codes[i] = static_cast<const Derived*>(this)->encode_impl(
                vecs + i * dim_, buf.data(), centered.data());
        }
#endif
    }

    QueryType encode_query(const float* vec) const {
        thread_local EncoderWorkspace<D> ws(padded_dim_, dim_);
        ws.buf.resize(padded_dim_);
        ws.centered.resize(dim_);
        ws.q_bar_u.resize(padded_dim_);
        return encode_query_impl(vec, ws.buf.data(), ws.centered.data(), ws.q_bar_u.data());
    }

    QueryType encode_query_impl(const float* vec, float* buf, float* centered_buf,
                                uint8_t* q_bar_u) const {
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
        query.error_epsilon = 0.0f;

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

        query.coeff_fastscan = 2.0f * delta * inv_sqrt_d_;
        query.coeff_popcount = 2.0f * vl * inv_sqrt_d_;
        query.coeff_constant = -(delta * inv_sqrt_d_) * sum_qu - vl * sqrt_d_;

        return query;
    }

    template <typename CodeType>
    VertexAuxData compute_neighbor_aux(
        const CodeType& neighbor_code,
        const float* parent_vec,
        const float* neighbor_vec,
        float parent_dist_centroid = 0.0f,
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
            aux.dist_to_centroid = nop;  // ||o - p||

            if (nop < 1e-10f) {
                aux.ip_quantized_original = 0.0f;
                aux.ip_xbar_Pinv_c = 0.0f;
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

            aux.ip_quantized_original = l1_norm * inv_sqrt_d_;  // ip_qo_p

            float no = neighbor_code.dist_to_centroid;
            float np = parent_dist_centroid;
            aux.ip_xbar_Pinv_c = no * no - np * np;

            return aux;
        }

        aux.dist_to_centroid = neighbor_code.dist_to_centroid;
        aux.ip_quantized_original = neighbor_code.ip_quantized_original;
        aux.ip_xbar_Pinv_c = 0.0f;
        return aux;
    }

    size_t dim() const { return dim_; }
    size_t padded_dim() const { return padded_dim_; }

protected:
    size_t dim_;
    RotationPolicy rotation_;
    size_t padded_dim_;
    float norm_factor_;
    float inv_sqrt_d_;
    float sqrt_d_;
    std::vector<float> centroid_;
    bool has_centroid_;
};


template <size_t D, typename RotationPolicy = RandomHadamardRotation>
class RaBitQEncoder : public RaBitQEncoderBase<RaBitQEncoder<D, RotationPolicy>, D, RotationPolicy> {
    using Base = RaBitQEncoderBase<RaBitQEncoder<D, RotationPolicy>, D, RotationPolicy>;
    friend Base;

public:
    using CodeType = RaBitQCode<D>;
    using QueryType = RaBitQQuery<D>;
    static constexpr size_t DIMS = D;
    static constexpr size_t NUM_SUB_SEGMENTS = (D + 3) / 4;

    using Base::Base;  // inherit constructors

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

        if (norm < 1e-10f) {
            code.ip_quantized_original = 0.0f;
            code.code_popcount = 0;
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
        code.code_popcount = static_cast<uint16_t>(code.signs.popcount());

        return code;
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
};


template <size_t D, size_t BitWidth, typename RotationPolicy = RandomHadamardRotation>
class NbitRaBitQEncoder : public RaBitQEncoderBase<NbitRaBitQEncoder<D, BitWidth, RotationPolicy>, D, RotationPolicy> {
    using Base = RaBitQEncoderBase<NbitRaBitQEncoder<D, BitWidth, RotationPolicy>, D, RotationPolicy>;
    friend Base;

public:
    using CodeType = NbitRaBitQCode<D, BitWidth>;
    using QueryType = RaBitQQuery<D>;
    static constexpr size_t DIMS = D;
    static constexpr size_t BIT_WIDTH = BitWidth;
    static constexpr size_t NUM_SUB_SEGMENTS = (D + 3) / 4;
    static constexpr int K_INT = (1 << BitWidth) - 1;
    static constexpr float K = static_cast<float>(K_INT);

    using Base::Base;  // inherit constructors

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

        if (norm < 1e-10f) {
            code.ip_quantized_original = 0.0f;
            code.msb_popcount = 0;
            return code;
        }

        float inv_norm = 1.0f / norm;
        for (size_t i = 0; i < this->dim_; ++i) centered_buf[i] *= inv_norm;

        this->rotation_.apply_copy(centered_buf, buf);
        for (size_t i = 0; i < this->padded_dim_; ++i) buf[i] *= this->norm_factor_;


        constexpr float midpoint = K * 0.5f;
        constexpr size_t NUM_LEVELS = static_cast<size_t>(1) << BitWidth;  // 2^B
        const size_t pd = this->padded_dim_;


        float best_t = 0.0f;
        float best_cosine = -std::numeric_limits<float>::max();

        if constexpr (BitWidth <= 3) {
            thread_local std::vector<float> crits;
            crits.clear();
            crits.reserve(pd * NUM_LEVELS);

            for (size_t i = 0; i < pd; ++i) {
                float xi = buf[i];
                if (std::abs(xi) < 1e-12f) continue;
                for (size_t b = 0; b < NUM_LEVELS; ++b) {
                    float boundary = static_cast<float>(b) + 0.5f;  // 0.5, 1.5, ..., K-0.5
                    float t_crit = (boundary - midpoint) / xi;
                    if (t_crit > 0.0f) {
                        crits.push_back(t_crit);
                    }
                }
            }

            std::sort(crits.begin(), crits.end());

            auto evaluate_cosine = [&](float t) -> float {
                float dot = 0.0f;
                float norm_c_sq = 0.0f;
                for (size_t i = 0; i < pd; ++i) {
                    float val = t * buf[i] + midpoint;
                    int u = static_cast<int>(val + 0.5f);
                    if (u < 0) u = 0;
                    if (u > K_INT) u = K_INT;
                    float c = (2.0f * u - K) / K;
                    dot += c * buf[i];
                    norm_c_sq += c * c;
                }
                if (norm_c_sq < 1e-20f) return -std::numeric_limits<float>::max();
                return dot / std::sqrt(norm_c_sq);
            };

            {
                float cosine = evaluate_cosine(1e-6f);
                if (cosine > best_cosine) {
                    best_cosine = cosine;
                    best_t = 1e-6f;
                }
            }

            float prev = 0.0f;
            for (size_t k = 0; k < crits.size(); ++k) {
                float cur = crits[k];
                if (cur - prev < 1e-10f) { prev = cur; continue; }
                float t_mid = 0.5f * (prev + cur);
                float cosine = evaluate_cosine(t_mid);
                if (cosine > best_cosine) {
                    best_cosine = cosine;
                    best_t = t_mid;
                }
                prev = cur;
            }
            if (!crits.empty()) {
                float t_last = crits.back() * 1.1f + 0.1f;
                float cosine = evaluate_cosine(t_last);
                if (cosine > best_cosine) {
                    best_cosine = cosine;
                    best_t = t_last;
                }
            }
        } else {
            float max_abs = 0.0f;
            for (size_t i = 0; i < pd; ++i) {
                float a = std::abs(buf[i]);
                if (a > max_abs) max_abs = a;
            }

            float t_max = (max_abs > 1e-12f) ? (K + 0.5f) / max_abs : 1.0f;
            constexpr size_t NUM_GRID = 256;

            auto evaluate_cosine_grid = [&](float t) -> float {
                float dot_val = 0.0f;
                float norm_c_sq = 0.0f;
                for (size_t i = 0; i < pd; ++i) {
                    float val = t * buf[i] + midpoint;
                    int u = static_cast<int>(val + 0.5f);
                    if (u < 0) u = 0;
                    if (u > K_INT) u = K_INT;
                    float c = (2.0f * u - K) / K;
                    dot_val += c * buf[i];
                    norm_c_sq += c * c;
                }
                if (norm_c_sq < 1e-20f) return -std::numeric_limits<float>::max();
                return dot_val / std::sqrt(norm_c_sq);
            };

            for (size_t g = 1; g <= NUM_GRID; ++g) {
                float t = t_max * static_cast<float>(g) / static_cast<float>(NUM_GRID);
                float cosine = evaluate_cosine_grid(t);
                if (cosine > best_cosine) {
                    best_cosine = cosine;
                    best_t = t;
                }
            }

            float step = t_max / static_cast<float>(NUM_GRID);
            float lo = std::max(1e-8f, best_t - step);
            float hi = std::min(t_max, best_t + step);
            constexpr float phi = 0.6180339887f;

            for (int iter = 0; iter < 20; ++iter) {
                float m1 = hi - phi * (hi - lo);
                float m2 = lo + phi * (hi - lo);
                if (evaluate_cosine_grid(m1) < evaluate_cosine_grid(m2))
                    lo = m1;
                else
                    hi = m2;
            }
            float refined_t = (lo + hi) * 0.5f;
            float refined_cosine = evaluate_cosine_grid(refined_t);
            if (refined_cosine > best_cosine) {
                best_cosine = refined_cosine;
                best_t = refined_t;
            }
        }

        float ip_qo = 0.0f;
        for (size_t i = 0; i < pd; ++i) {
            float val = best_t * buf[i] + midpoint;
            int u = static_cast<int>(val + 0.5f);
            if (u < 0) u = 0;
            if (u > K_INT) u = K_INT;
            code.codes.set_value(i, static_cast<uint8_t>(u));
            float c = (2.0f * u - K) / K;
            ip_qo += c * buf[i];
        }

        code.ip_quantized_original = ip_qo * this->inv_sqrt_d_;
        code.msb_popcount = static_cast<uint16_t>(code.codes.msb_popcount());
        return code;
    }
};

}  // namespace cphnsw
