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
            code.msb_popcount = 0;
            return code;
        }

        float inv_norm = 1.0f / norm;
        for (size_t i = 0; i < this->dim_; ++i) centered_buf[i] *= inv_norm;

        this->rotation_.apply_copy(centered_buf, buf);
        for (size_t i = 0; i < this->padded_dim_; ++i) buf[i] *= this->norm_factor_;


        constexpr float midpoint = K * 0.5f;
        constexpr size_t NUM_LEVELS = static_cast<size_t>(1) << BitWidth;
        const size_t pd = this->padded_dim_;


        float best_t = 0.0f;
        float best_cosine = -std::numeric_limits<float>::max();

        if constexpr (adaptive_defaults::use_critical_point_search(D, BitWidth)) {
            thread_local std::vector<float> crits;
            crits.clear();
            crits.reserve(pd * NUM_LEVELS);

            for (size_t i = 0; i < pd; ++i) {
                float xi = buf[i];
                if (std::abs(xi) < adaptive_defaults::coordinate_epsilon(D)) continue;
                for (size_t b = 0; b < NUM_LEVELS; ++b) {
                    float boundary = static_cast<float>(b) + 0.5f;
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
                if (norm_c_sq < adaptive_defaults::division_epsilon()) return -std::numeric_limits<float>::max();
                return dot / std::sqrt(norm_c_sq);
            };

            {
                float cosine = evaluate_cosine(adaptive_defaults::nbit_initial_scale());
                if (cosine > best_cosine) {
                    best_cosine = cosine;
                    best_t = adaptive_defaults::nbit_initial_scale();
                }
            }

            float prev = 0.0f;
            for (size_t k = 0; k < crits.size(); ++k) {
                float cur = crits[k];
                if (cur - prev < adaptive_defaults::coordinate_epsilon(D)) { prev = cur; continue; }
                float t_mid = 0.5f * (prev + cur);
                float cosine = evaluate_cosine(t_mid);
                if (cosine > best_cosine) {
                    best_cosine = cosine;
                    best_t = t_mid;
                }
                prev = cur;
            }
            if (!crits.empty()) {
                float t_last = crits.back() * adaptive_defaults::nbit_overshoot_relative() + adaptive_defaults::nbit_overshoot_absolute();
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

            float t_max = (max_abs > adaptive_defaults::coordinate_epsilon(D)) ? (K + 0.5f) / max_abs : 1.0f;
            constexpr size_t NUM_GRID = adaptive_defaults::nbit_grid_resolution(D, BitWidth);

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
                if (norm_c_sq < adaptive_defaults::division_epsilon()) return -std::numeric_limits<float>::max();
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
            float lo = std::max(adaptive_defaults::nbit_scale_floor(), best_t - step);
            float hi = std::min(t_max, best_t + step);
            constexpr float phi = 0.6180339887f;

            constexpr size_t gs_iters = adaptive_defaults::golden_section_iters(
                adaptive_defaults::nbit_grid_resolution(D, BitWidth));
            for (size_t iter = 0; iter < gs_iters; ++iter) {
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
