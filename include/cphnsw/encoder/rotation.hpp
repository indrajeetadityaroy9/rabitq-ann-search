#pragma once

#include "../core/memory.hpp"
#include "transform/fht.hpp"
#include <vector>
#include <array>
#include <random>
#include <memory>
#include <cstring>

namespace cphnsw {

// Implements: Ψ(x) = H D3 H D2 H D1 x
// H: Walsh-Hadamard matrix, D: Random diagonal matrices (±1)
class RandomHadamardRotation {
public:
    static constexpr size_t NUM_LAYERS = 3;

    RandomHadamardRotation(size_t dim, uint64_t seed)
        : original_dim_(dim)
        , padded_dim_(next_power_of_two(dim)) {

        generate_signs(seed);
    }

    void apply(float* x) const {
        apply_diagonal_simd(x, signs_float_[0].data());
        fht(x, padded_dim_);

        apply_diagonal_simd(x, signs_float_[1].data());
        fht(x, padded_dim_);

        apply_diagonal_simd(x, signs_float_[2].data());
        fht(x, padded_dim_);
    }

    void apply_copy(const float* input, float* output) const {
        std::memcpy(output, input, original_dim_ * sizeof(float));
        std::memset(output + original_dim_, 0,
                    (padded_dim_ - original_dim_) * sizeof(float));

        apply(output);
    }

    size_t original_dim() const { return original_dim_; }
    size_t padded_dim() const { return padded_dim_; }

    const std::array<std::vector<int8_t>, NUM_LAYERS>& get_signs() const {
        return signs_;
    }

private:
    size_t original_dim_;
    size_t padded_dim_;
    std::array<std::vector<int8_t>, NUM_LAYERS> signs_;
    std::array<AlignedVector<float>, NUM_LAYERS> signs_float_;  // ±1.0f for SIMD diagonal

    static size_t next_power_of_two(size_t n) {
        size_t p = 1;
        while (p < n) p *= 2;
        return p;
    }

    void generate_signs(uint64_t seed) {
        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<int> coin(0, 1);

        for (size_t layer = 0; layer < NUM_LAYERS; ++layer) {
            signs_[layer].resize(padded_dim_);
            signs_float_[layer].resize(padded_dim_);
            for (size_t i = 0; i < padded_dim_; ++i) {
                int8_t s = coin(rng) ? 1 : -1;
                signs_[layer][i] = s;
                signs_float_[layer][i] = static_cast<float>(s);
            }
        }
    }

    void apply_diagonal_simd(float* x, const float* signs_f) const {
        size_t i = 0;
#ifdef __AVX2__
        for (; i + 8 <= padded_dim_; i += 8) {
            __m256 vx = _mm256_loadu_ps(x + i);
            __m256 vs = _mm256_load_ps(signs_f + i);
            _mm256_storeu_ps(x + i, _mm256_mul_ps(vx, vs));
        }
#endif
        for (; i < padded_dim_; ++i) {
            x[i] *= signs_f[i];
        }
    }
};

}  // namespace cphnsw