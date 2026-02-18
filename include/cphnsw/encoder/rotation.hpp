#pragma once

#include "../core/memory.hpp"
#include "../core/constants.hpp"
#include "../core/util.hpp"
#include "transform/fht.hpp"
#include <array>
#include <immintrin.h>
#include <random>
#include <cstring>

namespace cphnsw {

class RandomHadamardRotation {
public:
    static constexpr size_t NUM_LAYERS = 3;

    explicit RandomHadamardRotation(size_t dim, uint64_t seed = constants::kDefaultRotationSeed)
        : original_dim_(dim)
        , padded_dim_(next_power_of_two(dim))
        , seed_(seed) {

        generate_signs();
    }

    uint64_t seed() const { return seed_; }

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

    size_t padded_dim() const { return padded_dim_; }

private:
    size_t original_dim_;
    size_t padded_dim_;
    uint64_t seed_;
    std::array<AlignedVector<float>, NUM_LAYERS> signs_float_;

    void generate_signs() {
        std::mt19937_64 rng(seed_);
        std::uniform_int_distribution<int> coin(0, 1);

        for (size_t layer = 0; layer < NUM_LAYERS; ++layer) {
            signs_float_[layer].resize(padded_dim_);
            for (size_t i = 0; i < padded_dim_; ++i) {
                signs_float_[layer][i] = coin(rng) ? 1.0f : -1.0f;
            }
        }
    }

    void apply_diagonal_simd(float* x, const float* signs_f) const {
        size_t i = 0;
        for (; i + 8 <= padded_dim_; i += 8) {
            __m256 vx = _mm256_loadu_ps(x + i);
            __m256 vs = _mm256_load_ps(signs_f + i);
            _mm256_storeu_ps(x + i, _mm256_mul_ps(vx, vs));
        }
        for (; i < padded_dim_; ++i) {
            x[i] *= signs_f[i];
        }
    }
};

}  // namespace cphnsw
