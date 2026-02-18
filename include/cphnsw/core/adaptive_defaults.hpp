#pragma once

#include "constants.hpp"
#include <cmath>
#include <cstddef>
#include <algorithm>

namespace cphnsw {
namespace adaptive_defaults {

    inline float error_tolerance(size_t D) {
        return 1.0f / std::sqrt(static_cast<float>(D));
    }

    inline float nndescent_delta(size_t n, size_t R) {
        float base = 1.0f / (std::log2(static_cast<float>(std::max(n, size_t(64)))) * static_cast<float>(R));
        return std::clamp(base, constants::kNndescentDeltaMin, constants::kNndescentDeltaMax);
    }

    inline size_t nndescent_max_iters(size_t n) {
        size_t iters = static_cast<size_t>(std::log2(static_cast<float>(std::max(n, size_t(64)))));
        return std::clamp(iters, constants::kNndescentIterMin, constants::kNndescentIterMax);
    }

    inline size_t alpha_sample_size(size_t n) {
        size_t s = static_cast<size_t>(std::sqrt(static_cast<double>(n)));
        return std::clamp(s, constants::kAlphaSampleMin, constants::kAlphaSampleMax);
    }

    inline size_t alpha_inter_limit(size_t R) {
        size_t lim = static_cast<size_t>(2.0 * std::sqrt(static_cast<double>(R)));
        return std::clamp(lim, constants::kAlphaInterMin, R);
    }

    inline float alpha_default(size_t D) {
        float a = 1.0f + constants::kAlphaScaleCoeff * std::log2(static_cast<float>(D)) / constants::kAlphaScaleDenom;
        return std::clamp(a, constants::kAlphaDefaultMin, constants::kAlphaDefaultMax);
    }

    inline size_t random_init_pool(size_t R, size_t n) {
        size_t pool = static_cast<size_t>(static_cast<double>(R) * (1.0 + std::log(static_cast<double>(R))));
        return std::min(pool, n - 1);
    }

    inline size_t random_init_attempts(size_t R, size_t n) {
        size_t pool = random_init_pool(R, n);
        return std::min(pool + pool / 2, n);
    }

    inline size_t omp_chunk_size(size_t n, size_t num_threads) {
        if (num_threads == 0) num_threads = 1;
        size_t chunk = n / (num_threads * constants::kOmpChunkDiv);
        return std::clamp(chunk, constants::kOmpChunkMin, constants::kOmpChunkMax);
    }

    inline size_t ef_cap(size_t n, size_t k, float slack_factor) {
        float log_n = std::log2(static_cast<float>(std::max(n, size_t(64))));
        size_t cap = static_cast<size_t>(static_cast<float>(k) * slack_factor * log_n * constants::kEfCapLogScale);
        return std::clamp(cap, k * constants::kEfMinMultiplier, constants::kEfMaxCap);
    }

    inline size_t visitation_headroom(size_t n) {
        size_t headroom = std::max(constants::kVisitHeadroomMin, n / constants::kVisitHeadroomDiv);
        return std::min(headroom, constants::kVisitHeadroomMax);
    }

    inline size_t upper_layer_ef(size_t R, int level) {
        float ef = static_cast<float>(R) * std::pow(constants::kUpperEfGrowth, static_cast<float>(level - 1));
        return std::clamp(static_cast<size_t>(ef), R, R * constants::kUpperEfMaxMult);
    }

    constexpr size_t upper_layer_degree(size_t R, size_t D) {
        size_t base = R / 2;
        size_t bonus = (D >= constants::kUpperBonusDimThresh) ? (R / constants::kUpperBonusDivisor) : 0;
        return base + bonus;
    }

    inline size_t caq_max_iterations(size_t D) {
        size_t log2d = 0;
        size_t tmp = D;
        while (tmp > 1) { tmp >>= 1; ++log2d; }
        size_t iters = (log2d + constants::kCaqIterLogDiv - 1) / constants::kCaqIterLogDiv;
        return std::max(constants::kCaqIterMin, std::min(constants::kCaqIterMax, iters));
    }

}  // namespace adaptive_defaults
}  // namespace cphnsw
