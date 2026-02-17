#pragma once

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
        return std::clamp(base, 0.0001f, 0.01f);
    }

    inline size_t nndescent_max_iters(size_t n) {
        size_t iters = static_cast<size_t>(std::log2(static_cast<float>(std::max(n, size_t(64)))));
        return std::clamp(iters, size_t(8), size_t(40));
    }

    inline size_t alpha_sample_size(size_t n) {
        size_t s = static_cast<size_t>(std::sqrt(static_cast<double>(n)));
        return std::clamp(s, size_t(64), size_t(10000));
    }

    inline size_t alpha_inter_limit(size_t R) {
        size_t lim = static_cast<size_t>(2.0 * std::sqrt(static_cast<double>(R)));
        return std::clamp(lim, size_t(4), R);
    }

    constexpr size_t alpha_percentile_divisor() { return 4; }

    inline float alpha_default(size_t D) {
        float a = 1.0f + 0.1f * std::log2(static_cast<float>(D)) / 5.0f;
        return std::clamp(a, 1.1f, 1.5f);
    }

    constexpr float alpha_ceiling() { return 2.0f; }
    constexpr float alpha_max_cap() { return 2.5f; }
    constexpr float alpha_floor_threshold() { return 1.02f; }
    constexpr float tau_scaling_factor() { return 0.5f; }

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
        size_t chunk = n / (num_threads * 16);
        return std::clamp(chunk, size_t(16), size_t(1024));
    }

    inline size_t ef_cap(size_t n, size_t k, float slack_factor) {
        float log_n = std::log2(static_cast<float>(std::max(n, size_t(64))));
        size_t cap = static_cast<size_t>(static_cast<float>(k) * slack_factor * log_n * 0.5f);
        return std::clamp(cap, size_t(k * 4), size_t(8192));
    }

    constexpr size_t default_k() { return 10; }

    inline float beam_trim_trigger_ratio() { return 2.0f; }
    inline float beam_trim_keep_ratio() { return 0.75f; }

    inline size_t visitation_headroom(size_t n) {
        size_t headroom = std::max(size_t(256), n / 4);
        return std::min(headroom, size_t(100000));
    }

    inline size_t upper_layer_ef(size_t R, int level) {
        float ef = static_cast<float>(R) * std::pow(1.5f, static_cast<float>(level - 1));
        return std::clamp(static_cast<size_t>(ef), R, R * 4);
    }

    constexpr size_t upper_layer_degree(size_t R, size_t D) {
        size_t base = R / 2;
        size_t bonus = (D >= 256) ? (R / 8) : 0;
        return base + bonus;
    }

    constexpr size_t caq_max_iterations() { return 3; }

    inline float norm_epsilon(size_t D) {
        return 1e-8f / static_cast<float>(D);
    }

    constexpr float division_epsilon() { return 1e-20f; }

    inline float coordinate_epsilon(size_t D) {
        return 1e-10f / std::sqrt(static_cast<float>(D));
    }

    constexpr float ip_quality_epsilon() { return 1e-10f; }

    constexpr size_t prefetch_line_cap() { return 16; }

}  // namespace adaptive_defaults
}  // namespace cphnsw
