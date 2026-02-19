#pragma once

#include "constants.hpp"
#include <cmath>
#include <cstddef>
#include <algorithm>

namespace cphnsw {

// constexpr integer square root (C++17 compatible)
constexpr size_t isqrt(size_t n) {
    if (n < 2) return n;
    size_t x = n, y = (x + 1) / 2;
    while (y < x) { x = y; y = (x + n / x) / 2; }
    return x;
}

// Graph topology statistics — populated during NNDescent in optimize_graph_adaptive()
struct GraphStats {
    float avg_degree;
    float alpha;
    float tau;
    float alpha_max;
};

// Metadata-derived index parameters — computed once at finalize() time
struct IndexProfile {
    size_t n = 0;
    size_t D = 0;
    size_t R = 0;
    size_t bits = 0;

    // Derived parameters (populated by derive())
    size_t evt_min_tail;
    size_t min_calib_samples;
    int slack_levels;

    // Graph topology (populated after NNDescent)
    GraphStats graph_stats;

    void derive(size_t n_, size_t D_, size_t R_, size_t bits_) {
        n = n_; D = D_; R = R_; bits = bits_;

        // CLT-based: need sqrt(n) tail samples for stable GPD estimation
        evt_min_tail = std::max(size_t(64),
            static_cast<size_t>(std::sqrt(static_cast<double>(n))));

        // Sub-linear calibration sample count
        min_calib_samples = std::clamp(
            static_cast<size_t>(10.0 * std::sqrt(static_cast<double>(n))),
            size_t(200), n);

        // Slack levels adaptive to dataset size
        float log_n = std::log2(static_cast<float>(std::max(n, size_t(64))));
        slack_levels = std::clamp(
            static_cast<int>(std::ceil(std::log2(std::max(10.0f * log_n, 4.0f)))),
            4, constants::kMaxSlackArray);
    }
};

namespace adaptive_defaults {

    // Upper layer degree: continuous function of D via constexpr isqrt
    constexpr size_t upper_layer_degree(size_t R, size_t D) {
        size_t base = R / 2;
        size_t bonus = isqrt(D) / 4;
        size_t cap = R / 4;
        bonus = (bonus < cap) ? bonus : cap;
        return base + bonus;
    }

    // Threading heuristic — unchanged
    inline size_t omp_chunk_size(size_t n, size_t num_threads) {
        if (num_threads == 0) num_threads = 1;
        size_t chunk = n / (num_threads * constants::kOmpChunkDiv);
        return std::clamp(chunk, constants::kOmpChunkMin, constants::kOmpChunkMax);
    }

    // Simplified visitation headroom
    inline size_t visitation_headroom(size_t n) {
        return std::clamp(n / 4, size_t(256), n);
    }

}
}
