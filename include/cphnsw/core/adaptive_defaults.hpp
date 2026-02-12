#pragma once

#include <cmath>
#include <cstddef>
#include <algorithm>

namespace cphnsw {

struct AdaptiveDefaults {
    // --- Build-time constants ---
    static constexpr float ALPHA = 1.0f;            // Pass 1: standard Vamana threshold
    static constexpr float ALPHA_PASS2 = 1.2f;      // Pass 2: literature standard for long-range edges
    static constexpr float ERROR_EPSILON_BUILD = 1.665f;  // p_fail=0.25 -> sqrt(-2*ln(0.25))

    static size_t ef_construction(size_t N, size_t R) {
        size_t ef = static_cast<size_t>(std::ceil(2.0 * std::sqrt(static_cast<double>(N))));
        return std::clamp(ef, R, static_cast<size_t>(200));
    }

    static float error_tolerance(size_t D) {
        return 1.0f / std::sqrt(static_cast<float>(D));
    }

    // --- Search-time derivations ---
    static float error_epsilon_search(float recall_target) {
        // Map recall target to failure probability, then to epsilon via Gaussian tail bound.
        // Lower epsilon = tighter bounds = more candidates visited = better recall.
        float p = std::clamp(0.5f * (1.0f - recall_target), 0.001f, 0.5f);
        return std::sqrt(-2.0f * std::log(p));
    }

    static size_t ef_search(size_t k, float recall_target) {
        float difficulty = -std::log(1.0f - std::clamp(recall_target, 0.5f, 0.9999f));
        size_t ef = static_cast<size_t>(std::ceil(k * (difficulty + 3.0)));
        return std::max(ef, k);
    }
};

}  // namespace cphnsw
