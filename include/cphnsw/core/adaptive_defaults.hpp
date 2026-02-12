#pragma once

#include <cmath>
#include <cstddef>
#include <algorithm>

namespace cphnsw {

struct AdaptiveDefaults {
    static constexpr float BUILD_FALSE_PRUNE_RATE = 0.25f;

    static float error_epsilon_build() {
        return std::sqrt(-2.0f * std::log(BUILD_FALSE_PRUNE_RATE));
    }

    static float error_tolerance(size_t D) {
        return 1.0f / std::sqrt(static_cast<float>(D));
    }

    static constexpr float NNDESCENT_DELTA = 0.001f;

    static constexpr size_t NNDESCENT_MAX_ITERS = 20;

    static float gamma_from_recall(float recall_target, size_t D) {
        float epsilon = error_epsilon_search(recall_target);
        return epsilon / std::sqrt(static_cast<float>(D));
    }

    static float error_epsilon_search(float recall_target) {
        float p = 1.0f - std::clamp(recall_target, 0.5f, 0.9999f);
        return std::sqrt(-2.0f * std::log(p));
    }

    static constexpr size_t EF_SAFETY_CAP = 4096;
};

}  // namespace cphnsw
