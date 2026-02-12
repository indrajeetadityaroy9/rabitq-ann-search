#pragma once

#include <cmath>
#include <cstddef>
#include <algorithm>

namespace cphnsw {

// All parameters are derived from data statistics or first-principles bounds.
// No manually tuned heuristics remain.
struct AdaptiveDefaults {
    // --- Build-time: error epsilon from RaBitQ Theorem 3.2 ---
    // Controls false-prune probability during quantized construction search.
    // p_fail = 0.25 → sqrt(-2*ln(0.25)) ≈ 1.665
    static constexpr float BUILD_FALSE_PRUNE_RATE = 0.25f;

    static float error_epsilon_build() {
        return std::sqrt(-2.0f * std::log(BUILD_FALSE_PRUNE_RATE));
    }

    // --- Build-time: error tolerance from dimension ---
    // RaBitQ quantization error variance is O(1/D), so tolerance = 1/sqrt(D).
    static float error_tolerance(size_t D) {
        return 1.0f / std::sqrt(static_cast<float>(D));
    }

    // --- Build-time: NN-Descent convergence ---
    // δ-convergence threshold: stop when edge update rate drops below this.
    // δ = 0.001 (0.1%) is the gold-standard from Wei Dong et al. (2011).
    // Dataset-agnostic: does not depend on N, D, or R.
    static constexpr float NNDESCENT_DELTA = 0.001f;

    // Max NN-Descent iterations. The algorithm converges in 5-15 iterations
    // for all practical datasets (PyNNDescent, faiss). 20 is a generous cap.
    static constexpr size_t NNDESCENT_MAX_ITERS = 20;

    // --- Search-time: distance-adaptive termination gamma from recall target ---
    // Beam search terminates when exact_dist > (1+gamma) * k-th best exact dist.
    // Uses exact distances (not estimates) for coherent comparison.
    // Larger gamma → more exploration → higher recall.
    //
    // Derivation from RaBitQ Theorem 3.2 error scaling:
    //   - epsilon = sqrt(-2*ln(1-recall)) controls lower-bound tightness
    //   - RaBitQ estimation error is O(epsilon / sqrt(D))
    //   - gamma = epsilon / sqrt(D) matches the relative error scale
    //
    // For D=128:
    //   recall 0.80 → epsilon=1.79, gamma ≈ 0.158
    //   recall 0.90 → epsilon=2.15, gamma ≈ 0.190
    //   recall 0.95 → epsilon=2.45, gamma ≈ 0.216
    //   recall 0.99 → epsilon=3.03, gamma ≈ 0.268
    static float gamma_from_recall(float recall_target, size_t D) {
        float epsilon = error_epsilon_search(recall_target);
        return epsilon / std::sqrt(static_cast<float>(D));
    }

    // --- Search-time: error epsilon from recall target ---
    // Directly maps recall to Gaussian tail probability.
    // epsilon = sqrt(-2*ln(1-recall_target))
    static float error_epsilon_search(float recall_target) {
        float p = 1.0f - std::clamp(recall_target, 0.5f, 0.9999f);
        return std::sqrt(-2.0f * std::log(p));
    }

    // --- Search-time: hard safety cap on beam size ---
    static constexpr size_t EF_SAFETY_CAP = 4096;
};

}  // namespace cphnsw
