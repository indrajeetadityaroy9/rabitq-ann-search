#pragma once

#include "adaptive_defaults.hpp"
#include "memory.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>

namespace cphnsw {

// Per-query adaptive gamma estimation using CLT-based difficulty estimation (Ada-ef).
// Computes offline per-dimension statistics of the rotated dataset, then at search time
// estimates query difficulty from the entry point neighborhood and maps it to a gamma value.
class AdaptiveGammaEstimator {
public:
    static constexpr size_t NUM_DIFFICULTY_BINS = 20;
    static constexpr size_t NUM_RECALL_LEVELS = 10;

    static constexpr float RECALL_LEVELS[NUM_RECALL_LEVELS] = {
        0.50f, 0.60f, 0.70f, 0.80f, 0.85f, 0.90f, 0.93f, 0.95f, 0.97f, 0.99f
    };

    AdaptiveGammaEstimator() = default;

    // Called during finalize() with rotated, norm-factored sample vectors.
    void compute_statistics(const float* rotated_vectors, size_t num_vecs, size_t padded_dim) {
        padded_dim_ = padded_dim;
        mu_.assign(padded_dim, 0.0f);
        var_.assign(padded_dim, 0.0f);

        float inv_n = 1.0f / static_cast<float>(num_vecs);

        for (size_t i = 0; i < num_vecs; ++i) {
            const float* v = rotated_vectors + i * padded_dim;
            for (size_t d = 0; d < padded_dim; ++d) {
                mu_[d] += v[d];
            }
        }
        for (size_t d = 0; d < padded_dim; ++d) mu_[d] *= inv_n;

        for (size_t i = 0; i < num_vecs; ++i) {
            const float* v = rotated_vectors + i * padded_dim;
            for (size_t d = 0; d < padded_dim; ++d) {
                float diff = v[d] - mu_[d];
                var_[d] += diff * diff;
            }
        }
        for (size_t d = 0; d < padded_dim; ++d) var_[d] *= inv_n;

        // Build gamma lookup table: difficulty -> gamma for each recall level
        // Higher difficulty (harder queries) need higher gamma (more exploration)
        for (size_t db = 0; db < NUM_DIFFICULTY_BINS; ++db) {
            float difficulty = static_cast<float>(db + 1) / static_cast<float>(NUM_DIFFICULTY_BINS);
            for (size_t rl = 0; rl < NUM_RECALL_LEVELS; ++rl) {
                float base_gamma = -std::log(1.0f - RECALL_LEVELS[rl]);
                // Scale: easy queries (low difficulty) use less gamma, hard queries use more
                float scale = 0.5f + 1.5f * difficulty;
                gamma_table_[db][rl] = base_gamma * scale;
            }
        }

        initialized_ = true;
    }

    // Called per-query during search. distance_samples are distances from entry point neighborhood.
    float estimate_gamma(
        const float* query_rotated,
        const float* distance_samples,
        size_t num_samples,
        float recall_target) const
    {
        if (!initialized_ || num_samples < 4) {
            float p = 1.0f - std::clamp(recall_target, 0.5f, 0.9999f);
            return -std::log(p);
        }

        // Estimate query difficulty via CLT: compute expected IP distribution parameters
        float mu_ip = 0.0f;
        float var_ip = 0.0f;
        for (size_t d = 0; d < padded_dim_; ++d) {
            mu_ip += query_rotated[d] * mu_[d];
            var_ip += query_rotated[d] * query_rotated[d] * var_[d];
        }
        float sigma_ip = std::sqrt(var_ip);

        if (sigma_ip < adaptive_defaults::division_epsilon()) {
            float p = 1.0f - std::clamp(recall_target, 0.5f, 0.9999f);
            return -std::log(p);
        }

        // Estimate difficulty: what fraction of sampled distances are "surprisingly close"
        // Use median distance as reference, count fraction below median/2
        std::vector<float> sorted_dists(distance_samples, distance_samples + num_samples);
        std::sort(sorted_dists.begin(), sorted_dists.end());
        float median_dist = sorted_dists[num_samples / 2];
        float close_threshold = median_dist * 0.5f;

        size_t close_count = 0;
        for (size_t i = 0; i < num_samples; ++i) {
            if (distance_samples[i] < close_threshold) close_count++;
        }

        float difficulty = static_cast<float>(close_count) / static_cast<float>(num_samples);
        // Clamp to bin range
        size_t db = static_cast<size_t>(difficulty * NUM_DIFFICULTY_BINS);
        if (db >= NUM_DIFFICULTY_BINS) db = NUM_DIFFICULTY_BINS - 1;

        size_t rl = find_recall_index(recall_target);
        return gamma_table_[db][rl];
    }

    bool is_initialized() const { return initialized_; }

private:
    static size_t find_recall_index(float recall_target) {
        for (size_t i = NUM_RECALL_LEVELS; i > 0; --i) {
            if (recall_target >= RECALL_LEVELS[i - 1]) return i - 1;
        }
        return 0;
    }

    std::vector<float> mu_;
    std::vector<float> var_;
    size_t padded_dim_ = 0;
    float gamma_table_[NUM_DIFFICULTY_BINS][NUM_RECALL_LEVELS] = {};
    bool initialized_ = false;
};

}  // namespace cphnsw
