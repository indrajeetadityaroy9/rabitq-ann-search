#pragma once

#include "../core/types.hpp"
#include "../core/adaptive_defaults.hpp"
#include <vector>
#include <algorithm>
#include <cmath>

namespace cphnsw {

struct NeighborCandidate {
    NodeId id;
    float distance;

    bool operator<(const NeighborCandidate& other) const {
        return distance < other.distance;
    }
};

// alpha-CNG pruning with adaptive local alpha and convergence radius tau.
template <typename DistanceFn, typename ErrorFn>
std::vector<NeighborCandidate> select_neighbors_alpha_cng(
    std::vector<NeighborCandidate> candidates,
    size_t R,
    DistanceFn distance_fn,
    ErrorFn error_fn,
    float alpha,
    float tau)
{
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) {
                  return a.id < b.id || (a.id == b.id && a.distance < b.distance);
              });
    candidates.erase(
        std::unique(candidates.begin(), candidates.end(),
                    [](const auto& a, const auto& b) { return a.id == b.id; }),
        candidates.end());

    std::sort(candidates.begin(), candidates.end());

    if (candidates.size() <= R) return candidates;

    float local_alpha = alpha * std::sqrt(
        static_cast<float>(candidates.size()) / static_cast<float>(R));
    local_alpha = std::clamp(local_alpha, 1.0f, adaptive_defaults::alpha_max_cap());

    std::vector<NeighborCandidate> selected;
    selected.reserve(R);
    std::vector<bool> used(candidates.size(), false);

    for (size_t i = 0; i < candidates.size() && selected.size() < R; ++i) {
        bool should_add = true;
        float err_candidate = error_fn(candidates[i].id);
        float dist_cq = candidates[i].distance;

        for (const auto& existing : selected) {
            float dist_ce = distance_fn(candidates[i].id, existing.id);
            float err_existing = error_fn(existing.id);
            float margin = err_candidate + err_existing;

            float threshold = local_alpha * dist_cq + margin - (local_alpha - 1.0f) * tau;
            if (dist_ce < threshold) {
                should_add = false;
                break;
            }
        }
        if (should_add) {
            selected.push_back(candidates[i]);
            used[i] = true;
        }
    }

    if (selected.size() < R) {
        std::vector<size_t> backfill_indices;
        for (size_t i = 0; i < candidates.size(); ++i) {
            if (!used[i]) backfill_indices.push_back(i);
        }

        for (size_t idx : backfill_indices) {
            if (selected.size() >= R) break;
            selected.push_back(candidates[idx]);
        }
    }

    return selected;
}

}  // namespace cphnsw
