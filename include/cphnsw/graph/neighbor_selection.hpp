#pragma once

#include "../core/types.hpp"
#include "../core/constants.hpp"
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


template <typename DistanceFn, typename ErrorFn>
std::vector<NeighborCandidate> select_neighbors_alpha_cng(
    std::vector<NeighborCandidate> candidates,
    size_t R,
    DistanceFn distance_fn,
    ErrorFn error_fn,
    float alpha,
    float tau,
    float alpha_max = 0.0f)
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

    // If alpha_max not provided, default to 2·alpha (sqrt scaling cap)
    if (alpha_max <= 0.0f) alpha_max = 2.0f * alpha;

    float local_alpha = alpha * std::sqrt(
        static_cast<float>(candidates.size()) / static_cast<float>(R));
    local_alpha = std::clamp(local_alpha, 1.0f, alpha_max);

    std::vector<NeighborCandidate> selected;
    selected.reserve(R);

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
        }
    }

    if (selected.size() < R) {
        for (size_t i = 0; i < candidates.size() && selected.size() < R; ++i) {
            bool already_selected = false;
            for (const auto& s : selected) {
                if (s.id == candidates[i].id) { already_selected = true; break; }
            }
            if (!already_selected) {
                selected.push_back(candidates[i]);
            }
        }
    }

    return selected;
}

}
