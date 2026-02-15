#pragma once

#include "../core/types.hpp"
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
std::vector<NeighborCandidate> select_neighbors_alpha_cg(
    std::vector<NeighborCandidate> candidates,
    size_t R,
    DistanceFn distance_fn,
    ErrorFn error_fn,
    float alpha)
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

            if (dist_ce < alpha * dist_cq + margin) {
                should_add = false;
                break;
            }
        }
        if (should_add) {
            selected.push_back(candidates[i]);
            used[i] = true;
        }
    }

    for (size_t i = 0; i < candidates.size() && selected.size() < R; ++i) {
        if (!used[i]) {
            selected.push_back(candidates[i]);
        }
    }

    return selected;
}

}  // namespace cphnsw
