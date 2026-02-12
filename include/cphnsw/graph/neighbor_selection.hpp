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

// α-Convergent Graph pruning (arXiv:2510.05975, Definition 3.1).
//
// Shifted-scaled triangle inequality: prunes candidate if a previously
// selected neighbor is "close enough" relative to both the candidate's
// distance to query and the existing neighbor's distance to query.
//
// Pruning condition (candidate is pruned if this holds for any existing):
//   dist(candidate, existing) < alpha * dist(candidate, query) + (alpha-1) * dist(existing, query) + margin
//
// When error_fn returns 0 (exact distances), margin = 0 and this reduces
// to the standard α-CG definition. When alpha = 1.0 and error_fn = 0,
// this reduces to standard RNG diversity pruning.
//
// Phase 1: α-CG diversity pruning with error tolerance
// Phase 2: Fill remaining slots with closest unused candidates
template <typename DistanceFn, typename ErrorFn>
std::vector<NeighborCandidate> select_neighbors_alpha_cg(
    std::vector<NeighborCandidate> candidates,
    size_t R,
    DistanceFn distance_fn,
    ErrorFn error_fn,
    float alpha = 1.2f)
{
    // Deduplicate by id (keep closest distance)
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) {
                  return a.id < b.id || (a.id == b.id && a.distance < b.distance);
              });
    candidates.erase(
        std::unique(candidates.begin(), candidates.end(),
                    [](const auto& a, const auto& b) { return a.id == b.id; }),
        candidates.end());

    // Sort by distance for greedy selection
    std::sort(candidates.begin(), candidates.end());

    if (candidates.size() <= R) return candidates;

    std::vector<NeighborCandidate> selected;
    selected.reserve(R);
    std::vector<bool> used(candidates.size(), false);

    // Phase 1: α-CG diversity pruning
    for (size_t i = 0; i < candidates.size() && selected.size() < R; ++i) {
        bool should_add = true;
        float err_candidate = error_fn(candidates[i].id);
        float dist_cq = candidates[i].distance;  // dist(candidate, query)

        for (const auto& existing : selected) {
            float dist_ce = distance_fn(candidates[i].id, existing.id);
            float dist_eq = existing.distance;  // dist(existing, query)
            float err_existing = error_fn(existing.id);
            float margin = err_candidate + err_existing;

            // α-CG condition: prune if existing is close enough
            if (dist_ce < alpha * dist_cq + (alpha - 1.0f) * dist_eq + margin) {
                should_add = false;
                break;
            }
        }
        if (should_add) {
            selected.push_back(candidates[i]);
            used[i] = true;
        }
    }

    // Phase 2: Fill remaining slots with closest unused candidates
    for (size_t i = 0; i < candidates.size() && selected.size() < R; ++i) {
        if (!used[i]) {
            selected.push_back(candidates[i]);
        }
    }

    return selected;
}

}  // namespace cphnsw
