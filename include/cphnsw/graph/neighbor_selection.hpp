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

template <typename DistanceFn>
std::vector<NeighborCandidate> select_neighbors_heuristic(
    std::vector<NeighborCandidate> candidates,
    size_t M,
    DistanceFn distance_fn)
{
    std::sort(candidates.begin(), candidates.end());

    std::vector<NeighborCandidate> selected;
    selected.reserve(M);

    for (const auto& candidate : candidates) {
        if (selected.size() >= M) break;

        bool should_add = true;
        for (const auto& existing : selected) {
            float dist_to_existing = distance_fn(candidate.id, existing.id);
            if (dist_to_existing < candidate.distance) {
                should_add = false;
                break;
            }
        }

        if (should_add) {
            selected.push_back(candidate);
        }
    }

    return selected;
}

// Diversity pruning + slot filling (from SymphonyQG / Revisiting PG Construction).
// Phase 1: Standard HNSW diversity heuristic selects diverse neighbors.
// Phase 2: Fill remaining slots with closest unused candidates.
// This ensures all R slots in the FastScan block are populated.
template <typename DistanceFn>
std::vector<NeighborCandidate> select_neighbors_heuristic_fill(
    std::vector<NeighborCandidate> candidates,
    size_t M,
    DistanceFn distance_fn)
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

    if (candidates.size() <= M) return candidates;

    std::vector<NeighborCandidate> selected;
    selected.reserve(M);
    std::vector<bool> used(candidates.size(), false);

    // Phase 1: Diversity pruning (standard HNSW heuristic)
    for (size_t i = 0; i < candidates.size() && selected.size() < M; ++i) {
        bool should_add = true;
        for (const auto& existing : selected) {
            float dist_to_existing = distance_fn(candidates[i].id, existing.id);
            if (dist_to_existing < candidates[i].distance) {
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
    for (size_t i = 0; i < candidates.size() && selected.size() < M; ++i) {
        if (!used[i]) {
            selected.push_back(candidates[i]);
        }
    }

    return selected;
}

// QRG: Error-tolerant RobustPrune with per-vector error bounds.
// Preserves O(n^{2/3+ε}) degree bounds under quantized distances.
//
// Phase 1: Diversity pruning with alpha scaling and error margins.
//   Prunes candidate if: dist(candidate, existing) < alpha * dist(candidate, query) + margin
//   where margin = error_fn(candidate) + error_fn(existing) is the worst-case error on both sides.
// Phase 2: Fill remaining slots with closest unused candidates.
//
// When error_fn returns 0 for all nodes (exact distances), this reduces to standard Vamana RobustPrune.
template <typename DistanceFn, typename ErrorFn>
std::vector<NeighborCandidate> select_neighbors_robust_prune(
    std::vector<NeighborCandidate> candidates,
    size_t R,
    DistanceFn distance_fn,
    ErrorFn error_fn,
    float alpha = 1.0f)
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

    // Phase 1: Diversity pruning with alpha and per-vector error tolerance
    for (size_t i = 0; i < candidates.size() && selected.size() < R; ++i) {
        bool should_add = true;
        float err_candidate = error_fn(candidates[i].id);

        for (const auto& existing : selected) {
            float dist_to_existing = distance_fn(candidates[i].id, existing.id);
            float err_existing = error_fn(existing.id);
            float margin = err_candidate + err_existing;

            if (dist_to_existing < alpha * candidates[i].distance + margin) {
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

inline std::vector<NeighborCandidate> select_neighbors_simple(
    std::vector<NeighborCandidate> candidates,
    size_t M)
{
    std::sort(candidates.begin(), candidates.end());
    if (candidates.size() > M) {
        candidates.resize(M);
    }
    return candidates;
}

template <typename DistanceFn>
std::vector<NeighborCandidate> select_neighbors_fixed_degree(
    std::vector<NeighborCandidate> candidates,
    size_t R,
    DistanceFn distance_fn)
{
    if (candidates.empty()) return {};

    std::sort(candidates.begin(), candidates.end());

    if (candidates.size() <= R) {
        return candidates;
    }

    float lo = 0.5f, hi = 2.0f;

    std::vector<NeighborCandidate> best_result;

    for (int iter = 0; iter < 20; ++iter) {
        float threshold = (lo + hi) / 2.0f;

        std::vector<NeighborCandidate> selected;
        selected.reserve(R);

        for (const auto& cand : candidates) {
            if (selected.size() >= R) break;

            bool should_add = true;
            for (const auto& existing : selected) {
                float dist_to_existing = distance_fn(cand.id, existing.id);
                if (dist_to_existing < cand.distance * threshold) {
                    should_add = false;
                    break;
                }
            }

            if (should_add) {
                selected.push_back(cand);
            }
        }

        if (selected.size() == R) {
            return selected;
        } else if (selected.size() < R) {
            hi = threshold;
        } else {
            lo = threshold;
        }

        best_result = std::move(selected);
    }

    if (best_result.size() < R) {
        for (const auto& cand : candidates) {
            if (best_result.size() >= R) break;
            bool already_in = false;
            for (const auto& s : best_result) {
                if (s.id == cand.id) { already_in = true; break; }
            }
            if (!already_in) {
                best_result.push_back(cand);
            }
        }
    }

    if (best_result.size() > R) {
        best_result.resize(R);
    }

    return best_result;
}

}  // namespace cphnsw