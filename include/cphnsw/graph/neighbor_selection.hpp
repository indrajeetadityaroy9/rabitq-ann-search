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