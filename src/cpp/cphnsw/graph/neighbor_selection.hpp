#pragma once

#include "../core/types.hpp"
#include <vector>
#include <algorithm>
#include <cmath>

namespace cphnsw {

// ============================================================================
// Heuristic Neighbor Selection (HNSW Algorithm 4)
// ============================================================================

/**
 * SELECT-NEIGHBORS-HEURISTIC: Diversity-promoting neighbor selection.
 *
 * From the HNSW paper (Malkov & Yashunin, 2018), Algorithm 4:
 * For each candidate (sorted by distance to target), accept only if
 * it is closer to the target than to any already-selected neighbor.
 *
 * This ensures that selected neighbors are well-spread in angular space,
 * promoting good graph navigability by covering diverse directions.
 *
 * Used during both initial construction and graph refinement.
 */

struct NeighborCandidate {
    NodeId id;
    float distance;

    bool operator<(const NeighborCandidate& other) const {
        return distance < other.distance;
    }
};

/**
 * Select up to M diverse neighbors from candidates using the heuristic.
 *
 * @param candidates Candidate neighbors with distances to the target node
 * @param M Maximum number of neighbors to select
 * @param distance_fn Function(NodeId a, NodeId b) → float distance between nodes
 * @return Selected neighbors (at most M), sorted by distance
 */
template <typename DistanceFn>
std::vector<NeighborCandidate> select_neighbors_heuristic(
    std::vector<NeighborCandidate> candidates,
    size_t M,
    DistanceFn distance_fn)
{
    // Sort candidates by distance to target (closest first)
    std::sort(candidates.begin(), candidates.end());

    std::vector<NeighborCandidate> selected;
    selected.reserve(M);

    for (const auto& candidate : candidates) {
        if (selected.size() >= M) break;

        // Check if this candidate is closer to the target than to
        // any already-selected neighbor
        bool should_add = true;
        for (const auto& existing : selected) {
            float dist_to_existing = distance_fn(candidate.id, existing.id);
            if (dist_to_existing < candidate.distance) {
                // Candidate is closer to an existing neighbor than to target
                // → pruned for diversity
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

/**
 * Simplified heuristic: select up to M neighbors without inter-neighbor
 * distance checks. Just takes the M closest.
 */
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

/**
 * Adaptive angle-based pruning (SymphonyQG Section 4.2).
 *
 * For a given vertex, selects exactly R neighbors by binary searching
 * on an angle threshold. Larger threshold = more pruning.
 *
 * @param candidates Sorted candidate list
 * @param R Target degree (must get exactly this many)
 * @param distance_fn Function to compute inter-neighbor distances
 * @return Exactly R neighbors (padded with closest if necessary)
 */
template <typename DistanceFn>
std::vector<NeighborCandidate> select_neighbors_fixed_degree(
    std::vector<NeighborCandidate> candidates,
    size_t R,
    DistanceFn distance_fn)
{
    if (candidates.empty()) return {};

    std::sort(candidates.begin(), candidates.end());

    // If we have fewer candidates than R, return all
    if (candidates.size() <= R) {
        return candidates;
    }

    // Binary search on the pruning threshold
    // threshold = ratio of dist_to_existing / dist_to_target
    // Higher threshold = less pruning = more neighbors kept
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
            // Too much pruning, relax threshold
            hi = threshold;
        } else {
            lo = threshold;
        }

        best_result = std::move(selected);
    }

    // If binary search didn't converge exactly, pad or trim
    if (best_result.size() < R) {
        // Pad with closest remaining candidates
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
