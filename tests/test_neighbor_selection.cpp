#include <gtest/gtest.h>
#include <cphnsw/graph/neighbor_selection.hpp>
#include <cmath>
#include <vector>

using namespace cphnsw;

namespace {
// Simple distance function: abs(a - b) using IDs as 1D coordinates
float id_distance(NodeId a, NodeId b) {
    return std::abs(static_cast<float>(a) - static_cast<float>(b));
}
} // namespace

// ── select_neighbors_heuristic (used by HNSW upper layers) ──

TEST(NeighborSelection, HeuristicDiversityPruning) {
    // Create candidates where some are close to each other
    std::vector<NeighborCandidate> candidates = {
        {0, 1.0f}, {1, 1.5f}, {2, 2.0f}, {10, 3.0f}, {11, 3.5f}, {20, 5.0f}
    };

    auto result = select_neighbors_heuristic(candidates, 3, id_distance);

    // Should select diverse neighbors, not just closest
    ASSERT_LE(result.size(), 3u);
    ASSERT_GE(result.size(), 1u);
    // First should be closest
    EXPECT_EQ(result[0].id, 0u);
}

TEST(NeighborSelection, HeuristicEmptyCandidates) {
    std::vector<NeighborCandidate> empty;
    auto result = select_neighbors_heuristic(empty, 5, id_distance);
    EXPECT_TRUE(result.empty());
}

// ── select_neighbors_robust_prune (used by all layer-0 construction) ──

// Star graph: query at origin, neighbors at distances [1, 2, 3, 4].
// With M=2, should pick the 2 closest: [1, 2].
TEST(NeighborSelection, RobustPruneStarGraph) {
    // Nodes 1,2,3,4 at distances 1,2,3,4 from query.
    // id_distance(a,b) = |a-b|, so inter-neighbor distances are also known.
    // Query is implicit (distances to query are the .distance fields).
    std::vector<NeighborCandidate> candidates = {
        {1, 1.0f}, {2, 2.0f}, {3, 3.0f}, {4, 4.0f}
    };

    auto zero_error = [](NodeId) -> float { return 0.0f; };

    auto result = select_neighbors_robust_prune(
        candidates, 2, id_distance, zero_error, 1.0f);

    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0].id, 1u);  // distance 1.0
    EXPECT_EQ(result[1].id, 2u);  // distance 2.0
}

// Alpha > 1 preserves long edges that alpha=1 would prune (Vamana §3.4).
// Corrected formula: alpha * dist(candidate, existing) < dist(candidate, query) + margin
// With alpha > 1, left side grows → harder to prune → more long-range edges.
//
// Setup: A at dist 2, B at dist 3, dist(A,B) = 2.8.
// alpha=1.0: 1.0*2.8=2.8 < 3.0 → B pruned.
// alpha=1.1: 1.1*2.8=3.08 >= 3.0 → B survives!
//
// Use a custom distance fn that returns a fixed inter-neighbor distance.
TEST(NeighborSelection, RobustPruneAlphaPreservesLongEdge) {
    auto zero_error = [](NodeId) -> float { return 0.0f; };

    // Use R=2 with 3 candidates so R < candidates.size() triggers real pruning:
    // C at dist 10 (far away, filler)
    std::vector<NeighborCandidate> candidates3 = {
        {10, 2.0f}, {20, 3.0f}, {30, 10.0f}
    };

    auto custom_dist3 = [](NodeId a, NodeId b) -> float {
        if ((a == 10 && b == 20) || (a == 20 && b == 10)) return 2.8f;
        return std::abs(static_cast<float>(a) - static_cast<float>(b));
    };

    // alpha=1.0, R=2: A selected, B pruned by A (1.0*2.8=2.8 < 3.0).
    // C survives diversity pruning (dist(C,A)=20 >= 10.0). Phase 1 fills R=2.
    auto r1 = select_neighbors_robust_prune(
        candidates3, 2, custom_dist3, zero_error, 1.0f);
    ASSERT_EQ(r1.size(), 2u);
    EXPECT_EQ(r1[0].id, 10u);  // A first (closest)
    EXPECT_EQ(r1[1].id, 30u);  // C kept by diversity (far from A)

    // With alpha=1.1: 1.1*2.8=3.08 >= 3.0 → B survives Phase 1 (alpha > 1 preserves long edges)
    auto r2 = select_neighbors_robust_prune(
        candidates3, 2, custom_dist3, zero_error, 1.1f);
    ASSERT_EQ(r2.size(), 2u);
    EXPECT_EQ(r2[0].id, 10u);  // A first
    EXPECT_EQ(r2[1].id, 20u);  // B selected in Phase 1 (not just filled)
}

// Deduplication: same node ID with different distances should keep closest
TEST(NeighborSelection, RobustPruneDeduplicates) {
    std::vector<NeighborCandidate> candidates = {
        {5, 3.0f}, {5, 1.0f}, {10, 2.0f}
    };

    auto zero_error = [](NodeId) -> float { return 0.0f; };

    auto result = select_neighbors_robust_prune(
        candidates, 3, id_distance, zero_error, 1.0f);

    // Should deduplicate: only one entry for node 5
    size_t count_5 = 0;
    for (const auto& r : result) {
        if (r.id == 5) {
            count_5++;
            EXPECT_FLOAT_EQ(r.distance, 1.0f);  // kept the closer one
        }
    }
    EXPECT_EQ(count_5, 1u);
}

// Error tolerance: with error bounds, pruning becomes more aggressive.
// Formula: alpha * dist(candidate, existing) < dist(candidate, query) + margin
// Larger margin → right side grows → easier to prune → fewer Phase 1 selections.
// This accounts for distance estimation uncertainty: when errors are high,
// we prune more aggressively and rely on Phase 2 fill to recover neighbors.
TEST(NeighborSelection, RobustPruneErrorTolerance) {
    // A at dist 2.0, B at dist 3.0, dist(A,B) = 3.1
    // Without error: 1.0*3.1=3.1 >= 3.0 → B survives Phase 1
    // With error margin 0.3: 1.0*3.1=3.1 < 3.0+0.3=3.3 → B pruned in Phase 1
    std::vector<NeighborCandidate> candidates = {
        {10, 2.0f}, {20, 3.0f}, {30, 10.0f}
    };

    auto custom_dist = [](NodeId a, NodeId b) -> float {
        if ((a == 10 && b == 20) || (a == 20 && b == 10)) return 3.1f;
        return std::abs(static_cast<float>(a) - static_cast<float>(b));
    };

    auto zero_error = [](NodeId) -> float { return 0.0f; };
    auto some_error = [](NodeId) -> float { return 0.15f; };

    // Without error: 1.0*3.1=3.1 >= 3.0+0=3.0 → B survives Phase 1
    auto r1 = select_neighbors_robust_prune(
        candidates, 2, custom_dist, zero_error, 1.0f);
    ASSERT_EQ(r1.size(), 2u);
    EXPECT_EQ(r1[0].id, 10u);
    EXPECT_EQ(r1[1].id, 20u);  // B survived Phase 1

    // With error: 1.0*3.1=3.1 < 3.0+0.3=3.3 → B pruned in Phase 1.
    // C survives diversity (dist(C,A)=20 >= 10.0+0.3). Phase 1 fills R=2.
    auto r2 = select_neighbors_robust_prune(
        candidates, 2, custom_dist, some_error, 1.0f);
    ASSERT_EQ(r2.size(), 2u);
    EXPECT_EQ(r2[0].id, 10u);
    EXPECT_EQ(r2[1].id, 30u);  // C kept by diversity (B pruned by error margin)
}

// Edge case: fewer candidates than R returns all
TEST(NeighborSelection, RobustPruneFewerThanR) {
    std::vector<NeighborCandidate> candidates = {
        {1, 1.0f}, {2, 2.0f}
    };

    auto zero_error = [](NodeId) -> float { return 0.0f; };

    auto result = select_neighbors_robust_prune(
        candidates, 5, id_distance, zero_error, 1.0f);
    EXPECT_EQ(result.size(), 2u);
}

// Edge case: empty candidates
TEST(NeighborSelection, RobustPruneEmptyCandidates) {
    std::vector<NeighborCandidate> empty;
    auto zero_error = [](NodeId) -> float { return 0.0f; };

    auto result = select_neighbors_robust_prune(
        empty, 5, id_distance, zero_error, 1.0f);
    EXPECT_TRUE(result.empty());
}
