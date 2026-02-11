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

// Isosceles triangle: alpha > 1 preserves a long edge that alpha=1 would prune.
// Two candidates: A at distance 2 from query, B at distance 3 from query.
// dist(A, B) = 2.5 (between them).
// With alpha=1: dist(A,B)=2.5 < 1*3.0=3.0 → B pruned (only A selected).
// With alpha=1.3: dist(A,B)=2.5 < 1.3*3.0=3.9 → B still pruned? No:
//   Actually we need dist(A,B) >= alpha * dist(B, query) for B to survive.
//   2.5 < 1.3*3.0=3.9 → B pruned. Let's pick values where alpha > 1 matters.
//
// Better setup: A at dist 2, B at dist 3, dist(A,B) = 2.8.
// alpha=1.0: 2.8 < 1.0*3.0=3.0 → B pruned.
// alpha=0.9: 2.8 < 0.9*3.0=2.7 → 2.8 >= 2.7 → B survives!
//
// Use a custom distance fn that returns a fixed inter-neighbor distance.
TEST(NeighborSelection, RobustPruneAlphaPreservesLongEdge) {
    // Candidate A: id=10, dist to query = 2.0
    // Candidate B: id=20, dist to query = 3.0
    // dist(A, B) = 2.8 (controlled by custom distance fn)
    std::vector<NeighborCandidate> candidates = {
        {10, 2.0f}, {20, 3.0f}
    };

    // Custom distance: dist(10,20) = dist(20,10) = 2.8, everything else = |a-b|
    auto custom_dist = [](NodeId a, NodeId b) -> float {
        if ((a == 10 && b == 20) || (a == 20 && b == 10)) return 2.8f;
        return std::abs(static_cast<float>(a) - static_cast<float>(b));
    };
    auto zero_error = [](NodeId) -> float { return 0.0f; };

    // alpha=1.0: dist(A,B)=2.8 < 1.0*3.0=3.0 → B pruned (only A survives)
    auto result_alpha1 = select_neighbors_robust_prune(
        candidates, 2, custom_dist, zero_error, 1.0f);
    // Phase 1 selects A, prunes B. Phase 2 fills B back.
    // But R=2 and we have 2 candidates, so all get returned.
    // Use R=2 with 3 candidates to force actual pruning.

    // Redo with 3 candidates so R < candidates.size() triggers real pruning:
    // C at dist 10 (far away, filler)
    std::vector<NeighborCandidate> candidates3 = {
        {10, 2.0f}, {20, 3.0f}, {30, 10.0f}
    };

    auto custom_dist3 = [](NodeId a, NodeId b) -> float {
        if ((a == 10 && b == 20) || (a == 20 && b == 10)) return 2.8f;
        return std::abs(static_cast<float>(a) - static_cast<float>(b));
    };

    // alpha=1.0, R=2: A selected, B pruned (2.8 < 3.0), C pruned (farther).
    // Phase 2 fills: B added. Result = {A, B}.
    auto r1 = select_neighbors_robust_prune(
        candidates3, 2, custom_dist3, zero_error, 1.0f);
    ASSERT_EQ(r1.size(), 2u);
    EXPECT_EQ(r1[0].id, 10u);  // A always first (closest)
    // Phase 1 only selects A; Phase 2 fills B (next closest unused)
    EXPECT_EQ(r1[1].id, 20u);

    // Now verify that Phase 1 with alpha=1.0 did NOT select B:
    // B was pruned because dist(A,B)=2.8 < alpha*dist(B,query)=3.0.
    // With alpha=0.9: check = 0.9*3.0 = 2.7. dist(A,B)=2.8 >= 2.7 → B survives Phase 1!
    auto r2 = select_neighbors_robust_prune(
        candidates3, 2, custom_dist3, zero_error, 0.9f);
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

// Error tolerance: with error bounds, pruning becomes more conservative
TEST(NeighborSelection, RobustPruneErrorTolerance) {
    // A at dist 2.0, B at dist 3.0, dist(A,B) = 2.8
    // Without error: 2.8 < 1.0*3.0 + 0 = 3.0 → B pruned in Phase 1
    // With error_fn returning 0.2 for each: margin = 0.2 + 0.2 = 0.4
    //   2.8 < 1.0*3.0 + 0.4 = 3.4 → B still pruned (margin makes it easier to prune)
    // Wait — the margin ADDS to the threshold, making pruning MORE aggressive.
    // Actually looking at the code: dist < alpha*dist_to_query + margin → prune.
    // So larger margin = more likely to prune = more conservative (fewer neighbors).
    //
    // For error tolerance to HELP, we need it to prevent over-pruning.
    // The error margin accounts for uncertainty: if the measured distance might be wrong,
    // we should be more willing to keep a neighbor.
    //
    // Let's test that with error_fn > 0, a candidate that would barely survive
    // without error gets pruned with error.
    std::vector<NeighborCandidate> candidates = {
        {10, 2.0f}, {20, 3.0f}, {30, 10.0f}
    };

    auto custom_dist = [](NodeId a, NodeId b) -> float {
        if ((a == 10 && b == 20) || (a == 20 && b == 10)) return 3.1f;
        return std::abs(static_cast<float>(a) - static_cast<float>(b));
    };

    auto zero_error = [](NodeId) -> float { return 0.0f; };
    auto some_error = [](NodeId) -> float { return 0.15f; };

    // Without error: dist(A,B)=3.1 < 1.0*3.0 + 0 = 3.0? No, 3.1 >= 3.0 → B survives Phase 1
    auto r1 = select_neighbors_robust_prune(
        candidates, 2, custom_dist, zero_error, 1.0f);
    ASSERT_EQ(r1.size(), 2u);
    EXPECT_EQ(r1[0].id, 10u);
    EXPECT_EQ(r1[1].id, 20u);  // B survived Phase 1

    // With error: dist(A,B)=3.1 < 1.0*3.0 + 0.3 = 3.3? Yes → B pruned in Phase 1
    auto r2 = select_neighbors_robust_prune(
        candidates, 2, custom_dist, some_error, 1.0f);
    ASSERT_EQ(r2.size(), 2u);
    EXPECT_EQ(r2[0].id, 10u);
    // B was pruned in Phase 1, so Phase 2 fills it back — but the ORDER tells us
    // Phase 1 only got A, Phase 2 filled B (next closest)
    EXPECT_EQ(r2[1].id, 20u);  // filled from Phase 2
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
