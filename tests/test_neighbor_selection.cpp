#include <gtest/gtest.h>
#include <cphnsw/graph/neighbor_selection.hpp>
#include <cmath>

using namespace cphnsw;

namespace {
// Simple distance function: abs(a - b) using IDs as 1D coordinates
float id_distance(NodeId a, NodeId b) {
    return std::abs(static_cast<float>(a) - static_cast<float>(b));
}
} // namespace

TEST(NeighborSelection, SimpleSelectsClosest) {
    std::vector<NeighborCandidate> candidates = {
        {10, 5.0f}, {20, 3.0f}, {30, 1.0f}, {40, 4.0f}, {50, 2.0f}
    };

    auto result = select_neighbors_simple(candidates, 3);
    ASSERT_EQ(result.size(), 3u);
    EXPECT_EQ(result[0].id, 30u);  // dist 1.0
    EXPECT_EQ(result[1].id, 50u);  // dist 2.0
    EXPECT_EQ(result[2].id, 20u);  // dist 3.0
}

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

TEST(NeighborSelection, HeuristicFillAllSlots) {
    std::vector<NeighborCandidate> candidates = {
        {0, 1.0f}, {1, 1.5f}, {2, 2.0f}, {10, 3.0f}, {11, 3.5f}, {20, 5.0f}
    };

    auto result = select_neighbors_heuristic_fill(candidates, 4, id_distance);

    // Should fill all 4 slots
    EXPECT_EQ(result.size(), 4u);
}

TEST(NeighborSelection, HeuristicFillDeduplicates) {
    std::vector<NeighborCandidate> candidates = {
        {5, 1.0f}, {5, 2.0f}, {10, 3.0f}
    };

    auto result = select_neighbors_heuristic_fill(candidates, 3, id_distance);

    // Should deduplicate: only one entry for node 5
    size_t count_5 = 0;
    for (const auto& r : result) {
        if (r.id == 5) count_5++;
    }
    EXPECT_EQ(count_5, 1u);
    // And it should keep the closer one
    for (const auto& r : result) {
        if (r.id == 5) EXPECT_FLOAT_EQ(r.distance, 1.0f);
    }
}

TEST(NeighborSelection, EmptyCandidates) {
    std::vector<NeighborCandidate> empty;
    auto result = select_neighbors_simple(empty, 5);
    EXPECT_TRUE(result.empty());

    auto result2 = select_neighbors_heuristic(empty, 5, id_distance);
    EXPECT_TRUE(result2.empty());

    auto result3 = select_neighbors_heuristic_fill(empty, 5, id_distance);
    EXPECT_TRUE(result3.empty());
}

TEST(NeighborSelection, FewerCandidatesThanM) {
    std::vector<NeighborCandidate> candidates = {
        {1, 1.0f}, {2, 2.0f}
    };

    auto result = select_neighbors_simple(candidates, 5);
    EXPECT_EQ(result.size(), 2u);

    auto result2 = select_neighbors_heuristic_fill(candidates, 5, id_distance);
    EXPECT_EQ(result2.size(), 2u);
}
