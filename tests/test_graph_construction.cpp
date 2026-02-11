#include <gtest/gtest.h>
#include <cphnsw/api/hnsw_index.hpp>
#include <cphnsw/core/memory.hpp>
#include <random>
#include <queue>
#include <unordered_set>

using namespace cphnsw;

namespace {
constexpr size_t D = 128;
constexpr size_t R = 32;
constexpr size_t N = 500;  // small enough for fast tests

void generate_random_vectors(float* vecs, size_t n, size_t dim, uint64_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < n * dim; ++i) vecs[i] = dist(rng);
}
} // namespace

// BFS connectivity: all nodes reachable from entry
TEST(GraphConstruction, BFSConnectivity) {
    AlignedVector<float> data(N * D);
    generate_random_vectors(data.data(), N, D, 42);

    HNSWIndex<D, R> index(D);
    index.add_batch(data.data(), N);
    index.finalize();

    const auto& graph = index.graph();
    NodeId entry = graph.entry_point();
    ASSERT_NE(entry, INVALID_NODE);

    // BFS from entry
    std::unordered_set<NodeId> visited;
    std::queue<NodeId> bfs_queue;
    bfs_queue.push(entry);
    visited.insert(entry);

    while (!bfs_queue.empty()) {
        NodeId u = bfs_queue.front();
        bfs_queue.pop();
        const auto& nb = graph.get_neighbors(u);
        for (size_t i = 0; i < nb.count; ++i) {
            NodeId v = nb.neighbor_ids[i];
            if (v != INVALID_NODE && visited.find(v) == visited.end()) {
                visited.insert(v);
                bfs_queue.push(v);
            }
        }
    }

    // At least 90% of nodes should be reachable (allow some slack for small graphs)
    float reachability = static_cast<float>(visited.size()) / N;
    EXPECT_GT(reachability, 0.9f) << "Only " << visited.size() << "/" << N << " nodes reachable";
}

// Degree constraint: no node exceeds R neighbors
TEST(GraphConstruction, DegreeConstraint) {
    AlignedVector<float> data(N * D);
    generate_random_vectors(data.data(), N, D, 42);

    HNSWIndex<D, R> index(D);
    index.add_batch(data.data(), N);
    index.finalize();

    const auto& graph = index.graph();
    for (size_t i = 0; i < N; ++i) {
        const auto& nb = graph.get_neighbors(static_cast<NodeId>(i));
        EXPECT_LE(nb.count, R) << "Node " << i << " has degree " << nb.count << " > R=" << R;
    }
}

// Search quality: self-search should return self as nearest
TEST(GraphConstruction, SelfSearchQuality) {
    AlignedVector<float> data(N * D);
    generate_random_vectors(data.data(), N, D, 42);

    HNSWIndex<D, R> index(D);
    index.add_batch(data.data(), N);
    index.finalize();

    size_t self_found = 0;
    for (size_t i = 0; i < std::min<size_t>(N, 50); ++i) {
        auto results = index.search(data.data() + i * D, SearchParams().set_k(1).set_ef(50));
        if (!results.empty() && results[0].id == static_cast<NodeId>(i)) {
            self_found++;
        }
    }

    float self_recall = static_cast<float>(self_found) / std::min<size_t>(N, 50);
    EXPECT_GT(self_recall, 0.9f) << "Self-search recall too low: " << self_recall;
}

// Deterministic build: two builds with same seed produce identical results
TEST(GraphConstruction, DeterministicBuild) {
    constexpr size_t N_DET = 200;
    AlignedVector<float> data(N_DET * D);
    AlignedVector<float> query(D);
    generate_random_vectors(data.data(), N_DET, D, 42);
    generate_random_vectors(query.data(), 1, D, 99);

    auto build = [&]() {
        HNSWIndex<D, R> index(D);
        index.add_batch(data.data(), N_DET);
        index.finalize();
        return index.search(query.data(), SearchParams().set_k(10).set_ef(50));
    };

    auto results1 = build();
    auto results2 = build();

    ASSERT_EQ(results1.size(), results2.size());
    for (size_t i = 0; i < results1.size(); ++i) {
        EXPECT_EQ(results1[i].id, results2[i].id) << "Result " << i << " differs";
        EXPECT_FLOAT_EQ(results1[i].distance, results2[i].distance);
    }
}
