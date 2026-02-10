#include <gtest/gtest.h>
#include <cphnsw/api/hnsw_index.hpp>
#include <cphnsw/core/memory.hpp>
#include <random>

using namespace cphnsw;

namespace {
constexpr size_t D = 128;
constexpr size_t R = 32;

void generate_random_vectors(float* vecs, size_t n, size_t dim, uint64_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < n * dim; ++i) vecs[i] = dist(rng);
}
} // namespace

// Placeholder: Verify deterministic build produces consistent search results
TEST(Serialization, DeterministicBuild) {
    constexpr size_t N = 200;
    AlignedVector<float> data(N * D);
    AlignedVector<float> query(D);
    generate_random_vectors(data.data(), N, D, 42);
    generate_random_vectors(query.data(), 1, D, 99);

    // Build twice with same seed
    auto build = [&]() {
        HNSWIndex<D, R> index(D);
        index.add_batch(data.data(), N);
        index.finalize(1, false);
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

// Placeholder: Will test save/load once serialization is implemented
TEST(Serialization, DISABLED_SaveLoadRoundtrip) {
    // TODO: Implement once io/serialization.hpp exists
    GTEST_SKIP() << "Serialization not yet implemented";
}
