#include <gtest/gtest.h>
#include <cphnsw/api/rabitq_index.hpp>
#include <cphnsw/core/memory.hpp>
#include <cphnsw/core/adaptive_defaults.hpp>
#include <random>
#include <algorithm>
#include <unordered_set>

using namespace cphnsw;

namespace {
constexpr size_t D = 128;
constexpr size_t R = 32;

void generate_random_vectors(float* vecs, size_t n, size_t dim, uint64_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < n * dim; ++i) vecs[i] = dist(rng);
}

// Brute-force k-NN ground truth
std::vector<std::vector<NodeId>> brute_force_knn(
    const float* data, size_t n, size_t dim,
    const float* queries, size_t nq, size_t k)
{
    std::vector<std::vector<NodeId>> gt(nq);
    for (size_t q = 0; q < nq; ++q) {
        std::vector<std::pair<float, NodeId>> dists;
        dists.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            float d = l2_distance_simd<D>(queries + q * dim, data + i * dim);
            dists.push_back({d, static_cast<NodeId>(i)});
        }
        std::partial_sort(dists.begin(), dists.begin() + k, dists.end());
        gt[q].resize(k);
        for (size_t i = 0; i < k; ++i) gt[q][i] = dists[i].second;
    }
    return gt;
}

float compute_recall(
    const std::vector<SearchResult>& results,
    const std::vector<NodeId>& gt, size_t k)
{
    std::unordered_set<NodeId> gt_set(gt.begin(), gt.begin() + k);
    size_t found = 0;
    for (size_t i = 0; i < std::min(results.size(), k); ++i) {
        if (gt_set.count(results[i].id)) ++found;
    }
    return static_cast<float>(found) / static_cast<float>(k);
}
} // namespace

// Test 1: Default (adaptive) construction produces a navigable graph with reasonable recall
TEST(QuantizedConstruction, ProducesNavigableGraph) {
    constexpr size_t N = 1000;
    constexpr size_t NQ = 100;
    constexpr size_t K = 10;

    AlignedVector<float> data(N * D);
    generate_random_vectors(data.data(), N, D, 42);
    AlignedVector<float> queries(NQ * D);
    generate_random_vectors(queries.data(), NQ, D, 123);

    auto gt = brute_force_knn(data.data(), N, D, queries.data(), NQ, K);

    RaBitQIndex<D, R> index(D);
    index.add_batch(data.data(), N);
    index.finalize();  // Zero-config: all defaults auto-derived

    float total_recall = 0.0f;
    for (size_t q = 0; q < NQ; ++q) {
        auto results = index.search(queries.data() + q * D,
                                     SearchParams().set_k(K));
        total_recall += compute_recall(results, gt[q], K);
    }
    float avg_recall = total_recall / NQ;

    EXPECT_GE(avg_recall, 0.60f)
        << "Adaptive construction recall@" << K << " = " << avg_recall
        << " (expected >= 0.60)";
}

// Test 2: Error-aware pruning increases average degree (more conservative = more edges)
TEST(QuantizedConstruction, ErrorAwarePruningIncreasesAvgDegree) {
    constexpr size_t N = 500;

    AlignedVector<float> data(N * D);
    generate_random_vectors(data.data(), N, D, 42);

    // Build without error tolerance
    RaBitQIndex<D, R> index_no_err(D);
    index_no_err.add_batch(data.data(), N);
    BuildParams params_no_err;
    params_no_err.ef_construction = 200;
    params_no_err.error_tolerance = 0.0f;
    index_no_err.finalize(params_no_err);

    float avg_degree_no_err = index_no_err.graph().average_degree();

    // Build with error tolerance
    RaBitQIndex<D, R> index_err(D);
    index_err.add_batch(data.data(), N);
    BuildParams params_err;
    params_err.ef_construction = 200;
    params_err.error_tolerance = 1.0f;
    index_err.finalize(params_err);

    float avg_degree_err = index_err.graph().average_degree();

    // Error-tolerant pruning is more conservative, should keep more edges
    EXPECT_GE(avg_degree_err, avg_degree_no_err)
        << "Expected error-tolerant pruning to keep more edges. "
        << "Without error: " << avg_degree_no_err
        << ", with error: " << avg_degree_err;
}

// Test 3: Default finalize (two-pass) should produce good recall
TEST(QuantizedConstruction, DefaultFinalizeProducesGoodRecall) {
    constexpr size_t N = 500;
    constexpr size_t NQ = 50;
    constexpr size_t K = 10;

    AlignedVector<float> data(N * D);
    generate_random_vectors(data.data(), N, D, 42);
    AlignedVector<float> queries(NQ * D);
    generate_random_vectors(queries.data(), NQ, D, 123);

    auto gt = brute_force_knn(data.data(), N, D, queries.data(), NQ, K);

    // Default finalize: two-pass, quantized, adaptive parameters
    RaBitQIndex<D, R> index(D);
    index.add_batch(data.data(), N);
    index.finalize();

    float total_recall = 0.0f;
    for (size_t q = 0; q < NQ; ++q) {
        auto results = index.search(queries.data() + q * D,
                                     SearchParams().set_k(K));
        total_recall += compute_recall(results, gt[q], K);
    }
    float avg_recall = total_recall / NQ;

    EXPECT_GE(avg_recall, 0.70f)
        << "Default finalize recall@" << K << " = " << avg_recall
        << " (expected >= 0.70)";
}

// Test 4: Small graph (N < R) doesn't crash
TEST(QuantizedConstruction, SmallGraphDoesNotCrash) {
    constexpr size_t N_SMALL = 16;  // Less than R=32

    AlignedVector<float> data(N_SMALL * D);
    generate_random_vectors(data.data(), N_SMALL, D, 42);

    RaBitQIndex<D, R> index(D);
    index.add_batch(data.data(), N_SMALL);

    // Should not crash
    ASSERT_NO_THROW(index.finalize());

    // Search should still work
    auto results = index.search(data.data(), SearchParams().set_k(1).set_ef(16));
    EXPECT_FALSE(results.empty());
    EXPECT_EQ(results[0].id, 0u);  // Self-search should return self
}

// Test 5: AdaptiveDefaults produce sane values
TEST(QuantizedConstruction, AdaptiveDefaultsSanity) {
    // ef_construction
    EXPECT_EQ(AdaptiveDefaults::ef_construction(1000, 32), 64u);  // ceil(2*sqrt(1000))=64
    EXPECT_EQ(AdaptiveDefaults::ef_construction(1000000, 32), 200u);  // clamped to 200
    EXPECT_EQ(AdaptiveDefaults::ef_construction(4, 32), 32u);  // clamped to R

    // error_tolerance
    float tol_128 = AdaptiveDefaults::error_tolerance(128);
    EXPECT_NEAR(tol_128, 0.0884f, 0.001f);

    // error_epsilon_search
    float eps_95 = AdaptiveDefaults::error_epsilon_search(0.95f);
    EXPECT_GT(eps_95, 1.5f);
    EXPECT_LT(eps_95, 3.0f);

    // ef_search
    size_t ef_95 = AdaptiveDefaults::ef_search(10, 0.95f);
    EXPECT_GE(ef_95, 10u);
    EXPECT_LE(ef_95, 100u);

    // ef grows for higher recall
    size_t ef_99 = AdaptiveDefaults::ef_search(10, 0.99f);
    EXPECT_GT(ef_99, ef_95);

    // Constants
    EXPECT_FLOAT_EQ(AdaptiveDefaults::ALPHA, 1.0f);
    EXPECT_FLOAT_EQ(AdaptiveDefaults::ALPHA_PASS2, 1.2f);
}
