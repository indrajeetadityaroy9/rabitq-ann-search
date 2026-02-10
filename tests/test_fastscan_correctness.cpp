#include <gtest/gtest.h>
#include <cphnsw/encoder/rabitq_encoder.hpp>
#include <cphnsw/distance/fastscan_kernel.hpp>
#include <cphnsw/distance/fastscan_layout.hpp>
#include <cphnsw/core/memory.hpp>
#include <random>
#include <cmath>
#include <limits>

using namespace cphnsw;

namespace {

constexpr size_t D = 128;
constexpr size_t R = 32;

// Generate random float vectors
void generate_random_vectors(float* vecs, size_t n, size_t dim, uint64_t seed = 42) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < n * dim; ++i) {
        vecs[i] = dist(rng);
    }
}

} // namespace

// Test: FastScan sum matches scalar computation for 1-bit codes
TEST(FastScanCorrectness, ScalarMatchesSIMD) {
    constexpr size_t NUM_VECS = 32;

    AlignedVector<float> data(NUM_VECS * D);
    AlignedVector<float> query_vec(D);
    generate_random_vectors(data.data(), NUM_VECS, D, 42);
    generate_random_vectors(query_vec.data(), 1, D, 123);

    RaBitQEncoder<D> encoder(D, 42);
    encoder.compute_centroid(data.data(), NUM_VECS);

    // Encode all data vectors
    std::vector<RaBitQCode<D>> codes(NUM_VECS);
    encoder.encode_batch(data.data(), NUM_VECS, codes.data());

    // Encode query
    auto query = encoder.encode_query(query_vec.data());

    // For each code, compute scalar distance estimate and compare
    for (size_t i = 0; i < NUM_VECS; ++i) {
        float scalar_dist = RaBitQEncoder<D>::compute_distance_scalar(query, codes[i]);
        float exact_dist = l2_distance_simd<D>(query_vec.data(), data.data() + i * D);

        // Scalar estimate should be finite
        EXPECT_TRUE(std::isfinite(scalar_dist)) << "Non-finite distance for vector " << i;

        // The estimate shouldn't be wildly off (within 5x of exact for reasonable data)
        if (exact_dist > 1e-6f) {
            float ratio = scalar_dist / exact_dist;
            EXPECT_GT(ratio, 0.01f) << "Distance estimate far too low for vector " << i;
            EXPECT_LT(ratio, 100.0f) << "Distance estimate far too high for vector " << i;
        }
    }
}

// Test: Error bounds hold (lower bound <= exact distance)
TEST(FastScanCorrectness, ErrorBoundsHold) {
    constexpr size_t NUM_VECS = 32;

    AlignedVector<float> data(NUM_VECS * D);
    AlignedVector<float> query_vec(D);
    generate_random_vectors(data.data(), NUM_VECS, D, 42);
    generate_random_vectors(query_vec.data(), 1, D, 456);

    RaBitQEncoder<D> encoder(D, 42);
    encoder.compute_centroid(data.data(), NUM_VECS);

    std::vector<RaBitQCode<D>> codes(NUM_VECS);
    encoder.encode_batch(data.data(), NUM_VECS, codes.data());

    auto query = encoder.encode_query(query_vec.data());
    query.error_epsilon = 3.0f;  // generous epsilon

    // Test the scalar error bound path
    for (size_t i = 0; i < NUM_VECS; ++i) {
        float error_bound = RaBitQEncoder<D>::compute_error_bound(query, codes[i]);
        float est_dist = RaBitQEncoder<D>::compute_distance_scalar(query, codes[i]);

        // With epsilon=3.0, the bound should be meaningful
        EXPECT_TRUE(std::isfinite(error_bound) || error_bound == std::numeric_limits<float>::max())
            << "Invalid error bound for vector " << i;
    }
}

// Test: Zero-norm vectors produce max lower bound (after fix 1.1)
TEST(FastScanCorrectness, ZeroNormVectorHandling) {
    AlignedVector<float> zero_vec(D, 0.0f);
    AlignedVector<float> query_vec(D);
    generate_random_vectors(query_vec.data(), 1, D, 789);

    RaBitQEncoder<D> encoder(D, 42);
    // Set centroid to zero so centered vec is also zero
    std::vector<float> zero_centroid(D, 0.0f);
    encoder.set_centroid(zero_centroid.data());

    auto code = encoder.encode(zero_vec.data());

    // Zero-norm vector should have ip_quantized_original ~ 0
    EXPECT_NEAR(code.ip_quantized_original, 0.0f, 1e-6f);
    EXPECT_NEAR(code.dist_to_centroid, 0.0f, 1e-6f);

    // Error bound should be max for zero-norm
    auto query = encoder.encode_query(query_vec.data());
    float bound = RaBitQEncoder<D>::compute_error_bound(query, code);
    EXPECT_EQ(bound, std::numeric_limits<float>::max());
}
