#include <gtest/gtest.h>
#include <cphnsw/encoder/rabitq_encoder.hpp>
#include <cphnsw/core/memory.hpp>
#include <random>
#include <cmath>
#include <numeric>
#include <algorithm>

using namespace cphnsw;

namespace {
constexpr size_t D = 128;

void generate_random_vectors(float* vecs, size_t n, size_t dim, uint64_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < n * dim; ++i) vecs[i] = dist(rng);
}
} // namespace

// 1-bit encoder roundtrip
TEST(EncoderRoundtrip, OneBitEncodePreservesNorm) {
    constexpr size_t N = 100;
    AlignedVector<float> data(N * D);
    generate_random_vectors(data.data(), N, D, 42);

    RaBitQEncoder<D> encoder(D, 42);
    encoder.compute_centroid(data.data(), N);

    for (size_t i = 0; i < N; ++i) {
        const float* vec = data.data() + i * D;
        auto code = encoder.encode(vec);

        // dist_to_centroid should be positive for non-zero vectors
        EXPECT_GT(code.dist_to_centroid, 0.0f) << "Vector " << i << " has zero dist_to_centroid";

        // ip_quantized_original should be in reasonable range [0, 2]
        EXPECT_GE(code.ip_quantized_original, 0.0f);
        EXPECT_LE(code.ip_quantized_original, 2.0f);
    }
}

// Distance estimates correlate with true distances
TEST(EncoderRoundtrip, DistanceEstimateCorrelation) {
    constexpr size_t N = 50;
    AlignedVector<float> data(N * D);
    AlignedVector<float> query_vec(D);
    generate_random_vectors(data.data(), N, D, 42);
    generate_random_vectors(query_vec.data(), 1, D, 99);

    RaBitQEncoder<D> encoder(D, 42);
    encoder.compute_centroid(data.data(), N);

    std::vector<RaBitQCode<D>> codes(N);
    encoder.encode_batch(data.data(), N, codes.data());
    auto query = encoder.encode_query(query_vec.data());

    // Compute exact and estimated distances
    std::vector<float> exact(N), estimated(N);
    for (size_t i = 0; i < N; ++i) {
        exact[i] = l2_distance_simd<D>(query_vec.data(), data.data() + i * D);
        estimated[i] = RaBitQEncoder<D>::compute_distance_scalar(query, codes[i]);
    }

    // Check: nearest by exact should be near-nearest by estimated
    size_t exact_nn = static_cast<size_t>(
        std::min_element(exact.begin(), exact.end()) - exact.begin());

    // The exact NN should be in the top-5 by estimated distance
    std::vector<size_t> est_order(N);
    std::iota(est_order.begin(), est_order.end(), 0);
    std::sort(est_order.begin(), est_order.end(),
              [&](size_t a, size_t b) { return estimated[a] < estimated[b]; });

    bool found_in_top5 = false;
    for (size_t i = 0; i < std::min<size_t>(5, N); ++i) {
        if (est_order[i] == exact_nn) { found_in_top5 = true; break; }
    }
    EXPECT_TRUE(found_in_top5) << "Exact NN not found in top-5 estimated neighbors";
}

// Multi-bit encoder roundtrip
TEST(EncoderRoundtrip, NbitEncodePreservesNorm) {
    constexpr size_t N = 50;
    constexpr size_t BitWidth = 2;
    AlignedVector<float> data(N * D);
    generate_random_vectors(data.data(), N, D, 42);

    NbitRaBitQEncoder<D, BitWidth> encoder(D, 42);
    encoder.compute_centroid(data.data(), N);

    for (size_t i = 0; i < N; ++i) {
        const float* vec = data.data() + i * D;
        auto code = encoder.encode(vec);

        EXPECT_GT(code.dist_to_centroid, 0.0f);
        EXPECT_GE(code.ip_quantized_original, 0.0f);
    }
}

// Batch encode matches single encode
TEST(EncoderRoundtrip, BatchMatchesSingle) {
    constexpr size_t N = 20;
    AlignedVector<float> data(N * D);
    generate_random_vectors(data.data(), N, D, 42);

    RaBitQEncoder<D> encoder(D, 42);
    encoder.compute_centroid(data.data(), N);

    // Single encode
    std::vector<RaBitQCode<D>> single_codes(N);
    for (size_t i = 0; i < N; ++i) {
        single_codes[i] = encoder.encode(data.data() + i * D);
    }

    // Batch encode (need fresh encoder with same centroid)
    RaBitQEncoder<D> encoder2(D, 42);
    encoder2.set_centroid(encoder.centroid().data());
    std::vector<RaBitQCode<D>> batch_codes(N);
    encoder2.encode_batch(data.data(), N, batch_codes.data());

    for (size_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(single_codes[i].dist_to_centroid, batch_codes[i].dist_to_centroid)
            << "Mismatch at vector " << i;
        EXPECT_FLOAT_EQ(single_codes[i].ip_quantized_original, batch_codes[i].ip_quantized_original)
            << "Mismatch at vector " << i;
    }
}
