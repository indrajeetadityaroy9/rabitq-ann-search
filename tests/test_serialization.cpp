#include <gtest/gtest.h>
#include <cphnsw/api/rabitq_index.hpp>
#include <cphnsw/io/serialization.hpp>
#include <cphnsw/core/memory.hpp>
#include <random>
#include <cstdio>
#include <string>

using namespace cphnsw;

namespace {
constexpr size_t D = 128;
constexpr size_t R = 32;
constexpr size_t N = 200;

void generate_random_vectors(float* vecs, size_t n, size_t dim, uint64_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < n * dim; ++i) vecs[i] = dist(rng);
}

std::string temp_path() {
    return std::string(std::tmpnam(nullptr)) + ".rbq";
}
} // namespace

TEST(Serialization, SaveLoadRoundtrip) {
    AlignedVector<float> data(N * D);
    generate_random_vectors(data.data(), N, D, 42);

    // Build index and save graph
    RaBitQIndex<D, R> index(D);
    index.add_batch(data.data(), N);
    index.finalize();

    const auto& graph = index.graph();
    std::string path = temp_path();
    IndexSerializer<D, R>::save(path, graph);

    // Load graph back
    auto loaded = IndexSerializer<D, R>::load(path);

    EXPECT_EQ(loaded.size(), graph.size());
    EXPECT_EQ(loaded.entry_point(), graph.entry_point());
    EXPECT_EQ(loaded.dim(), graph.dim());

    // Verify neighbor structure matches for a sample of nodes
    for (size_t i = 0; i < std::min<size_t>(N, 20); ++i) {
        NodeId id = static_cast<NodeId>(i);
        const auto& orig_nb = graph.get_neighbors(id);
        const auto& load_nb = loaded.get_neighbors(id);
        EXPECT_EQ(orig_nb.count, load_nb.count)
            << "Neighbor count mismatch at node " << i;
        for (size_t j = 0; j < orig_nb.count; ++j) {
            EXPECT_EQ(orig_nb.neighbor_ids[j], load_nb.neighbor_ids[j])
                << "Neighbor mismatch at node " << i << " slot " << j;
        }
    }

    // Verify vectors match
    for (size_t i = 0; i < std::min<size_t>(N, 10); ++i) {
        NodeId id = static_cast<NodeId>(i);
        const float* orig_vec = graph.get_vector(id);
        const float* load_vec = loaded.get_vector(id);
        for (size_t d = 0; d < D; ++d) {
            EXPECT_FLOAT_EQ(orig_vec[d], load_vec[d])
                << "Vector mismatch at node " << i << " dim " << d;
        }
    }

    std::remove(path.c_str());
}

TEST(Serialization, InvalidMagicRejected) {
    std::string path = temp_path();
    FILE* f = std::fopen(path.c_str(), "wb");
    ASSERT_NE(f, nullptr);
    char garbage[64] = {};
    std::fwrite(garbage, 1, sizeof(garbage), f);
    std::fclose(f);

    EXPECT_THROW(
        (IndexSerializer<D, R>::load(path)),
        std::runtime_error
    );

    std::remove(path.c_str());
}
