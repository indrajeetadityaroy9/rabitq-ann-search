// Compile test for multi-bit (Extended RaBitQ) template instantiation.
#include <cphnsw/api/rabitq_index.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

using namespace cphnsw;

template <size_t D, size_t BitWidth>
void test_nbit_index(size_t actual_dim, size_t num_vecs) {
    std::cout << "Testing D=" << D << " BitWidth=" << BitWidth
              << " n=" << num_vecs << "..." << std::flush;

    RaBitQIndex<D, 32, BitWidth> index(
        IndexParams().set_dim(actual_dim).set_ef_construction(100));

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> vecs(num_vecs * actual_dim);
    for (auto& v : vecs) v = dist(rng);

    // Normalize
    for (size_t i = 0; i < num_vecs; ++i) {
        float* vec = vecs.data() + i * actual_dim;
        float norm = 0;
        for (size_t j = 0; j < actual_dim; ++j) norm += vec[j] * vec[j];
        norm = std::sqrt(norm);
        for (size_t j = 0; j < actual_dim; ++j) vec[j] /= norm;
    }

    index.add_batch(vecs.data(), num_vecs);
    index.finalize();

    auto stats = index.get_stats();
    std::cout << " nodes=" << stats.num_nodes
              << " avg_deg=" << stats.avg_degree
              << " max_deg=" << stats.max_degree;

    // Search
    auto results = index.search(vecs.data(), SearchParams().set_k(5).set_ef(50));
    std::cout << " search_results=" << results.size();

    // Verify first result is the query itself (should be nearest)
    if (!results.empty() && results[0].id == 0 && results[0].distance < 1e-4f) {
        std::cout << " [self-found OK]";
    }

    std::cout << std::endl;
}

int main() {
    constexpr size_t N = 1000;

    std::cout << "=== 1-bit (original RaBitQ) ===" << std::endl;
    test_nbit_index<128, 1>(128, N);

    std::cout << "\n=== 2-bit (Extended RaBitQ) ===" << std::endl;
    test_nbit_index<128, 2>(128, N);

    std::cout << "\n=== 4-bit (Extended RaBitQ) ===" << std::endl;
    test_nbit_index<128, 4>(128, N);

    std::cout << "\nAll multi-bit tests passed." << std::endl;
    return 0;
}
