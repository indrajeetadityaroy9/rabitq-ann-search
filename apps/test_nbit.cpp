#include <cphnsw/api/rabitq_index.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

using namespace cphnsw;

template <size_t D, size_t BitWidth>
void test_nbit_index(size_t actual_dim, size_t num_vecs) {
    RaBitQIndex<D, 32, BitWidth> index(IndexParams().set_dim(actual_dim));

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> vecs(num_vecs * actual_dim);
    for (auto& v : vecs) v = dist(rng);

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

    auto results = index.search(vecs.data(), SearchParams().set_k(5));
    const bool self_found = (!results.empty() && results[0].id == 0 && results[0].distance < 1e-4f);

    std::cout << "event=nbit_test_case"
              << " template_dim=" << D
              << " bit_width=" << BitWidth
              << " n=" << num_vecs
              << " nodes=" << stats.num_nodes
              << " avg_degree=" << stats.avg_degree
              << " max_degree=" << stats.max_degree
              << " search_results=" << results.size()
              << " self_found=" << (self_found ? 1 : 0) << std::endl;
}

int main() {
    constexpr size_t N = 1000;

    test_nbit_index<128, 1>(128, N);
    test_nbit_index<128, 2>(128, N);
    test_nbit_index<128, 4>(128, N);
    std::cout << "event=nbit_test_done cases=3" << std::endl;
    return 0;
}
