#include <cphnsw/api/hnsw_index.hpp>
#include <cphnsw/api/rabitq_index.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>

using namespace cphnsw;

int main() {
    constexpr size_t DIM = 128;
    constexpr size_t N = 10000;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> data(N * DIM);
    for (auto& v : data) v = dist(rng);
    for (size_t i = 0; i < N; ++i) {
        float* vec = data.data() + i * DIM;
        float norm = 0;
        for (size_t j = 0; j < DIM; ++j) norm += vec[j] * vec[j];
        norm = std::sqrt(norm);
        for (size_t j = 0; j < DIM; ++j) vec[j] /= norm;
    }

    HNSWIndex<128, 32, 1> index(IndexParams().set_dim(DIM));

    auto t0 = std::chrono::high_resolution_clock::now();
    index.add_batch(data.data(), N);
    index.finalize(BuildParams().set_verbose(true));
    auto t1 = std::chrono::high_resolution_clock::now();
    double build_time = std::chrono::duration<double>(t1 - t0).count();

    auto stats = index.get_stats();
    std::cout << "event=hnsw_test_build"
              << " build_time_s=" << build_time
              << " nodes=" << stats.num_nodes
              << " avg_degree=" << stats.avg_degree
              << " max_level=" << stats.max_level
              << " above_layer0=" << stats.nodes_above_layer0 << std::endl;

    size_t self_found = 0;
    size_t queries = std::min<size_t>(100, N);

    auto t2 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < queries; ++i) {
        auto results = index.search(data.data() + i * DIM,
                                    SearchParams().set_k(10));
        if (!results.empty() && results[0].id == static_cast<uint32_t>(i)
            && results[0].distance < 1e-4f) {
            ++self_found;
        }
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double search_time = std::chrono::duration<double>(t3 - t2).count();

    std::cout << "event=hnsw_test_search"
              << " self_found=" << self_found
              << " queries=" << queries
              << " search_time_s=" << search_time
              << " qps=" << (queries / search_time) << std::endl;

    RaBitQIndex<128, 32, 1> flat_index(IndexParams().set_dim(DIM));

    auto t4 = std::chrono::high_resolution_clock::now();
    flat_index.add_batch(data.data(), N);
    flat_index.finalize();
    auto t5 = std::chrono::high_resolution_clock::now();
    double flat_build_time = std::chrono::duration<double>(t5 - t4).count();

    size_t flat_self_found = 0;
    auto t6 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < queries; ++i) {
        auto results = flat_index.search(data.data() + i * DIM,
                                         SearchParams().set_k(10));
        if (!results.empty() && results[0].id == static_cast<uint32_t>(i)
            && results[0].distance < 1e-4f) {
            ++flat_self_found;
        }
    }
    auto t7 = std::chrono::high_resolution_clock::now();
    double flat_search_time = std::chrono::duration<double>(t7 - t6).count();

    std::cout << "event=hnsw_test_flat"
              << " build_time_s=" << flat_build_time
              << " self_found=" << flat_self_found
              << " queries=" << queries
              << " search_time_s=" << flat_search_time
              << " qps=" << (queries / flat_search_time) << std::endl;

    const bool pass = self_found >= queries * 0.9;
    std::cout << "event=hnsw_test_result pass=" << (pass ? 1 : 0)
              << " threshold=0.9 self_found=" << self_found
              << " queries=" << queries << std::endl;
    return 0;
}
