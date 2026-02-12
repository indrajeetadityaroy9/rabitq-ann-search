#include <cphnsw/encoder/dense_rotation.hpp>
#include <cphnsw/encoder/rabitq_encoder.hpp>
#include <cphnsw/api/rabitq_index.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>

using namespace cphnsw;

int main() {
    constexpr size_t DIM = 128;
    constexpr size_t N_TRAIN = 5000;
    constexpr size_t N_TOTAL = 10000;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> data(N_TOTAL * DIM);
    for (auto& v : data) v = dist(rng);
    for (size_t i = 0; i < N_TOTAL; ++i) {
        float* vec = data.data() + i * DIM;
        float norm = 0;
        for (size_t j = 0; j < DIM; ++j) norm += vec[j] * vec[j];
        norm = std::sqrt(norm);
        for (size_t j = 0; j < DIM; ++j) vec[j] /= norm;
    }

    DenseRotation rotation(DIM, 42);

    auto quantization_error = [&](const DenseRotation& rot) {
        double total_err = 0.0;
        std::vector<float> buf(DIM);
        for (size_t k = 0; k < N_TRAIN; ++k) {
            rot.apply_copy(data.data() + k * DIM, buf.data());
            for (size_t i = 0; i < DIM; ++i) {
                float sign = (buf[i] >= 0.0f) ? 1.0f : -1.0f;
                float err = buf[i] - sign;
                total_err += err * err;
            }
        }
        return total_err / N_TRAIN;
    };

    double err_before = quantization_error(rotation);

    auto t0 = std::chrono::high_resolution_clock::now();
    rotation.learn_rotation(data.data(), N_TRAIN, 20);
    auto t1 = std::chrono::high_resolution_clock::now();
    double itq_time = std::chrono::duration<double>(t1 - t0).count();

    double err_after = quantization_error(rotation);
    const bool itq_pass = err_after < err_before;
    std::cout << "event=itq_learning"
              << " train_size=" << N_TRAIN
              << " iterations=20"
              << " error_before=" << err_before
              << " error_after=" << err_after
              << " error_reduction_pct=" << ((1.0 - err_after / err_before) * 100.0)
              << " fit_time_s=" << itq_time
              << " pass=" << (itq_pass ? 1 : 0) << std::endl;

    RaBitQIndex<128, 32, 1, DenseRotation> index(IndexParams().set_dim(DIM));
    index.add_batch(data.data(), N_TOTAL);
    index.finalize();

    auto stats = index.get_stats();
    auto results = index.search(data.data(), SearchParams().set_k(10));
    const bool self_found = (!results.empty() && results[0].id == 0);
    std::cout << "event=itq_index"
              << " nodes=" << stats.num_nodes
              << " avg_degree=" << stats.avg_degree
              << " search_results=" << results.size()
              << " self_found=" << (self_found ? 1 : 0) << std::endl;
    std::cout << "event=itq_test_done pass=" << ((itq_pass && self_found) ? 1 : 0) << std::endl;
    return 0;
}
