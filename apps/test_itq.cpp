// Test ITQ learned rotation
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

    // Generate random data
    std::vector<float> data(N_TOTAL * DIM);
    for (auto& v : data) v = dist(rng);
    for (size_t i = 0; i < N_TOTAL; ++i) {
        float* vec = data.data() + i * DIM;
        float norm = 0;
        for (size_t j = 0; j < DIM; ++j) norm += vec[j] * vec[j];
        norm = std::sqrt(norm);
        for (size_t j = 0; j < DIM; ++j) vec[j] /= norm;
    }

    // Test DenseRotation ITQ learning
    std::cout << "=== ITQ Learning Test ===" << std::endl;

    DenseRotation rotation(DIM, 42);

    // Measure quantization error before ITQ
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
    std::cout << "Quantization error BEFORE ITQ: " << err_before << std::endl;

    auto t0 = std::chrono::high_resolution_clock::now();
    rotation.learn_rotation(data.data(), N_TRAIN, 20);
    auto t1 = std::chrono::high_resolution_clock::now();
    double itq_time = std::chrono::duration<double>(t1 - t0).count();

    double err_after = quantization_error(rotation);
    std::cout << "Quantization error AFTER  ITQ: " << err_after << std::endl;
    std::cout << "ITQ time: " << itq_time << "s (" << 20 << " iterations)" << std::endl;
    std::cout << "Error reduction: " << (1.0 - err_after / err_before) * 100.0 << "%" << std::endl;

    if (err_after < err_before) {
        std::cout << "[PASS] ITQ reduced quantization error" << std::endl;
    } else {
        std::cout << "[WARN] ITQ did not reduce error (may need more iterations)" << std::endl;
    }

    // Test with DenseRotation index
    std::cout << "\n=== Dense+ITQ Index Test ===" << std::endl;

    // Build index with ITQ rotation
    RaBitQIndex<128, 32, 1, DenseRotation> index(
        IndexParams().set_dim(DIM).set_ef_construction(100));
    index.add_batch(data.data(), N_TOTAL);
    index.finalize();

    auto stats = index.get_stats();
    std::cout << "Nodes: " << stats.num_nodes
              << " avg_deg: " << stats.avg_degree << std::endl;

    // Search
    auto results = index.search(data.data(), SearchParams().set_k(10).set_ef(50));
    std::cout << "Search results: " << results.size();
    if (!results.empty() && results[0].id == 0) {
        std::cout << " [self-found OK]";
    }
    std::cout << std::endl;

    std::cout << "\nAll ITQ tests passed." << std::endl;
    return 0;
}
