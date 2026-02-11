/**
 * Master Evaluation Protocol for CP-HNSW PhD Portfolio
 */

#include <cphnsw/api/rabitq_index.hpp>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <thread>
#include <unordered_set>
#include <omp.h>
#include <sys/resource.h>

using namespace cphnsw;

#ifndef EVAL_D
#define EVAL_D 128
#endif
constexpr size_t PADDED_DIM = EVAL_D;

struct Dataset {
    std::vector<float> base_vectors;
    std::vector<float> query_vectors;
    std::vector<std::vector<NodeId>> ground_truth;
    size_t dim;
    size_t num_base;
    size_t num_queries;
    size_t k_gt;
};

inline std::vector<float> load_fvecs(const std::string& path, size_t& dim, size_t& count) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open file: " + path);
    int32_t d;
    file.read(reinterpret_cast<char*>(&d), sizeof(int32_t));
    dim = static_cast<size_t>(d);
    file.seekg(0, std::ios::end);
    size_t file_size = static_cast<size_t>(file.tellg());
    size_t record_size = sizeof(int32_t) + dim * sizeof(float);
    count = file_size / record_size;
    std::vector<float> data(count * dim);
    file.seekg(0, std::ios::beg);
    for (size_t i = 0; i < count; ++i) {
        int32_t d_check;
        file.read(reinterpret_cast<char*>(&d_check), sizeof(int32_t));
        file.read(reinterpret_cast<char*>(data.data() + i * dim), dim * sizeof(float));
    }
    return data;
}

inline std::vector<std::vector<NodeId>> load_ivecs(const std::string& path, size_t& k, size_t& count) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open file: " + path);
    int32_t k_val;
    file.read(reinterpret_cast<char*>(&k_val), sizeof(int32_t));
    k = static_cast<size_t>(k_val);
    file.seekg(0, std::ios::end);
    size_t file_size = static_cast<size_t>(file.tellg());
    size_t record_size = sizeof(int32_t) + k * sizeof(int32_t);
    count = file_size / record_size;
    std::vector<std::vector<NodeId>> data(count);
    std::vector<int32_t> buffer(k);
    file.seekg(0, std::ios::beg);
    for (size_t i = 0; i < count; ++i) {
        int32_t k_check;
        file.read(reinterpret_cast<char*>(&k_check), sizeof(int32_t));
        file.read(reinterpret_cast<char*>(buffer.data()), k * sizeof(int32_t));
        data[i].resize(k);
        for (size_t j = 0; j < k; ++j) data[i][j] = static_cast<NodeId>(buffer[j]);
    }
    return data;
}

inline void normalize_vectors(float* vecs, size_t count, size_t dim) {
    for (size_t i = 0; i < count; ++i) {
        float* v = vecs + i * dim;
        float norm_sq = 0;
        for (size_t d = 0; d < dim; ++d) norm_sq += v[d] * v[d];
        float norm = std::sqrt(norm_sq);
        if (norm > 1e-10f) {
            float inv_norm = 1.0f / norm;
            for (size_t d = 0; d < dim; ++d) v[d] *= inv_norm;
        }
    }
}

inline Dataset load_sift1m(const std::string& dir) {
    Dataset ds;
    size_t base_dim, base_count;
    ds.base_vectors = load_fvecs(dir + "/sift_base.fvecs", base_dim, base_count);
    ds.dim = base_dim;
    ds.num_base = base_count;
    size_t query_dim, query_count;
    ds.query_vectors = load_fvecs(dir + "/sift_query.fvecs", query_dim, query_count);
    ds.num_queries = query_count;
    size_t k_gt, gt_count;
    ds.ground_truth = load_ivecs(dir + "/sift_groundtruth.ivecs", k_gt, gt_count);
    ds.k_gt = k_gt;
    normalize_vectors(ds.base_vectors.data(), ds.num_base, ds.dim);
    normalize_vectors(ds.query_vectors.data(), ds.num_queries, ds.dim);
    return ds;
}

inline double compute_recall(const std::vector<SearchResult>& results, const std::vector<NodeId>& ground_truth, size_t k) {
    std::unordered_set<NodeId> gt_set(ground_truth.begin(), ground_truth.begin() + std::min(k, ground_truth.size()));
    size_t hits = 0;
    for (size_t i = 0; i < std::min(k, results.size()); ++i) {
        if (gt_set.count(results[i].id)) ++hits;
    }
    return static_cast<double>(hits) / static_cast<double>(gt_set.size());
}

class Timer {
    std::chrono::high_resolution_clock::time_point start_;
public:
    void start() { start_ = std::chrono::high_resolution_clock::now(); }
    double elapsed_s() const {
        return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_).count();
    }
    double elapsed_us() const { return elapsed_s() * 1e6; }
};

int main(int argc, char** argv) {
    std::string sift_dir = (argc > 1) ? argv[1] : "data/sift";
    Dataset sift = load_sift1m(sift_dir);
    
    RaBitQIndex<PADDED_DIM, 32> index(IndexParams().set_dim(sift.dim).set_ef_construction(200));
    Timer timer;
    timer.start();
    index.add_batch(sift.base_vectors.data(), sift.num_base);
    index.finalize(BuildParams().set_verbose(true));
    std::cout << "Build time: " << index.size() / timer.elapsed_s() << " vec/s\n";

    std::vector<size_t> ef_values = {10, 20, 40, 80, 100, 200, 400};
    for (size_t ef : ef_values) {
        std::vector<double> latencies;
        double total_recall = 0.0;
        for (size_t q = 0; q < sift.num_queries; ++q) {
            timer.start();
            auto results = index.search(sift.query_vectors.data() + q * sift.dim, SearchParams().set_k(10).set_ef(ef));
            latencies.push_back(timer.elapsed_us());
            total_recall += compute_recall(results, sift.ground_truth[q], 10);
        }
        std::sort(latencies.begin(), latencies.end());
        std::cout << "ef=" << ef << " Recall=" << total_recall / sift.num_queries << " QPS=" << 1e6 / (std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size()) << "\n";
    }
    return 0;
}
