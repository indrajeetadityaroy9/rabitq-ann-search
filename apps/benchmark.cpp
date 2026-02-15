

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

inline size_t get_rss_kb() {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            std::istringstream iss(line.substr(6));
            size_t val;
            iss >> val;
            return val;
        }
    }
    return 0;
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

    IndexParams idx_params;
    idx_params.dim = sift.dim;
    RaBitQIndex<PADDED_DIM, 32> index(idx_params);
    Timer timer;

    timer.start();
    index.add_batch(sift.base_vectors.data(), sift.num_base);
    BuildParams build_params;
    build_params.verbose = true;
    index.finalize(build_params);
    double build_time_s = timer.elapsed_s();
    size_t rss_kb = get_rss_kb();

    printf("event=benchmark_build build_time_min=%.3f memory_gib=%.3f throughput_vec_s=%.0f\n",
           build_time_s / 60.0,
           static_cast<double>(rss_kb) / (1024.0 * 1024.0),
           index.size() / build_time_s);

    std::vector<float> recall_targets = {0.80f, 0.90f, 0.95f, 0.97f, 0.99f};
    for (float rt : recall_targets) {
        std::vector<double> latencies;
        double total_recall = 0.0;
        for (size_t q = 0; q < sift.num_queries; ++q) {
            timer.start();
            SearchParams search_params;
            search_params.k = 10;
            search_params.recall_target = rt;
            auto results = index.search(
                sift.query_vectors.data() + q * sift.dim, search_params);
            latencies.push_back(timer.elapsed_us());
            total_recall += compute_recall(results, sift.ground_truth[q], 10);
        }
        std::sort(latencies.begin(), latencies.end());
        double avg_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
        float gamma = adaptive_defaults::gamma_from_recall(rt, PADDED_DIM);
        float eps = adaptive_defaults::error_epsilon_search(rt);
        printf("event=benchmark_point recall_target=%.2f gamma=%.3f eps=%.2f recall_at_10=%.4f qps=%.0f p50_us=%.0f p99_us=%.0f\n",
               rt, gamma, eps, total_recall / sift.num_queries,
               1e6 / avg_latency,
               latencies[latencies.size() / 2],
               latencies[static_cast<size_t>(latencies.size() * 0.99)]);
    }
    return 0;
}
