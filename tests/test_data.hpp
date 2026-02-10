#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace cphnsw {
namespace test_data {

// ── fvecs / ivecs binary format loaders ──
// Format: each record is [int32 dim][dim × element] back-to-back.
// Returns a flat contiguous buffer + metadata (n, dim).

struct FvecsData {
    std::vector<float> data;   // flat: n × dim
    size_t n   = 0;
    size_t dim = 0;

    const float* row(size_t i) const { return data.data() + i * dim; }
};

struct IvecsData {
    std::vector<int32_t> data; // flat: n × k
    size_t n = 0;
    size_t k = 0;

    const int32_t* row(size_t i) const { return data.data() + i * k; }
};

inline FvecsData load_fvecs(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open fvecs file: " + path);

    // Read dimension from first record
    int32_t dim;
    in.read(reinterpret_cast<char*>(&dim), 4);
    if (!in || dim <= 0) throw std::runtime_error("Invalid fvecs header: " + path);

    // Determine file size to compute number of vectors
    in.seekg(0, std::ios::end);
    auto file_size = in.tellg();
    in.seekg(0, std::ios::beg);

    size_t record_bytes = 4 + static_cast<size_t>(dim) * sizeof(float);
    if (file_size % record_bytes != 0)
        throw std::runtime_error("Corrupt fvecs file (size not multiple of record): " + path);

    size_t n = static_cast<size_t>(file_size) / record_bytes;

    FvecsData result;
    result.n   = n;
    result.dim = static_cast<size_t>(dim);
    result.data.resize(n * result.dim);

    for (size_t i = 0; i < n; ++i) {
        int32_t d;
        in.read(reinterpret_cast<char*>(&d), 4);
        if (d != dim)
            throw std::runtime_error("Inconsistent dimension in fvecs at vector " + std::to_string(i));
        in.read(reinterpret_cast<char*>(result.data.data() + i * result.dim),
                result.dim * sizeof(float));
    }

    if (!in) throw std::runtime_error("Premature EOF reading fvecs: " + path);
    return result;
}

inline IvecsData load_ivecs(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open ivecs file: " + path);

    int32_t k;
    in.read(reinterpret_cast<char*>(&k), 4);
    if (!in || k <= 0) throw std::runtime_error("Invalid ivecs header: " + path);

    in.seekg(0, std::ios::end);
    auto file_size = in.tellg();
    in.seekg(0, std::ios::beg);

    size_t record_bytes = 4 + static_cast<size_t>(k) * sizeof(int32_t);
    if (file_size % record_bytes != 0)
        throw std::runtime_error("Corrupt ivecs file (size not multiple of record): " + path);

    size_t n = static_cast<size_t>(file_size) / record_bytes;

    IvecsData result;
    result.n = n;
    result.k = static_cast<size_t>(k);
    result.data.resize(n * result.k);

    for (size_t i = 0; i < n; ++i) {
        int32_t d;
        in.read(reinterpret_cast<char*>(&d), 4);
        if (d != k)
            throw std::runtime_error("Inconsistent k in ivecs at vector " + std::to_string(i));
        in.read(reinterpret_cast<char*>(result.data.data() + i * result.k),
                result.k * sizeof(int32_t));
    }

    if (!in) throw std::runtime_error("Premature EOF reading ivecs: " + path);
    return result;
}

// ── SIFT dataset descriptor ──

struct SIFTDataset {
    FvecsData base;         // database vectors (10K or 1M × 128)
    FvecsData queries;      // query vectors (100 or 10K × 128)
    IvecsData groundtruth;  // ground-truth k-NN indices

    size_t dim() const { return base.dim; }
    size_t num_base() const { return base.n; }
    size_t num_queries() const { return queries.n; }
    size_t gt_k() const { return groundtruth.k; }

    // Compute recall@k: fraction of queries whose true NN appears in result top-k.
    // `results` is (num_queries × result_k) flat array of retrieved IDs.
    float compute_recall(const std::vector<std::vector<uint32_t>>& results, size_t k) const {
        if (results.size() != queries.n) return 0.0f;
        size_t hits = 0;
        for (size_t q = 0; q < queries.n; ++q) {
            size_t check_k = std::min(k, results[q].size());
            size_t gt_check = std::min(k, groundtruth.k);
            for (size_t i = 0; i < check_k; ++i) {
                for (size_t j = 0; j < gt_check; ++j) {
                    if (static_cast<int32_t>(results[q][i]) == groundtruth.row(q)[j]) {
                        ++hits;
                        break;
                    }
                }
            }
        }
        return static_cast<float>(hits) / static_cast<float>(queries.n * k);
    }
};

// ── Dataset loaders ──
// Follows the same file layout as the Python datasets.py:
//   data/sift1m/{sift_base.fvecs, sift_query.fvecs, sift_groundtruth.ivecs}
//   data/siftsmall/{siftsmall_base.fvecs, siftsmall_query.fvecs, siftsmall_groundtruth.ivecs}

inline SIFTDataset load_sift1m(const std::string& base_dir = "data/sift1m") {
    SIFTDataset ds;
    ds.base        = load_fvecs(base_dir + "/sift_base.fvecs");
    ds.queries     = load_fvecs(base_dir + "/sift_query.fvecs");
    ds.groundtruth = load_ivecs(base_dir + "/sift_groundtruth.ivecs");
    return ds;
}

inline SIFTDataset load_sift10k(const std::string& base_dir = "data/siftsmall") {
    SIFTDataset ds;
    ds.base        = load_fvecs(base_dir + "/siftsmall_base.fvecs");
    ds.queries     = load_fvecs(base_dir + "/siftsmall_query.fvecs");
    ds.groundtruth = load_ivecs(base_dir + "/siftsmall_groundtruth.ivecs");
    return ds;
}

inline SIFTDataset load_gist1m(const std::string& base_dir = "data/gist1m") {
    SIFTDataset ds;
    ds.base        = load_fvecs(base_dir + "/gist_base.fvecs");
    ds.queries     = load_fvecs(base_dir + "/gist_query.fvecs");
    ds.groundtruth = load_ivecs(base_dir + "/gist_groundtruth.ivecs");
    return ds;
}

// Check whether a dataset directory exists and contains the expected files.
inline bool sift10k_available(const std::string& base_dir = "data/siftsmall") {
    std::ifstream f(base_dir + "/siftsmall_base.fvecs", std::ios::binary);
    return f.good();
}

inline bool sift1m_available(const std::string& base_dir = "data/sift1m") {
    std::ifstream f(base_dir + "/sift_base.fvecs", std::ios::binary);
    return f.good();
}

}  // namespace test_data
}  // namespace cphnsw
