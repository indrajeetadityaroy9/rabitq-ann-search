#pragma once

#include <cstddef>
#include <cstdint>

namespace cphnsw {

struct IndexParams {
    size_t dim = 0;
    uint64_t seed = 42;

    IndexParams& set_dim(size_t d) { dim = d; return *this; }
    IndexParams& set_seed(uint64_t s) { seed = s; return *this; }
};

struct BuildParams {
    size_t num_threads = 0;      // 0 = auto-detect via omp_get_max_threads()
    bool verbose = false;

    BuildParams& set_threads(size_t n) { num_threads = n; return *this; }
    BuildParams& set_verbose(bool v) { verbose = v; return *this; }
};

struct SearchParams {
    size_t k = 10;
    float recall_target = 0.95f;

    SearchParams& set_k(size_t num) { k = num; return *this; }
    SearchParams& set_recall_target(float r) { recall_target = r; return *this; }
};

}  // namespace cphnsw
