#pragma once

#include <cstddef>
#include <cstdint>

namespace cphnsw {

struct IndexParams {
    size_t dim = 0;
    size_t ef_construction = 200;
    uint64_t seed = 42;

    IndexParams& set_dim(size_t d) { dim = d; return *this; }
    IndexParams& set_ef_construction(size_t ef) { ef_construction = ef; return *this; }
    IndexParams& set_seed(uint64_t s) { seed = s; return *this; }
};

struct BuildParams {
    size_t num_threads = 0;            // 0 = auto-detect via omp_get_max_threads()
    size_t ef_construction = 0;        // 0 = auto from N via AdaptiveDefaults
    float error_tolerance = -1.0f;     // -1 = auto from D via AdaptiveDefaults
    float error_epsilon = 0.0f;        // 0 = auto via AdaptiveDefaults
    bool verbose = false;

    BuildParams& set_threads(size_t n) { num_threads = n; return *this; }
    BuildParams& set_ef_construction(size_t ef) { ef_construction = ef; return *this; }
    BuildParams& set_error_tolerance(float t) { error_tolerance = t; return *this; }
    BuildParams& set_error_epsilon(float e) { error_epsilon = e; return *this; }
    BuildParams& set_verbose(bool v) { verbose = v; return *this; }
};

struct SearchParams {
    size_t k = 10;
    size_t ef = 0;                     // 0 = auto from k and recall_target
    float error_epsilon = 0.0f;        // 0 = auto from recall_target
    float recall_target = 0.95f;

    SearchParams& set_k(size_t num) { k = num; return *this; }
    SearchParams& set_ef(size_t e) { ef = e; return *this; }
    SearchParams& set_error_epsilon(float e) { error_epsilon = e; return *this; }
    SearchParams& set_recall_target(float r) { recall_target = r; return *this; }
};

}  // namespace cphnsw
