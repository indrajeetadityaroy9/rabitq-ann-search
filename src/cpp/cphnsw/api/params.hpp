#pragma once

#include <cstddef>
#include <cstdint>

namespace cphnsw {

// ============================================================================
// Index Configuration
// ============================================================================

struct IndexParams {
    size_t dim = 0;
    size_t M = 32;
    size_t ef_construction = 200;
    uint64_t seed = 42;
    size_t initial_capacity = 1024;

    IndexParams& set_dim(size_t d) { dim = d; return *this; }
    IndexParams& set_M(size_t m) { M = m; return *this; }
    IndexParams& set_ef_construction(size_t ef) { ef_construction = ef; return *this; }
    IndexParams& set_seed(uint64_t s) { seed = s; return *this; }
    IndexParams& set_capacity(size_t c) { initial_capacity = c; return *this; }
};

struct BuildParams {
    size_t num_threads = 0;
    bool verbose = false;

    BuildParams& set_threads(size_t n) { num_threads = n; return *this; }
    BuildParams& set_verbose(bool v) { verbose = v; return *this; }
};

struct SearchParams {
    size_t k = 10;
    size_t ef = 100;

    SearchParams& set_k(size_t num) { k = num; return *this; }
    SearchParams& set_ef(size_t e) { ef = e; return *this; }
};

}  // namespace cphnsw
