#pragma once

#include <cstddef>
#include <cstdint>

namespace cphnsw {

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
    size_t ef_construction = 200;
    float alpha = 1.0f;
    float alpha_pass2 = 1.2f;
    bool enable_pass2 = true;
    bool quantized_construction = true;
    bool auto_degree = false;
    bool fill_slots = true;
    float error_tolerance = 0.0f;  // QRG: per-vector error bounds for construction
    bool verbose = false;

    BuildParams& set_threads(size_t n) { num_threads = n; return *this; }
    BuildParams& set_ef_construction(size_t ef) { ef_construction = ef; return *this; }
    BuildParams& set_alpha(float a) { alpha = a; return *this; }
    BuildParams& set_alpha_pass2(float a) { alpha_pass2 = a; return *this; }
    BuildParams& set_enable_pass2(bool v) { enable_pass2 = v; return *this; }
    BuildParams& set_quantized(bool v) { quantized_construction = v; return *this; }
    BuildParams& set_auto_degree(bool v) { auto_degree = v; return *this; }
    BuildParams& set_fill_slots(bool v) { fill_slots = v; return *this; }
    BuildParams& set_error_tolerance(float t) { error_tolerance = t; return *this; }
    BuildParams& set_verbose(bool v) { verbose = v; return *this; }
};

struct SearchParams {
    size_t k = 10;
    size_t ef = 100;
    float error_epsilon = 1.9f;

    SearchParams& set_k(size_t num) { k = num; return *this; }
    SearchParams& set_ef(size_t e) { ef = e; return *this; }
    SearchParams& set_error_epsilon(float e) { error_epsilon = e; return *this; }
};

}  // namespace cphnsw