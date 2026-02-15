#pragma once

#include <cstddef>
#include <cstdint>

namespace cphnsw {

struct IndexParams {
    size_t dim = 0;
    uint64_t seed = 42;
};

struct BuildParams {
    size_t num_threads = 0;
    bool verbose = false;
};

struct SearchParams {
    size_t k = 10;
    float recall_target = 0.95f;
    float gamma_override = -1.0f;
};

}  // namespace cphnsw
