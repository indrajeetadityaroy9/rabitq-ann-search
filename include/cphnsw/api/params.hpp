#pragma once

#include <cstddef>

namespace cphnsw {

struct IndexParams {
    size_t dim = 0;
};

struct SearchParams {
    size_t k = 10;
    float recall_target = 0.95f;
};

}  // namespace cphnsw
