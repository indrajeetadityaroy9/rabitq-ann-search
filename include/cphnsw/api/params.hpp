#pragma once

#include "../core/constants.hpp"
#include <cstddef>

namespace cphnsw {

struct IndexParams {
    size_t dim = 0;
};

struct SearchRequest {
    size_t k = constants::kDefaultK;
    float target_recall = constants::kDefaultRecall;
};

}  // namespace cphnsw
