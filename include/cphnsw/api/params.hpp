#pragma once

#include "../core/constants.hpp"
#include <cstddef>
#include <cstdint>
#include <random>

namespace cphnsw {

struct IndexParams {
    size_t dim = 0;
    uint64_t rotation_seed = 0;  // 0 = random seed from std::random_device

    uint64_t effective_seed() const {
        if (rotation_seed != 0) return rotation_seed;
        std::random_device rd;
        return (static_cast<uint64_t>(rd()) << 32) | rd();
    }
};

struct SearchParams {
    size_t k = constants::kDefaultK;
    float recall_target = constants::kDefaultRecall;
};

}  // namespace cphnsw
