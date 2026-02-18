#pragma once

#include <cstddef>

namespace cphnsw {

inline size_t next_power_of_two(size_t n) {
    size_t p = 1;
    while (p < n) p *= 2;
    return p;
}

}  // namespace cphnsw
