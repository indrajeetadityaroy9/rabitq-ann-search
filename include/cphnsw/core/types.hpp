#pragma once

#include <cstdint>
#include <vector>
#include <queue>

namespace cphnsw {

using NodeId = uint32_t;
constexpr NodeId INVALID_NODE = 0xFFFFFFFF;

struct SearchResult {
    NodeId id;
    float distance;

    bool operator<(const SearchResult& other) const {
        return distance < other.distance;
    }

    bool operator>(const SearchResult& other) const {
        return distance > other.distance;
    }
};

using MinHeap = std::priority_queue<SearchResult, std::vector<SearchResult>,
                                     std::greater<SearchResult>>;

using MaxHeap = std::priority_queue<SearchResult, std::vector<SearchResult>,
                                     std::less<SearchResult>>;

}  // namespace cphnsw
