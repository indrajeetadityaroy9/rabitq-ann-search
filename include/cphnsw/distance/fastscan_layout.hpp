#pragma once

#include "../core/codes.hpp"
#include <cstdint>
#include <cstring>

namespace cphnsw {

template <size_t D, size_t BatchSize = 32>
struct FastScanCodeBlock {
    static constexpr size_t DIMS = D;
    static constexpr size_t BATCH_SIZE = BatchSize;
    static constexpr size_t NUM_SUB_SEGMENTS = (D + 3) / 4;
    static constexpr size_t NUM_SUB_PAIRS = (NUM_SUB_SEGMENTS + 1) / 2;

    alignas(64) uint8_t packed[NUM_SUB_PAIRS][BatchSize];

    void store(size_t idx, const BinaryCodeStorage<D>& code) {
        for (size_t sp = 0; sp < NUM_SUB_PAIRS; ++sp) {
            size_t seg_lo = 2 * sp;
            size_t seg_hi = 2 * sp + 1;

            uint8_t lo = extract_sub_segment(code, seg_lo);
            uint8_t hi = (seg_hi < NUM_SUB_SEGMENTS)
                         ? extract_sub_segment(code, seg_hi)
                         : 0;

            packed[sp][idx] = static_cast<uint8_t>((hi << 4) | lo);
        }
    }

    void clear() {
        std::memset(packed, 0, sizeof(packed));
    }

private:
    static uint8_t extract_sub_segment(const BinaryCodeStorage<D>& code, size_t seg_idx) {
        size_t bit_start = seg_idx * 4;
        uint8_t result = 0;
        for (size_t b = 0; b < 4 && (bit_start + b) < D; ++b) {
            if (code.get_bit(bit_start + b)) {
                result |= (1 << b);
            }
        }
        return result;
    }

};

template <size_t D, size_t R = 32, size_t BatchSize = 32>
struct FastScanNeighborBlock {
    static_assert(R % BatchSize == 0,
                  "R must be a multiple of BatchSize for efficient SIMD");
    static constexpr size_t DEGREE = R;
    static constexpr size_t NUM_BATCHES = R / BatchSize;

    FastScanCodeBlock<D, BatchSize> code_blocks[NUM_BATCHES];
    alignas(64) VertexAuxData aux[R];
    alignas(64) uint16_t popcounts[R];
    alignas(64) uint32_t neighbor_ids[R];

    uint32_t count;

    FastScanNeighborBlock() : count(0) {
        std::memset(neighbor_ids, 0xFF, sizeof(neighbor_ids));
        std::memset(popcounts, 0, sizeof(popcounts));
    }

    void set_neighbor(size_t slot, uint32_t id,
                      const BinaryCodeStorage<D>& code,
                      const VertexAuxData& aux_data) {
        size_t batch = slot / BatchSize;
        size_t idx_in_batch = slot % BatchSize;

        neighbor_ids[slot] = id;
        code_blocks[batch].store(idx_in_batch, code);
        aux[slot] = aux_data;
        popcounts[slot] = static_cast<uint16_t>(code.popcount());

        if (slot >= count) count = static_cast<uint32_t>(slot + 1);
    }

    size_t size() const { return count; }
    bool empty() const { return count == 0; }
};

template <size_t D, size_t BitWidth, size_t BatchSize = 32>
struct NbitFastScanCodeBlock {
    static constexpr size_t BIT_WIDTH = BitWidth;
    static constexpr size_t DIMS = D;
    static constexpr size_t BATCH_SIZE = BatchSize;

    FastScanCodeBlock<D, BatchSize> planes[BitWidth];

    void store(size_t idx, const NbitCodeStorage<D, BitWidth>& code) {
        for (size_t b = 0; b < BitWidth; ++b) {
            BinaryCodeStorage<D> plane_binary;
            std::memcpy(plane_binary.signs, code.planes[b],
                        BinaryCodeStorage<D>::NUM_WORDS * sizeof(uint64_t));
            planes[b].store(idx, plane_binary);
        }
    }

    void clear() { for (auto& p : planes) p.clear(); }
};

template <size_t D, size_t R, size_t BitWidth, size_t BatchSize = 32>
struct NbitFastScanNeighborBlock {
    static_assert(R % BatchSize == 0);
    static constexpr size_t DEGREE = R;
    static constexpr size_t NUM_BATCHES = R / BatchSize;
    static constexpr size_t BIT_WIDTH = BitWidth;

    NbitFastScanCodeBlock<D, BitWidth, BatchSize> code_blocks[NUM_BATCHES];
    alignas(64) VertexAuxData aux[R];
    alignas(64) uint16_t popcounts[R];
    alignas(64) uint16_t weighted_popcounts[R];
    alignas(64) uint32_t neighbor_ids[R];
    uint32_t count;

    NbitFastScanNeighborBlock() : count(0) {
        std::memset(neighbor_ids, 0xFF, sizeof(neighbor_ids));
        std::memset(popcounts, 0, sizeof(popcounts));
        std::memset(weighted_popcounts, 0, sizeof(weighted_popcounts));
    }

    void set_neighbor(size_t slot, uint32_t id,
                      const NbitCodeStorage<D, BitWidth>& code,
                      const VertexAuxData& aux_data) {
        size_t batch = slot / BatchSize;
        size_t idx_in_batch = slot % BatchSize;
        neighbor_ids[slot] = id;
        code_blocks[batch].store(idx_in_batch, code);
        aux[slot] = aux_data;
        popcounts[slot] = static_cast<uint16_t>(code.msb_popcount());
        weighted_popcounts[slot] = static_cast<uint16_t>(code.weighted_popcount());
        if (slot >= count) count = static_cast<uint32_t>(slot + 1);
    }

    size_t size() const { return count; }
    bool empty() const { return count == 0; }
};

}  // namespace cphnsw