#pragma once

#include "../core/codes.hpp"
#include <cstdint>
#include <cstring>

namespace cphnsw {

// ============================================================================
// FastScan Code Layout: Nibble-interleaved packing for vpshufb SIMD
// ============================================================================

/**
 * FastScanCodeBlock: Stores BatchSize binary codes in nibble-interleaved
 * format optimized for AVX2 vpshufb-based distance computation.
 *
 * PACKING FORMAT:
 * The D-dimensional binary code is split into D/4 sub-segments of 4 bits each.
 * Adjacent sub-segment pairs are packed into nibbles (lo/hi) of a byte:
 *   byte = (sub_segment[2k+1] << 4) | sub_segment[2k]
 *
 * Each sub-pair has BatchSize bytes (one byte per code in the batch).
 * This layout enables vpshufb to process 32 codes simultaneously:
 *   - Load 32 bytes (one sub-pair across all 32 codes) into ymm
 *   - AND with 0x0F → lo nibbles (sub-segment 2k)
 *   - SHR 4, AND 0x0F → hi nibbles (sub-segment 2k+1)
 *   - vpshufb with LUT → 32 parallel lookups per sub-segment
 *
 * @tparam D Number of dimensions (bits) per code
 * @tparam BatchSize Number of codes per block (32 for AVX2, 64 for AVX-512)
 */
template <size_t D, size_t BatchSize = 32>
struct FastScanCodeBlock {
    static constexpr size_t DIMS = D;
    static constexpr size_t BATCH_SIZE = BatchSize;
    static constexpr size_t NUM_SUB_SEGMENTS = (D + 3) / 4;
    // Sub-segments are paired: (0,1), (2,3), ...
    // If odd number of sub-segments, last one is unpaired (hi nibble = 0)
    static constexpr size_t NUM_SUB_PAIRS = (NUM_SUB_SEGMENTS + 1) / 2;

    // Nibble-interleaved packed codes
    // packed[pair_idx][code_idx] has lo=sub_segment[2*pair_idx], hi=sub_segment[2*pair_idx+1]
    alignas(64) uint8_t packed[NUM_SUB_PAIRS][BatchSize];

    /**
     * Store a binary code at position idx in the block.
     *
     * Extracts 4-bit sub-segments from the code and packs them
     * into the nibble-interleaved layout.
     *
     * @param idx Position in the batch (0 to BatchSize-1)
     * @param code The D-bit binary code to store
     */
    void store(size_t idx, const BinaryCodeStorage<D>& code) {
        for (size_t sp = 0; sp < NUM_SUB_PAIRS; ++sp) {
            size_t seg_lo = 2 * sp;
            size_t seg_hi = 2 * sp + 1;

            // Extract 4-bit sub-segment from the binary code
            uint8_t lo = extract_sub_segment(code, seg_lo);
            uint8_t hi = (seg_hi < NUM_SUB_SEGMENTS)
                         ? extract_sub_segment(code, seg_hi)
                         : 0;

            packed[sp][idx] = static_cast<uint8_t>((hi << 4) | lo);
        }
    }

    /**
     * Load a binary code from position idx in the block.
     *
     * Unpacks nibble-interleaved data back into a BinaryCodeStorage.
     *
     * @param idx Position in the batch
     * @param code Output: the reconstructed D-bit code
     */
    void load(size_t idx, BinaryCodeStorage<D>& code) const {
        code.clear();
        for (size_t sp = 0; sp < NUM_SUB_PAIRS; ++sp) {
            size_t seg_lo = 2 * sp;
            size_t seg_hi = 2 * sp + 1;

            uint8_t byte = packed[sp][idx];
            uint8_t lo = byte & 0x0F;
            uint8_t hi = (byte >> 4) & 0x0F;

            inject_sub_segment(code, seg_lo, lo);
            if (seg_hi < NUM_SUB_SEGMENTS) {
                inject_sub_segment(code, seg_hi, hi);
            }
        }
    }

    /**
     * Clear the entire block.
     */
    void clear() {
        std::memset(packed, 0, sizeof(packed));
    }

private:
    /**
     * Extract a 4-bit sub-segment from a binary code.
     *
     * Sub-segment j corresponds to bits [4*j, 4*j+3] of the code.
     * Returns the 4 bits packed into the low nibble of a uint8_t.
     */
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

    /**
     * Inject a 4-bit sub-segment into a binary code.
     */
    static void inject_sub_segment(BinaryCodeStorage<D>& code, size_t seg_idx, uint8_t nibble) {
        size_t bit_start = seg_idx * 4;
        for (size_t b = 0; b < 4 && (bit_start + b) < D; ++b) {
            code.set_bit(bit_start + b, (nibble >> b) & 1);
        }
    }
};

// ============================================================================
// FastScan Neighbor Block: Codes + Auxiliary Data for Graph Edges
// ============================================================================

/**
 * FastScanNeighborBlock: Stores R neighbor codes in FastScan layout
 * along with per-neighbor auxiliary data for RaBitQ distance estimation.
 *
 * In SymphonyQG's design, each vertex stores:
 *   - FastScan-packed codes of all R neighbors
 *   - Per-neighbor auxiliary scalars (norms, inner products)
 *   - Neighbor IDs
 *
 * R must be a multiple of BatchSize for efficient SIMD processing
 * (no partial batches).
 *
 * @tparam D Number of dimensions per code
 * @tparam R Number of neighbors (must be multiple of BatchSize)
 * @tparam BatchSize SIMD batch size (32 for AVX2)
 */
template <size_t D, size_t R = 32, size_t BatchSize = 32>
struct FastScanNeighborBlock {
    static_assert(R % BatchSize == 0,
                  "R must be a multiple of BatchSize for efficient SIMD");
    static constexpr size_t DEGREE = R;
    static constexpr size_t NUM_BATCHES = R / BatchSize;

    // FastScan-packed neighbor codes (R/BatchSize blocks)
    FastScanCodeBlock<D, BatchSize> code_blocks[NUM_BATCHES];

    // Per-neighbor auxiliary data
    alignas(64) VertexAuxData aux[R];

    // Pre-computed popcount of each neighbor's code
    alignas(64) uint16_t popcounts[R];

    // Neighbor IDs
    alignas(64) uint32_t neighbor_ids[R];

    // Actual neighbor count (may be < R during construction, == R after refinement)
    uint32_t count;

    FastScanNeighborBlock() : count(0) {
        std::memset(neighbor_ids, 0xFF, sizeof(neighbor_ids)); // INVALID_NODE
        std::memset(popcounts, 0, sizeof(popcounts));
    }

    /**
     * Set a neighbor at the given slot.
     *
     * @param slot Index in [0, R)
     * @param id Neighbor node ID
     * @param code Neighbor's binary code
     * @param aux_data Neighbor's auxiliary data
     */
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

// Type aliases for common configurations
using FastScanBlock128 = FastScanCodeBlock<128, 32>;
using FastScanBlock256 = FastScanCodeBlock<256, 32>;
using FastScanBlock1024 = FastScanCodeBlock<1024, 32>;

using FastScanNeighbors128_32 = FastScanNeighborBlock<128, 32, 32>;
using FastScanNeighbors256_32 = FastScanNeighborBlock<256, 32, 32>;
using FastScanNeighbors1024_32 = FastScanNeighborBlock<1024, 32, 32>;

}  // namespace cphnsw
