#pragma once

#include <cstdint>
#include <cstring>

namespace cphnsw {

template <size_t Bits>
struct alignas(64) BinaryCodeStorage {
    static constexpr size_t NUM_BITS = Bits;
    static constexpr size_t NUM_WORDS = (Bits + 63) / 64;

    uint64_t signs[NUM_WORDS];

    void clear() {
        for (size_t i = 0; i < NUM_WORDS; ++i) {
            signs[i] = 0;
        }
    }

    void set_bit(size_t idx, bool value) {
        if (value) {
            signs[idx / 64] |= (1ULL << (idx % 64));
        } else {
            signs[idx / 64] &= ~(1ULL << (idx % 64));
        }
    }

    bool get_bit(size_t idx) const {
        return (signs[idx / 64] >> (idx % 64)) & 1;
    }

    bool operator==(const BinaryCodeStorage& other) const {
        for (size_t i = 0; i < NUM_WORDS; ++i) {
            if (signs[i] != other.signs[i]) return false;
        }
        return true;
    }

    uint32_t popcount() const {
        uint32_t count = 0;
        for (size_t i = 0; i < NUM_WORDS; ++i) {
            count += static_cast<uint32_t>(__builtin_popcountll(signs[i]));
        }
        return count;
    }
};

template <size_t D>
struct RaBitQCode {
    static constexpr size_t DIMS = D;
    static constexpr size_t NUM_WORDS = (D + 63) / 64;

    BinaryCodeStorage<D> signs;
    float dist_to_centroid;
    float ip_quantized_original; // <o_bar, o> = ||P^-1 * o||_1 / sqrt(D)
    uint16_t code_popcount;

    void clear() {
        signs.clear();
        dist_to_centroid = 0.0f;
        ip_quantized_original = 0.0f;
        code_popcount = 0;
    }
};

struct VertexAuxData {
    float dist_to_centroid;
    float ip_quantized_original;
    float ip_xbar_Pinv_c; // <x_bar, P^-1 * c>, precomputed for SymphonyQG decomposition
};

template <size_t D>
struct RaBitQQuery {
    static constexpr size_t DIMS = D;
    static constexpr size_t NUM_SUB_SEGMENTS = (D + 3) / 4;

    alignas(64) uint8_t lut[NUM_SUB_SEGMENTS][16];

    float vl;
    float delta;
    float sum_qu;

    float query_norm;
    float query_norm_sq;

    float coeff_fastscan;
    float coeff_popcount;
    float coeff_constant;

    float error_epsilon;
    float inv_sqrt_d;
};

using RaBitQCode128 = RaBitQCode<128>;
using RaBitQCode256 = RaBitQCode<256>;
using RaBitQCode512 = RaBitQCode<512>;
using RaBitQCode1024 = RaBitQCode<1024>;

using RaBitQQuery128 = RaBitQQuery<128>;
using RaBitQQuery256 = RaBitQQuery<256>;
using RaBitQQuery512 = RaBitQQuery<512>;
using RaBitQQuery1024 = RaBitQQuery<1024>;

// === Extended RaBitQ: Multi-bit Code Storage (SIGMOD'25) ===
// Bit-plane layout enables reuse of 1-bit FastScan kernel per plane.

template <size_t D, size_t BitWidth>
struct alignas(64) NbitCodeStorage {
    static_assert(BitWidth >= 1 && BitWidth <= 8, "BitWidth must be 1-8");
    static constexpr size_t NUM_BITS = D;
    static constexpr size_t BIT_WIDTH = BitWidth;
    static constexpr size_t NUM_WORDS = (D + 63) / 64;

    // planes[0] = MSB, planes[BitWidth-1] = LSB
    uint64_t planes[BitWidth][NUM_WORDS];

    void clear() { std::memset(planes, 0, sizeof(planes)); }

    void set_value(size_t idx, uint8_t value) {
        size_t word = idx / 64;
        size_t bit = idx % 64;
        for (size_t b = 0; b < BitWidth; ++b) {
            if ((value >> (BitWidth - 1 - b)) & 1)
                planes[b][word] |= (1ULL << bit);
            else
                planes[b][word] &= ~(1ULL << bit);
        }
    }

    uint8_t get_value(size_t idx) const {
        size_t word = idx / 64;
        size_t bit = idx % 64;
        uint8_t value = 0;
        for (size_t b = 0; b < BitWidth; ++b) {
            if ((planes[b][word] >> bit) & 1)
                value |= (1 << (BitWidth - 1 - b));
        }
        return value;
    }

    uint32_t msb_popcount() const {
        uint32_t count = 0;
        for (size_t i = 0; i < NUM_WORDS; ++i)
            count += static_cast<uint32_t>(__builtin_popcountll(planes[0][i]));
        return count;
    }

    BinaryCodeStorage<D> msb_as_binary() const {
        BinaryCodeStorage<D> result;
        std::memcpy(result.signs, planes[0], NUM_WORDS * sizeof(uint64_t));
        return result;
    }

    // sum_b 2^(B-1-b) * popcount(plane_b)
    uint32_t weighted_popcount() const {
        uint32_t total = 0;
        for (size_t b = 0; b < BitWidth; ++b) {
            uint32_t pc = 0;
            for (size_t i = 0; i < NUM_WORDS; ++i)
                pc += static_cast<uint32_t>(__builtin_popcountll(planes[b][i]));
            total += pc * (1u << (BitWidth - 1 - b));
        }
        return total;
    }
};

template <size_t D, size_t BitWidth = 1>
struct NbitRaBitQCode {
    static constexpr size_t DIMS = D;
    static constexpr size_t BIT_WIDTH = BitWidth;

    NbitCodeStorage<D, BitWidth> codes;
    float dist_to_centroid;
    float ip_quantized_original;
    uint16_t msb_popcount;

    void clear() {
        codes.clear();
        dist_to_centroid = 0.0f;
        ip_quantized_original = 0.0f;
        msb_popcount = 0;
    }
};

}  // namespace cphnsw