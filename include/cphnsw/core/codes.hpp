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

}  // namespace cphnsw