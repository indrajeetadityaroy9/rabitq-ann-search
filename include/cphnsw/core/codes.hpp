#pragma once

#include <cstdint>
#include <cstring>

namespace cphnsw {

template <size_t D>
constexpr size_t num_sub_segments = (D + 3) / 4;

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
    float nop;
    // L1-norm of the post-rotation, norm_factor_-scaled vector, divided by
    // sqrt(D). Serves as the denominator in the distance estimator; the
    // implicit ||x_bar|| factor cancels with a matching factor in the
    // numerator (ip_corrected).
    float ip_qo;

    void clear() {
        signs.clear();
        nop = 0.0f;
        ip_qo = 0.0f;
    }
};

struct VertexAuxData {
    float nop;
    // Inner product proxy between the quantized code and the rotated
    // neighbor-offset direction. For 1-bit: ||rotated||_1 / sqrt(D).
    // For N-bit: <c_bar, rotated> / sqrt(D) where c_bar = (2u-K)/K are
    // UNNORMALIZED code values (not divided by ||c_bar||). This is NOT
    // the true cosine <c_bar/||c_bar||, rotated>. The implicit ||c_bar||
    // factor cancels in the ratio ip_corrected/ip_qo used in the distance
    // estimator, since ip_corrected carries the same factor.
    float ip_qo;
    float ip_cp;
};

template <size_t D>
struct RaBitQQuery {
    static constexpr size_t DIMS = D;
    static constexpr size_t NUM_SUB_SEGMENTS = num_sub_segments<D>;

    alignas(64) uint8_t lut[NUM_SUB_SEGMENTS][16];

    float coeff_fastscan;
    float coeff_popcount;
    float coeff_constant;

    float affine_a = 1.0f;
    float affine_b = 0.0f;
    float ip_qo_floor = 0.0f;
    float dot_slack = 0.0f;
};


template <size_t D, size_t BitWidth>
struct alignas(64) NbitCodeStorage {
    static_assert(BitWidth >= 1 && BitWidth <= 8, "BitWidth must be 1-8");
    static constexpr size_t NUM_BITS = D;
    static constexpr size_t BIT_WIDTH = BitWidth;
    static constexpr size_t NUM_WORDS = (D + 63) / 64;

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

    uint32_t msb_popcount() const {
        uint32_t count = 0;
        for (size_t i = 0; i < NUM_WORDS; ++i)
            count += static_cast<uint32_t>(__builtin_popcountll(planes[0][i]));
        return count;
    }

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
    float nop;
    // Unnormalized N-bit inner product: sum((2u_i-K)/K * rotated[i]) /
    // sqrt(D). NOT the cosine <c/||c||, rotated> — the ||c|| denominator
    // is omitted but cancels in the ratio ip_corrected/ip_qo. See
    // caq_quantize() in rabitq_encoder.hpp.
    float ip_qo;

    void clear() {
        codes.clear();
        nop = 0.0f;
        ip_qo = 0.0f;
    }
};

}
