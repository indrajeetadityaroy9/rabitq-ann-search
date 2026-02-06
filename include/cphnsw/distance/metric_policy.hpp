#pragma once

#include "../core/codes.hpp"
#include "../core/memory.hpp"
#include <cstddef>
#include <type_traits>
#include <immintrin.h>

namespace cphnsw {

using DistanceType = float;

namespace detail {

inline uint32_t popcount64(uint64_t x) {
    return static_cast<uint32_t>(__builtin_popcountll(x));
}

template <size_t K, size_t R, int Shift>
inline void hamming_batch_avx2(
    const CodeQuery<K, R, Shift>& query,
    const CodeSoALayout<ResidualCode<K, R>, 64>& soa_layout,
    size_t offset,
    uint32_t* out) {

    __m256i prim_sums = _mm256_setzero_si256();
    __m256i res_sums = _mm256_setzero_si256();

    for (size_t w = 0; w < ResidualCode<K, R>::PRIMARY_WORDS; ++w) {
        __m256i q = _mm256_set1_epi64x(query.code.primary.signs[w]);
        __m256i n = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&soa_layout.primary_transposed[w][offset]));
        __m256i x = _mm256_xor_si256(q, n);
        alignas(32) uint64_t x_arr[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(xor_arr), x); // Fixed typo x_arr
        alignas(32) uint64_t p_arr[4];
        for (int i = 0; i < 4; ++i) p_arr[i] = popcount64(x_arr[i]);
        prim_sums = _mm256_add_epi64(prim_sums, _mm256_load_si256(reinterpret_cast<const __m256i*>(p_arr)));
    }

    if constexpr (R > 0) {
        for (size_t w = 0; w < ResidualCode<K, R>::RESIDUAL_WORDS; ++w) {
            __m256i q = _mm256_set1_epi64x(query.code.residual.signs[w]);
            __m256i n = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&soa_layout.residual_transposed[w][offset]));
            __m256i x = _mm256_xor_si256(q, n);
            alignas(32) uint64_t x_arr[4];
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(xor_arr), x); // Fixed typo x_arr
            alignas(32) uint64_t p_arr[4];
            for (int i = 0; i < 4; ++i) p_arr[i] = popcount64(x_arr[i]);
            res_sums = _mm256_add_epi64(res_sums, _mm256_load_si256(reinterpret_cast<const __m256i*>(p_arr)));
        }
    }

    __m256i combined = (R == 0) ? prim_sums : _mm256_add_epi64(_mm256_slli_epi64(prim_sums, Shift), res_sums);
    alignas(32) uint64_t res[4];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(res), combined);
    for (int i = 0; i < 4; ++i) out[i] = static_cast<uint32_t>(res[i]);
}

} // namespace detail

template <size_t K, size_t R = 0, int Shift = 2>
struct UnifiedMetricPolicy {
    using CodeType = ResidualCode<K, R>;
    using QueryType = CodeQuery<K, R, Shift>;
    using SoALayoutType = CodeSoALayout<CodeType, 64>;

    static constexpr size_t PRIMARY_BITS = K;
    static constexpr size_t RESIDUAL_BITS = R;
    static constexpr int WEIGHT_SHIFT = Shift;
    static constexpr bool HAS_RESIDUAL = (R > 0);

    static inline DistanceType compute_distance(const QueryType& query, const CodeType& code) {
        uint32_t p_dist = 0;
        for (size_t w = 0; w < CodeType::PRIMARY_WORDS; ++w) p_dist += detail::popcount64(query.code.primary.signs[w] ^ code.primary.signs[w]);
        if constexpr (R == 0) return query.base + query.scale * static_cast<float>(p_dist);
        uint32_t r_dist = 0;
        for (size_t w = 0; w < CodeType::RESIDUAL_WORDS; ++w) r_dist += detail::popcount64(query.code.residual.signs[w] ^ code.residual.signs[w]);
        return query.base + query.scale * static_cast<float>((p_dist << Shift) + r_dist);
    }

    static void compute_distance_batch(const QueryType& query, const SoALayoutType& soa_layout, size_t count, DistanceType* out) {
        size_t n = 0;
#if defined(__AVX2__)
        for (; n + 4 <= count; n += 4) {
            alignas(32) uint32_t bh[4];
            detail::hamming_batch_avx2(query, soa_layout, n, bh);
            for (int i = 0; i < 4; ++i) out[n + i] = query.base + query.scale * static_cast<float>(bh[i]);
        }
#endif
        for (; n < count; ++n) {
            CodeType code;
            soa_layout.load(n, code);
            out[n] = compute_distance(query, code);
        }
    }
};

} // namespace cphnsw
