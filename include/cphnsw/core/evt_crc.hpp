#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <algorithm>

namespace cphnsw {

struct EVTState {
    float u = 0.0f;
    float p_u = 0.0f;
    float xi = 0.0f;
    float beta = 0.0f;
    float nop_p95 = 0.0f;
    uint32_t n_resid = 0;
    uint32_t n_tail = 0;
    bool fitted = false;
};

namespace evt_crc {

// GPD tail quantile for exceedance probability alpha.
inline float evt_quantile(float alpha, const EVTState& evt, float resid_q99_dot) {
    if (!evt.fitted) {
        return resid_q99_dot;
    }

    alpha = std::clamp(alpha, 1e-12f, 0.5f);

    if (alpha >= evt.p_u) {
        return evt.u;
    }

    float ratio = evt.p_u / alpha;

    if (std::abs(evt.xi) < 1e-6f) {
        return evt.u + evt.beta * std::log(ratio);
    } else {
        return evt.u + (evt.beta / evt.xi) * (std::pow(ratio, evt.xi) - 1.0f);
    }
}

// Basel-series alpha spending across pruning steps.
inline float alpha_spend(int i, float delta_prune) {
    constexpr float BASEL_K = 6.0f / (3.14159265358979f * 3.14159265358979f);
    float i_f = static_cast<float>(std::max(i, 1));
    return delta_prune * BASEL_K / (i_f * i_f);
}

// Fit GPD on sorted absolute residuals.
inline EVTState fit_gpd(const float* sorted_abs_resid, size_t n,
                        float threshold_quantile, size_t min_tail) {
    EVTState state;
    state.n_resid = static_cast<uint32_t>(n);

    if (n < min_tail * 2) {
        return state;
    }

    size_t u_idx = static_cast<size_t>(static_cast<float>(n) * threshold_quantile);
    u_idx = std::min(u_idx, n - 1);
    state.u = sorted_abs_resid[u_idx];

    double sum_y = 0.0;
    double sum_y2 = 0.0;
    uint32_t m = 0;

    for (size_t i = u_idx + 1; i < n; ++i) {
        float y = sorted_abs_resid[i] - state.u;
        if (y > 0.0f) {
            sum_y += y;
            sum_y2 += static_cast<double>(y) * y;
            ++m;
        }
    }

    state.n_tail = m;
    state.p_u = static_cast<float>(m) / static_cast<float>(n);

    if (m < min_tail) {
        return state;
    }

    double mean_y = sum_y / m;
    double var_y = sum_y2 / m - mean_y * mean_y;

    if (var_y < 1e-20) {
        state.xi = 0.0f;
        state.beta = std::max(static_cast<float>(mean_y), 1e-8f);
    } else {
        state.xi = 0.5f * (1.0f - static_cast<float>(mean_y * mean_y / var_y));
        state.beta = static_cast<float>(mean_y) * (1.0f - state.xi);
    }

    state.xi = std::clamp(state.xi, -0.2f, 0.5f);
    state.beta = std::max(state.beta, 1e-8f);

    state.fitted = true;
    return state;
}

}  // namespace evt_crc
}  // namespace cphnsw
