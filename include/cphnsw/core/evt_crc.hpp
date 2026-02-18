#pragma once

#include "constants.hpp"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <vector>

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

    alpha = std::clamp(alpha, constants::kEvtAlphaMin, constants::kEvtAlphaMax);

    if (alpha >= evt.p_u) {
        return evt.u;
    }

    float ratio = evt.p_u / alpha;

    if (std::abs(evt.xi) < constants::kEvtXiEps) {
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

// Grimshaw MLE for GPD parameters (xi, beta).
// Solves profile log-likelihood by Newton iteration on the shape parameter.
// Falls back to method-of-moments if MLE doesn't converge.
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

    // Collect exceedances above threshold.
    std::vector<double> y;
    y.reserve(n - u_idx);
    for (size_t i = u_idx + 1; i < n; ++i) {
        double yi = sorted_abs_resid[i] - state.u;
        if (yi > 0.0) y.push_back(yi);
    }

    uint32_t m = static_cast<uint32_t>(y.size());
    state.n_tail = m;
    state.p_u = static_cast<float>(m) / static_cast<float>(n);

    if (m < min_tail) {
        return state;
    }

    double sum_y = 0.0, sum_y2 = 0.0;
    double y_max = 0.0;
    for (uint32_t i = 0; i < m; ++i) {
        sum_y += y[i];
        sum_y2 += y[i] * y[i];
        if (y[i] > y_max) y_max = y[i];
    }
    double mean_y = sum_y / m;

    // Method-of-moments initial guess.
    double var_y = sum_y2 / m - mean_y * mean_y;
    double xi_mom, beta_mom;
    if (var_y < constants::kGpdVarEps) {
        xi_mom = 0.0;
        beta_mom = std::max(mean_y, static_cast<double>(constants::kGpdBetaMin));
    } else {
        xi_mom = 0.5 * (1.0 - mean_y * mean_y / var_y);
        beta_mom = mean_y * (1.0 - xi_mom);
    }

    // Grimshaw MLE: profile log-likelihood on xi.
    // For xi != 0: beta(xi) = mean(y) * xi / (1 - (mean of (1+xi*y/beta)^0) ... )
    // We iterate Newton on the score equation.
    double xi = xi_mom;
    double beta = std::max(beta_mom, static_cast<double>(constants::kGpdBetaMin));
    bool mle_converged = false;

    for (int iter = 0; iter < constants::kGrimshawMaxIter; ++iter) {
        // Given current xi, compute MLE beta.
        if (std::abs(xi) < constants::kEvtXiEps) {
            // Exponential case: beta = mean_y.
            beta = mean_y;
            xi = 0.0;
            mle_converged = true;
            break;
        }

        // Check feasibility: 1 + xi * y_i / beta > 0 for all i.
        bool feasible = true;
        for (uint32_t i = 0; i < m; ++i) {
            if (1.0 + xi * y[i] / beta <= 0.0) { feasible = false; break; }
        }
        if (!feasible) break;

        // Profile log-likelihood derivatives w.r.t. xi.
        // l(xi, beta) = -m*log(beta) - (1+1/xi) * sum(log(1+xi*y/beta))
        // With beta = (xi / m) * sum(y / (1+xi*y/beta))... use iterative approach.
        // Simpler: fix xi, solve for beta via score equation, then update xi.

        // Score for beta (set to 0): beta = (1+xi) * mean(y / (1 + xi*y/beta))
        // This is implicit; iterate.
        double beta_new = beta;
        for (int j = 0; j < constants::kGrimshawBetaIter; ++j) {
            double s = 0.0;
            for (uint32_t i = 0; i < m; ++i) {
                double z = 1.0 + xi * y[i] / beta_new;
                if (z <= 0.0) { s = -1.0; break; }
                s += 1.0 / z;
            }
            if (s <= 0.0) break;
            beta_new = (1.0 + xi) * sum_y / s;
            if (beta_new < constants::kGpdBetaMin) beta_new = constants::kGpdBetaMin;
        }
        beta = beta_new;

        // Score for xi (Newton step).
        double score = 0.0, info = 0.0;
        for (uint32_t i = 0; i < m; ++i) {
            double z = 1.0 + xi * y[i] / beta;
            if (z <= 0.0) { score = 0.0; break; }
            double lz = std::log(z);
            double w = y[i] / (beta * z);
            score += -lz / (xi * xi) + (1.0 + 1.0 / xi) * w;
            info += 2.0 * lz / (xi * xi * xi) - 2.0 * w / (xi * xi)
                    + (1.0 + 1.0 / xi) * w * w;
        }

        if (std::abs(info) < constants::kGpdVarEps) break;
        double xi_new = xi - score / info;
        xi_new = std::max(xi_new, static_cast<double>(constants::kGpdXiMin));
        xi_new = std::min(xi_new, static_cast<double>(constants::kGpdXiMax));

        if (std::abs(xi_new - xi) < constants::kGrimshawTol) {
            xi = xi_new;
            mle_converged = true;
            break;
        }
        xi = xi_new;
    }

    if (!mle_converged) {
        // Fall back to method-of-moments.
        xi = xi_mom;
        beta = beta_mom;
    }

    state.xi = std::clamp(static_cast<float>(xi), constants::kGpdXiMin, constants::kGpdXiMax);
    state.beta = std::max(static_cast<float>(beta), constants::kGpdBetaMin);
    state.fitted = true;
    return state;
}

}  // namespace evt_crc
}  // namespace cphnsw
