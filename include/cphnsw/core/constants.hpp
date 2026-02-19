#pragma once

#include <cstddef>
#include <cstdint>

namespace cphnsw {
namespace constants {

// --- Semantic epsilon tiers (numerical stability) ---
namespace eps {
    constexpr float kTiny   = 1e-20f;  // division guards, variance floors
    constexpr float kSmall  = 1e-12f;  // near-zero squared norms/distances
    constexpr float kMedium = 1e-10f;  // quality gate thresholds (ip_qo denominators)
    constexpr float kLarge  = 1e-6f;   // shape parameter zero-tests (GPD xi)
}

// --- GPD/EVT theoretical bounds ---
// GPD shape parameter xi: [-0.2, 0.5] covers Weibull-to-Frechet tail classes.
// Hosking & Wallis 1987: xi > 0.5 → infinite-variance tail.
// xi < -0.2 → upper-bounded tail (rare for distance residuals).
constexpr float kGpdBetaMin       = 1e-8f;   // positive-definite GPD scale floor
constexpr float kEvtAlphaMin      = 1e-12f;   // exceedance probability floor (avoid log(0))
constexpr float kEvtAlphaMax      = 0.5f;     // max exceedance prob (50% = body, not tail)
constexpr float kGpdXiMin         = -0.2f;    // Weibull-class boundary
constexpr float kGpdXiMax         = 0.5f;     // finite-variance boundary
constexpr double kMinLayerRandom  = 1e-15;    // prevent log(0) in HNSW level assignment

// --- Slack bounds ---
constexpr int   kMaxSlackArray    = 32;

// --- Grimshaw MLE solver (Grimshaw 1993) ---
// 50 outer + 5 inner iterations are generous; convergence typically ≤15 outer.
constexpr int   kGrimshawMaxIter  = 50;
constexpr int   kGrimshawBetaIter = 5;
constexpr float kGrimshawTol      = 1e-6f;

// --- Huber robust regression (Huber 1964) ---
// 1.345σ: 95% asymptotic efficiency under Gaussian noise.
// 1.4826: MAD-to-σ factor assuming Gaussian (MAD = σ · Φ⁻¹(3/4)).
constexpr float kHuberDeltaScale  = 1.345f;
constexpr float kMadNormFactor    = 1.4826f;
constexpr int   kHuberMaxIter     = 10;
constexpr float kHuberConvergeTol = 1e-6f;

// --- Threading/scheduling ---
constexpr size_t kOmpChunkDiv     = 16;
constexpr size_t kOmpChunkMin     = 16;
constexpr size_t kOmpChunkMax     = 1024;

// --- Prefetching (x86_64 cache line = 64 bytes) ---
constexpr size_t kPrefetchNeighbors = 8;
constexpr size_t kMaxVecPrefetchLines = 4;
constexpr size_t kPrefetchLineCap = 16;
constexpr size_t kPrefetchStride = 4;

// --- SIMD/quantization (hardware constraints) ---
// 1e-4: sufficient for 4-bit quantization (16 levels → ~6% per-level gap).
constexpr float  kCaqEarlyExitTol  = 1e-4f;
constexpr size_t kFlushInterval   = 2;
// 15 = 2⁴ - 1: max value for 4-bit unsigned LUT entries.
constexpr float  kLutLevels       = 15.0f;
// 32 = AVX2 register width / 8-bit lanes: natural SIMD batch size.
constexpr size_t kFastScanBatch   = 32;

// --- User-facing defaults ---
constexpr size_t kDefaultK        = 10;

// --- Reproducibility seeds ---
constexpr uint64_t kDefaultRotationSeed    = 42;
constexpr uint64_t kDefaultLayerSeed       = 42;
constexpr uint64_t kDefaultCalibrationSeed = 99999;
constexpr uint64_t kDefaultGraphSeed       = 42;

// --- Mathematical identity ---
// 6/π² = 1/ζ(2) (Basel problem): normalizes Bonferroni-style per-level risk
// allocation so that Σ(1/i²) from i=1..∞ sums to 1.
constexpr float kBaselK = 6.0f / (3.14159265358979f * 3.14159265358979f);

// --- Calibration ---
// Minimum nodes for calibration: need enough for Huber regression + EVT tail fitting.
constexpr size_t kMinCalibrateNodes = 50;

// --- Dimension-scaled epsilons ---
inline float norm_epsilon(size_t D) {
    return 1e-8f / static_cast<float>(D);
}

inline float coordinate_epsilon(size_t D) {
    return 1e-10f / static_cast<float>(D);
}

}
}
