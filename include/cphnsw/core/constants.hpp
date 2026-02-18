#pragma once

#include <cstddef>
#include <cstdint>

namespace cphnsw {
namespace constants {

// --- Numerical stability epsilons ---
constexpr float kNearZeroSq       = 1e-12f;
constexpr float kDivisionEps      = 1e-20f;
constexpr float kGpdVarEps        = 1e-20f;
constexpr float kGpdBetaMin       = 1e-8f;
constexpr float kEvtXiEps         = 1e-6f;
constexpr float kIpQualityEps     = 1e-10f;
constexpr double kMinLayerRandom  = 1e-15;  // floor for layer assignment log-uniform draw

// --- EVT / GPD statistical bounds ---
constexpr float kEvtAlphaMin      = 1e-12f;
constexpr float kEvtAlphaMax      = 0.5f;
constexpr float kGpdXiMin         = -0.2f;
constexpr float kGpdXiMax         = 0.5f;

// --- Calibration parameters ---
constexpr float kEvtThresholdQ    = 0.90f;
constexpr size_t kEvtMinTail      = 64;
constexpr float kEdgeNopQuantile  = 0.95f;
constexpr size_t kMinCalibNodes   = 50;
constexpr size_t kDefaultCalibSamples = 2000;
constexpr size_t kVarEstSamples   = 500;
constexpr size_t kMinAffineSamples = 20;
constexpr size_t kIpQoFloorPct    = 5;
constexpr size_t kResidQuantilePct = 99;

// --- Risk budget / search control ---
constexpr float kPruneRiskFrac    = 0.7f;
constexpr int   kSlackLevels      = 16;
constexpr int   kMaxSlackArray    = 32;
constexpr float kMinRecallTarget  = 0.5f;
constexpr float kMaxRecallTarget  = 0.9999f;
constexpr float kMinSlackSq       = 0.01f;
constexpr size_t kEfMinMultiplier = 4;
constexpr size_t kEfMaxCap        = 8192;
constexpr float kSlackMultiplier  = 2.0f;
constexpr size_t kAlphaTermKMult  = 4;

// --- Distance-ratio termination (gamma) ---
constexpr float kGammaMin         = 1.05f;
constexpr float kGammaMax         = 3.0f;
constexpr float kGammaDefault     = 1.5f;
constexpr float kAdaptiveGammaMin = 1.02f;
constexpr float kAdaptiveGammaMax = 4.0f;
constexpr float kAdaptiveGammaBeta = 0.5f;   // sensitivity to per-query variance
constexpr size_t kAdaptiveGammaWarmup = 8;   // min exact-dist comparisons before adapting

// --- Grimshaw MLE GPD ---
constexpr int   kGrimshawMaxIter  = 50;
constexpr int   kGrimshawBetaIter = 5;      // inner fixed-point iterations for beta
constexpr float kGrimshawTol      = 1e-6f;

// --- Huber regression ---
constexpr float kHuberDeltaScale  = 1.345f;  // Huber delta = scale * MAD
constexpr float kMadNormFactor    = 1.4826f;  // MAD-to-sigma conversion
constexpr int   kHuberMaxIter     = 2;        // IRLS iterations (OLS seed + 2 refinements)
constexpr float kAffineAMin       = 0.5f;
constexpr float kAffineAMax       = 2.0f;

// --- Graph construction ---
constexpr float kNndescentDeltaMin = 0.0001f;
constexpr float kNndescentDeltaMax = 0.01f;
constexpr size_t kNndescentIterMin = 8;
constexpr size_t kNndescentIterMax = 40;

// --- Alpha (neighbor selection) ---
constexpr float kAlphaScaleCoeff  = 0.1f;
constexpr float kAlphaScaleDenom  = 5.0f;
constexpr float kAlphaDefaultMin  = 1.1f;
constexpr float kAlphaDefaultMax  = 1.5f;
constexpr float kAlphaCeiling     = 2.0f;
constexpr float kAlphaMaxCap      = 2.5f;
constexpr float kAlphaFloor       = 1.02f;
constexpr float kTauScale         = 0.5f;
constexpr size_t kAlphaPercentileDiv = 4;
constexpr size_t kAlphaSampleMin  = 64;
constexpr size_t kAlphaSampleMax  = 10000;
constexpr size_t kAlphaInterMin   = 4;

// --- Beam management ---
constexpr float kBeamTrimTrigger  = 2.0f;
constexpr float kBeamTrimKeep     = 0.75f;

// --- Memory / performance ---
constexpr size_t kVisitHeadroomMin = 256;
constexpr size_t kVisitHeadroomMax = 100000;
constexpr size_t kVisitHeadroomDiv = 4;
constexpr size_t kOmpChunkDiv     = 16;
constexpr size_t kOmpChunkMin     = 16;
constexpr size_t kOmpChunkMax     = 1024;
constexpr size_t kPrefetchNeighbors = 8;
constexpr size_t kMaxVecPrefetchLines = 4;
constexpr size_t kPrefetchLineCap = 16;

// --- Upper layer ---
constexpr float kUpperEfGrowth    = 1.5f;
constexpr size_t kUpperEfMaxMult  = 4;
constexpr size_t kUpperLayerDistSamples = 200;
constexpr size_t kUpperLayerNnLimit = 500;
constexpr size_t kTauPercentileDiv = 10;
constexpr size_t kUpperBonusDimThresh = 256; // dimension threshold for degree bonus
constexpr size_t kUpperBonusDivisor  = 8;    // R/this gives bonus degree for high-dim

// --- Encoding ---
constexpr size_t kCaqIterMin       = 2;
constexpr size_t kCaqIterMax       = 6;
constexpr size_t kCaqIterLogDiv    = 3;     // log2(D) / this gives base iterations
constexpr float  kCaqEarlyExitTol  = 1e-4f; // early exit when residual change < tol
constexpr size_t kFlushInterval   = 4;
constexpr float  kLutLevels       = 15.0f;  // 2^4 - 1: quantization levels for 4-bit LUT
constexpr size_t kFastScanBatch   = 32;     // AVX2 batch width for fastscan kernel

// --- Defaults ---
constexpr size_t kDefaultK        = 10;
constexpr float  kDefaultRecall   = 0.95f;

// --- Seeds ---
constexpr uint64_t kDefaultRotationSeed    = 42;
constexpr uint64_t kDefaultLayerSeed       = 42;
constexpr uint64_t kDefaultCalibrationSeed = 99999;
constexpr uint64_t kDefaultGraphSeed       = 42;

// --- Encoding parameters ---
constexpr float kEfCapLogScale    = 0.5f;

// --- Norm epsilon ---
inline float norm_epsilon(size_t D) {
    return 1e-8f / static_cast<float>(D);
}

inline float coordinate_epsilon(size_t D) {
    return 1e-10f / static_cast<float>(D);
}

}  // namespace constants
}  // namespace cphnsw
