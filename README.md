# CP-HNSW: Calibration-Parameterless Approximate Nearest Neighbor Search

A graph-based approximate nearest neighbor (ANN) search system that eliminates manually tuned search parameters through statistical calibration. It combines RaBitQ quantization with HNSW graph routing and an EVT-CRC calibration pipeline that automatically derives all pruning thresholds, termination conditions, and distance correction coefficients from the data distribution at index construction time.


1. **Parameter-free search via EVT-CRC calibration.** All search hyperparameters (beam termination threshold, distance slack levels, gamma bounds) are derived from Extreme Value Theory applied to calibration residuals. The Generalized Pareto Distribution (GPD) models the tail of distance estimation errors, with Kolmogorov-Smirnov goodness-of-fit validation and automatic fallback to empirical quantile interpolation when the parametric model is rejected.

2. **Distance-Adaptive Beam Search (DABS).** Candidate-queue filtering at the adaptive distance-ratio threshold `gamma_q * d_k` prevents unbounded beam growth without requiring an explicit `ef` parameter. The threshold adapts per-query via online estimation of the distance approximation error variance.

3. **N-bit RaBitQ with SAQ-style coordinate refinement.** Extends 1-bit RaBitQ sign quantization to 2-bit and 4-bit codes via Cosine-Aligned Quantization (CAQ). For 4-bit codes, a +/-1 coordinate-descent refinement (inspired by SAQ) reduces encoding cost from O(K*D) to O(2*D) per iteration while maintaining near-optimal cosine alignment from the LVQ initialization.

4. **SIMD-native FastScan distance computation.** All distance estimates are computed via AVX2 VPSHUFB lookup-table kernels operating on packed 4-bit sub-segments, with a two-stage lower-bound/full-distance pipeline for N-bit codes that skips full computation when the MSB-only lower bound already exceeds the current worst result.


### Pipeline Stages

**Build.** Vectors are centered by the dataset centroid, then transformed by a 3-layer random Hadamard rotation (Fast Hadamard Transform with pseudorandom sign flips) to distribute information uniformly across coordinates. Rotated vectors are quantized into 1-bit (sign), 2-bit, or 4-bit RaBitQ codes. For N-bit codes (B >= 2), an initial Linear Vector Quantization assignment is refined by coordinate-descent CAQ that iteratively maximizes cosine similarity between the code and the rotated residual.

**Finalize.** HNSW layer assignments follow the standard exponential decay with an adaptive upper-layer degree `M_upper = R/2 + sqrt(D)/4`. The base layer (layer 0) is optimized by NNDescent with adaptive convergence detection (EMA-based rate monitoring, geometric extrapolation for minimum rounds). Neighbor selection uses alpha-CNG pruning with data-driven alpha and tau derived from the graph's nearest-neighbor distance distribution (MAD-sigma robust estimation). After convergence, a BFS reorder from the hub entry point improves cache locality.

Calibration fits an affine correction `a * est + b` to the relationship between FastScan distance estimates and true L2 distances via Huber robust regression (1.345-sigma, 95% asymptotic efficiency). Distance estimation residuals are modeled by fitting a GPD to the upper tail using Grimshaw MLE with multi-threshold stability selection. A KS test with Lilliefors-corrected critical values validates the GPD fit; rejected fits fall back to 8-point empirical quantile interpolation with log-linear extrapolation. The calibrated EVT model produces per-level distance slack values (allocated via Basel-series Bonferroni correction) and the initial search gamma threshold.

**Search.** Upper-layer navigation uses greedy search. At layer 0, beam search processes neighbor codes in batches of 32 via FastScan. The beam terminates when the best candidate's estimated distance exceeds `gamma_q * d_k`, where `gamma_q` starts at the calibrated `search_gamma` and adapts per-query based on the running standard deviation of observed estimation error ratios. Candidates are only enqueued if their estimated distance falls below the DABS threshold, naturally bounding beam size without an explicit `ef` parameter. All k-NN candidates receive exact L2 reranking.

## References

1. Gao and Long. [RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search](https://arxiv.org/abs/2405.12497).
2. Gao et al. [Practical and Asymptotically Optimal Quantization of High-Dimensional Vectors in Euclidean Space for Approximate Nearest Neighbor Search](https://arxiv.org/abs/2409.09913).
3. Malkov and Yashunin. [Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs](https://arxiv.org/abs/1603.09320).
4. Gou et al. [SymphonyQG: Towards Symphonious Integration of Quantization and Graph for Approximate Nearest Neighbor Search](https://arxiv.org/abs/2411.12229).
5. Al-Jazzazi et al. [Distance Adaptive Beam Search for Provably Accurate Graph-Based Nearest Neighbor Search](https://arxiv.org/abs/2505.15636).
6. Li et al. [SAQ: Pushing the Limits of Vector Quantization through Code Adjustment and Dimension Segmentation](https://arxiv.org/abs/2509.12086).
7. Zhong et al. [VSAG: An Optimized Search Framework for Graph-based Approximate Nearest Neighbor Search](https://arxiv.org/abs/2503.17911).
