# Calibration-Parameterless Hierarchical Navigable Small World (CP-HNSW)

Calibration-Parameterless Hierarchical Navigable Small World (CP-HNSW) is a CPU approximate nearest neighbor (ANN) system that unifies quantization, graph navigation, and statistical calibration under a single user-facing search parameter, `recall_target`. The method integrates RaBitQ-style random-rotation vector quantization (1/2/4-bit), HNSW-style hierarchical routing, and EVT-CRC calibration to convert target recall into principled pruning and termination decisions at query time. During indexing, CP-HNSW learns calibration parameters by fitting robust affine correction and generalized Pareto tail statistics on estimator residuals; during search, these statistics are used to allocate risk across sequential pruning steps and derive adaptive slack levels without manual dataset-specific tuning. The implementation includes parent-relative edge metadata for bounded distance estimation, SIMD fastscan kernels, and a unified lifecycle (`add -> finalize -> search`) that enforces calibrated inference. 

## References

1. Gao, J., Long, C. (2024). *RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search*. [arXiv:2405.12497](https://arxiv.org/abs/2405.12497)
2. Malkov, Y. A., Yashunin, D. A. (2018). *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs*. [arXiv:1603.09320](https://arxiv.org/abs/1603.09320)
3. Gao, J., Gou, Y., Xu, Y., Yang, Y., Long, C., Wong, R. C.-W. (2024). *Practical and Asymptotically Optimal Quantization of High-Dimensional Vectors in Euclidean Space for Approximate Nearest Neighbor Search*. [arXiv:2409.09913](https://arxiv.org/abs/2409.09913)
4. Gou, Y., Gao, J., Xu, Y., Long, C. (2024). *SymphonyQG: Towards Symphonious Integration of Quantization and Graph for Approximate Nearest Neighbor Search*. [arXiv:2411.12229](https://arxiv.org/abs/2411.12229)
