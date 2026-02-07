# Randomized Bit Quantization ANN Search

A library for approximate nearest neighbor (ANN) search on dense vectors.

### 1. Randomized Bit Quantization
*   **Problem:** Traditional quantization (PQ/Scalar) lacks rigorous error bounds.
*   **Solution:** Uses **Structured Random Householder Transforms (SRHT)** to "spread" vector energy across dimensions before 1-bit quantization.
*   **Outcome:** Provides a provable $O(1/\sqrt{D})$ error bound on distance estimation, enabling high recall even with extreme compression (32x smaller than float32).

### 2. SIMD-Accelerated Graph Search
*   **Problem:** Graph traversal (HNSW) is memory-bound and suffers from quantization bias.
*   **Solution:**
    *   **FastScan Kernel:** A specialized **4-bit SIMD Lookup Table (LUT)** implementation using AVX-512/AVX2 intrinsics (`vpshufb`) to estimate distances for 32 neighbors in parallel.
    *   **Distance Decomposition:** A novel distance correction term (`ip_corrected`) that compensates for quantization error during traversal using precomputed edge data, eliminating the need for expensive re-ranking.
