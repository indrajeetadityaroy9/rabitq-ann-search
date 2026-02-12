# CP-HNSW: Zero-Configuration Quantization-Aware Graph Search for Approximate Nearest Neighbors

A C++17 header-only library that unifies **RaBitQ** quantization, **SymphonyQG** parent-relative encoding, **NN-Descent** graph construction, and principled adaptive termination into a single cohesive system for billion-scale approximate nearest neighbor (ANN) search. All internal parameters are derived from data statistics and first-principles error bounds — the user specifies only the desired recall target.

## Overview

Standard graph-based ANN systems (HNSW, Vamana, DiskANN) expose a large surface of interacting hyperparameters: `ef_construction`, `ef_search`, `M`, pruning thresholds, error tolerances, and convergence criteria. These require dataset-specific tuning and make reproducibility difficult.

CP-HNSW eliminates this configuration burden. The system accepts a single quality knob — `recall_target` — and derives every internal decision from the dimensionality of the data, the structure of the constructed graph, and the probabilistic error bounds of RaBitQ quantization (Theorem 3.2). The result is a zero-configuration ANN index that builds and searches without manual tuning while maintaining competitive recall and throughput.

### Contributions

1. **Unified quantization-aware construction pipeline.** NN-Descent local join with delta-convergence detection replaces fixed-iteration refinement. Alpha-Convergent Graph (alpha-CG) pruning with data-derived alpha replaces manual pruning thresholds. SymphonyQG parent-relative encoding is applied uniformly across all bit widths (1, 2, 4-bit), eliminating the code-path divergence between binary and multi-bit quantization.

2. **Principled search termination from recall target.** Two complementary mechanisms replace the `ef_search` parameter: (a) lower-bound pruning via RaBitQ Theorem 3.2, where the false-prune probability equals exactly `1 - recall_target`, and (b) distance-adaptive beam search (DABS) termination with gamma derived from the quantization error scale `epsilon / sqrt(D)`. Both compare exact L2 distances on both sides of the inequality.

3. **Complete parameter elimination.** The public API reduces to `{dim, recall_target, k}`. Build-time alpha, convergence thresholds, error epsilons, beam widths, and entry point selection are all derived automatically.

## Method

### System Architecture

```
                         User API
                    dim, recall_target, k
                            |
                            v
    +-----------------------------------------------+
    |               Adaptive Defaults                |
    |  epsilon = sqrt(-2 ln(1 - recall))   [Thm 3.2]|
    |  gamma = epsilon / sqrt(D)           [DABS]   |
    |  delta = 0.001                       [NN-Desc] |
    |  alpha = median(d_nb) / median(d_inter) [data] |
    +-----------------------------------------------+
            |                           |
            v                           v
    +-----------------+       +-------------------+
    |   Construction  |       |      Search       |
    |                 |       |                   |
    | 1. Random init  |       | Beam ordered by   |
    | 2. NN-Descent   |       | RaBitQ estimates  |
    |    local join   |       |        |          |
    | 3. alpha-CG     |       |  Lower-bound      |
    |    pruning +    |       |  pruning (Thm 3.2)|
    |    SymphonyQG   |       |        |          |
    |    encoding     |       |  Exact L2 rerank  |
    | 4. Reverse edges|       |        |          |
    | 5. Hub entry    |       |  DABS terminate   |
    +-----------------+       +-------------------+
            |                           |
            v                           v
    +-----------------------------------------------+
    |          Colocated Vertex Storage              |
    |  [code | neighbor codes | aux | raw vector]   |
    |          aligned to 64-byte cache lines        |
    +-----------------------------------------------+
```

### Construction (5 phases, no `ef_construction` parameter)

**Phase 1: Random initialization.** Each vertex samples `4R` random candidates and retains the closest `R` as initial neighbors. Parallel over vertices.

**Phase 2: NN-Descent with local join.** The core insight from Dong et al. (2011): *a neighbor of my neighbor is likely my neighbor*. Each round performs local joins between each vertex's neighbor lists, updating edges when shorter distances are found. A new/old flag optimization ensures each pair is evaluated at most once. Convergence is detected automatically: the algorithm stops when the edge update rate drops below delta = 0.001 (0.1%), which is dataset-agnostic and typically reached in 5-15 rounds.

**Phase 3: alpha-CG pruning + SymphonyQG encoding.** Candidates are pruned using the alpha-Convergent Graph criterion (Def. 3.1, arXiv:2510.05975): a candidate is redundant if an existing neighbor provides sufficient coverage via a shifted-scaled triangle inequality. The pruning threshold alpha is derived from the graph's own distance statistics (median neighbor distance / median inter-neighbor distance), not set manually. Simultaneously, for each retained edge, SymphonyQG parent-relative auxiliary data is computed: the per-edge correction term `||o-c||^2 - ||p-c||^2`, the parent-relative distance `||o-p||`, and a binary code encoding `sign(P * unit(o - p))`. These are packed into SIMD-friendly FastScan code blocks (32 neighbors per block, 4-bit LUT layout).

**Phase 4: Reverse edge pass.** For each forward edge `u -> v`, the reverse edge `v -> u` is added as a candidate, and the neighbor list is re-pruned with alpha-CG. This improves graph connectivity.

**Phase 5: Hub entry point selection.** The entry point is chosen as the highest-degree node among the top-sqrt(n) nodes closest to the dataset centroid. This combines centrality with connectivity, inspired by GATE (arXiv:2506.15986).

### Search (no `ef_search` parameter)

The beam search processes candidates in order of their RaBitQ estimated distances and applies two termination mechanisms:

**Lower-bound pruning (Theorem 3.2).** Each candidate carries a guaranteed lower bound on its true L2 distance, computed from the RaBitQ error bound:

```
lower_bound = correction + ||q-p||^2 - 2 * ||q-c|| * ||o-p|| * cos_upper
cos_upper = min(ip_est + epsilon * sqrt((1 - rho^2) / (rho^2 * D)), 1)
```

where `epsilon = sqrt(-2 ln(1 - recall_target))` controls tightness. A candidate is skipped (not terminated) when its lower bound exceeds the k-th best exact distance found so far. The false-prune probability is exactly `1 - recall_target` by construction.

**Distance-adaptive termination (DABS).** After computing a candidate's exact L2 distance, the search terminates entirely if:

```
exact_dist > (1 + gamma) * nn.worst_distance()
```

where `gamma = epsilon / sqrt(D)` matches the scale of RaBitQ's relative estimation error (which is `O(epsilon / sqrt(D))` from the concentration of the quantized inner product). Both sides are exact L2 distances. An estimate-based pre-filter applies the same threshold to the RaBitQ estimate before computing exact L2, avoiding unnecessary reranking.

### Quantization

**1-bit (RaBitQ).** Each vector is centered, rotated via a Subsampled Randomized Hadamard Transform (SRHT), normalized, and binarized to signs. The inner product is estimated via a 4-bit lookup table (FastScan) and corrected with the vector's L1 norm. Distance formula:

```
||q - o||^2 = ||o-c||^2 + ||q-c||^2 - 2 * ||o-c|| * ||q-c|| * cos_est
```

**Multi-bit (Extended RaBitQ).** Vectors are quantized to `2^B` levels (B = 2 or 4 bits) by optimizing a scaling parameter that minimizes quantization error. The optimal scaling is found by enumerating critical thresholds for B <= 3 or via golden-section search for B > 3. Storage uses a bit-plane layout where the MSB plane carries the sign (used for parent-relative encoding) and lower planes refine the magnitude.

**Parent-relative encoding (SymphonyQG).** Rather than encoding each neighbor relative to the global centroid, the system encodes the MSB as `sign(P * unit(o - p))` where `p` is the parent vertex. This decomposes the squared distance as:

```
||q - o||^2 = (||o-c||^2 - ||p-c||^2) + ||q-p||^2 - 2 * ||q-c|| * ||o-p|| * <unit(q-c), unit(o-p)>
```

The `||q-p||^2` term is the already-computed exact distance to the parent, and the inner product term is estimated via FastScan on the parent-relative code. This is applied uniformly for all bit widths.

## SIFT-1M Benchmark Results

Dataset: SIFT-1M (10^6 base vectors, 10^4 queries, D=128, L2 distance).
Hardware: 30-core server, AVX2, single-threaded search.

### Index Construction

| Metric | Value |
|---|---|
| Construction time | 1.29 min |
| Throughput | 12,890 vec/s |
| Memory footprint | 2.81 GiB |
| Graph degree | R=32, avg=32.0 |
| NN-Descent rounds | 11 (delta-converged) |
| Derived alpha | 1.099 |

### Search Quality vs. Throughput

| recall_target | gamma | epsilon | Recall@10 | QPS | p50 latency | p99 latency |
|---|---|---|---|---|---|---|
| 0.80 | 1.609 | 1.79 | 0.9597 | 535 | 1,631 us | 6,137 us |
| 0.90 | 2.303 | 2.15 | 0.9735 | 330 | 2,652 us | 9,599 us |
| 0.95 | 2.996 | 2.45 | 0.9815 | 220 | 3,975 us | 13,656 us |
| 0.97 | 3.507 | 2.65 | 0.9849 | 176 | 5,047 us | 16,523 us |
| 0.99 | 4.605 | 3.03 | 0.9896 | 121 | 7,691 us | 23,228 us |

All recall targets are exceeded. Recall increases monotonically with `recall_target` and QPS decreases monotonically, confirming that the single-parameter control provides the expected quality-throughput tradeoff.

## Installation

### Requirements

- C++17 compiler (GCC 9+, Clang 11+)
- CMake 3.18+
- OpenMP
- x86-64 with AVX2 (AVX-512 optional, auto-detected)

### C++ (header-only)

```bash
git clone https://github.com/your-repo/rabitq-ann-search.git
cd rabitq-ann-search
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Python

```bash
pip install .
```

Requires `pybind11 >= 2.12` and `numpy >= 1.20`.

## Quick Start

### C++

```cpp
#include <cphnsw/api/rabitq_index.hpp>
using namespace cphnsw;

// Build index (D=128, R=32, 1-bit quantization)
RaBitQIndex<128, 32> index(IndexParams().set_dim(128));
index.add_batch(vectors, num_vectors);
index.finalize(BuildParams().set_verbose(true));

// Search — only k and recall_target
auto results = index.search(query,
    SearchParams().set_k(10).set_recall_target(0.95f));

for (const auto& r : results)
    printf("id=%u  dist=%.2f\n", r.id, r.distance);
```

### Python

```python
import cphnsw
import numpy as np

# Build
index = cphnsw.Index(dim=128, bits=1)
index.add_batch(base_vectors)           # (n, 128) float32
index.finalize(verbose=True)

# Search
ids, dists = index.search(query, k=10, recall_target=0.95)

# Batch search (parallel)
ids, dists = index.search_batch(queries, k=10, n_threads=8)
```

### Multi-bit quantization

```python
# 4-bit Extended RaBitQ (higher accuracy, larger index)
index = cphnsw.Index(dim=128, bits=4)
```

### HNSW (hierarchical)

```python
index = cphnsw.HNSWIndex(dim=128, bits=1)
index.add_batch(base_vectors)
index.finalize(verbose=True)
ids, dists = index.search(query, k=10, recall_target=0.99)
```

## Reproducing Results

### Download SIFT-1M

```bash
mkdir -p data/sift && cd data/sift
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar xzf sift.tar.gz && mv sift/* . && rmdir sift
cd ../..
```

### Run benchmark

```bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native -O3"
make -j$(nproc) benchmark
./benchmark ../data/sift
```

Expected output:

```
[Build] Phase 1: Random init (n=1000000, R=32, 30 threads)...
[Build] Phase 2: NN-Descent (delta=0.0010, threshold=32000)...
[Build]   Round 1: 126487072 updates (395.272%)
  ...
[Build]   Converged at round 11
[Build] Derived alpha=1.099
[Build] Phase 3: alpha-CG pruning + FastScan encoding...
[Build] Phase 4: Reverse edge pass...
[Build] Phase 5: Hub entry=123742 (degree=32)
[Build] Done. avg_degree=32.0, max_degree=32

[Metrics] Indexing: 1.29 min | Memory: 2.81 GiB
[Metrics] Throughput: 12890 vec/s

recall_target=0.80  gamma=1.609  eps=1.79  Recall@10=0.9597  QPS=535  ...
recall_target=0.95  gamma=2.996  eps=2.45  Recall@10=0.9815  QPS=220  ...
recall_target=0.99  gamma=4.605  eps=3.03  Recall@10=0.9896  QPS=121  ...
```

## Project Structure

```
include/cphnsw/
  api/
    params.hpp              IndexParams, BuildParams, SearchParams
    rabitq_index.hpp        RaBitQIndex — primary index API
    hnsw_index.hpp          HNSWIndex — hierarchical variant
  core/
    types.hpp               NodeId, SearchResult, Spinlock
    codes.hpp               RaBitQCode, NbitRaBitQCode, RaBitQQuery
    adaptive_defaults.hpp   Parameter derivation from recall_target and D
    memory.hpp              Aligned allocators, SIMD L2 distance
  encoder/
    rabitq_encoder.hpp      RaBitQEncoder, NbitRaBitQEncoder (CRTP)
    rotation.hpp            Subsampled Randomized Hadamard Transform
    dense_rotation.hpp      Dense orthogonal rotation with ITQ learning
    transform/fht.hpp       Fast Hadamard Transform (scalar/AVX2/AVX-512)
  distance/
    fastscan_kernel.hpp     SIMD batch inner products + distance conversion
    fastscan_layout.hpp     Cache-aligned code block and neighbor layouts
  graph/
    rabitq_graph.hpp        Colocated vertex storage with prefetching
    graph_refinement.hpp    5-phase NN-Descent construction pipeline
    neighbor_selection.hpp  alpha-CG pruning (Def. 3.1)
    visitation_table.hpp    Lock-free epoch-based deduplication
  search/
    rabitq_search.hpp       Beam search with dual principled termination
  io/
    serialization.hpp       Binary index save/load
src/
  bindings.cpp              pybind11 Python interface
apps/
  benchmark.cpp             SIFT-1M evaluation protocol
  test_nbit.cpp             Multi-bit quantization verification
  test_hnsw.cpp             HNSW multi-layer verification
  test_itq.cpp              ITQ learned rotation verification
python/cphnsw/
  __init__.py               Module exports
  datasets.py               .fvecs/.ivecs/.hdf5 loaders
  metrics.py                recall@k, QPS computation
  gpu.py                    Optional PyTorch GPU helpers
```

## Parameter Derivation Summary

The system exposes three user parameters. Everything else is derived.

| User parameter | Default | Purpose |
|---|---|---|
| `dim` | (required) | Vector dimensionality |
| `k` | 10 | Number of nearest neighbors to return |
| `recall_target` | 0.95 | Desired search quality |

| Internal parameter | Derivation | Source |
|---|---|---|
| epsilon (search) | `sqrt(-2 ln(1 - recall_target))` | Gaussian tail bound (Thm 3.2) |
| epsilon (build) | `sqrt(-2 ln(0.25))` = 1.665 | Fixed false-prune rate p=0.25 |
| gamma (DABS) | `epsilon / sqrt(D)` | RaBitQ error scaling O(1/sqrt(D)) |
| alpha (pruning) | `median(d_neighbor) / median(d_inter)` | Graph distance statistics |
| delta (convergence) | 0.001 | Wei Dong et al. (2011) gold standard |
| error tolerance | `1 / sqrt(D)` | Quantization variance O(1/D) |
| entry point | max-degree among top-sqrt(n) to centroid | Hub selection |

## References

This implementation integrates ideas from the following works:

**RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search.**
Jianyang Gao, Cheng Long. SIGMOD 2024.
*Core quantization scheme. Binary codes via random rotation + sign quantization. Theorem 3.2 provides the error bound that drives both lower-bound pruning and the epsilon/gamma derivations in this system.*

**RaBitQ Meets Multi-Bit: Extended Quantization for High-Dimensional Vectors.**
Jianyang Gao, Cheng Long. SIGMOD 2025.
*Multi-bit extension (2/4-bit). Bit-plane storage layout with MSB carrying the sign. Optimal scaling parameter found by critical-threshold enumeration.*

**SymphonyQG: Towards Symphonious Integration of Quantization and Graph for Approximate Nearest Neighbor Search.**
Yutong Gou, Jianyang Gao, Yuexuan Xu, Cheng Long. SIGMOD 2025.
*Parent-relative encoding. Per-edge auxiliary data decomposes the distance using the parent vertex as a reference, eliminating centroid bias. The MSB of each edge code is encoded as `sign(P * unit(o-p))` rather than `sign(P * unit(o-c))`.*

**Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs.**
Yu. A. Malkov, D. A. Yashunin. IEEE TPAMI 2020.
*HNSW layered graph structure used in the optional HNSWIndex variant. Layer assignment via exponential decay, greedy routing on upper layers.*

**Efficient K-Nearest Neighbor Graph Construction for Generic Similarity Measures.**
Wei Dong, Charikar Moses, Kai Li. WWW 2011.
*NN-Descent algorithm for graph construction. Local join with new/old flag optimization. Delta-convergence threshold of 0.001.*

**alpha-Convergent Graph Construction for Approximate Nearest Neighbor Search.**
arXiv:2510.05975.
*Definition 3.1: shifted-scaled triangle inequality for pruning. Reduces false-negative pruning compared to standard relative neighborhood graphs.*

**GATE: How to Keep Out Intruders.**
arXiv:2506.15986.
*Hub-based entry point selection. Highest-degree node among top-sqrt(n) closest to centroid provides better routing than medoid selection.*

**Distance-Adaptive Beam Search.**
Morozov, Babenko. 2018.
*DABS termination criterion. Terminate beam when candidate distance exceeds (1+gamma) times the best distance. This system derives gamma from the RaBitQ error scale rather than treating it as a tunable parameter.*

## License

MIT
