# CP-HNSW: Zero-Configuration Quantization-Aware ANN Search

Graph-based approximate nearest neighbor (ANN) systems typically expose multiple coupled parameters — beam width, pruning thresholds, quantization codebook size — that require joint tuning per dataset and recall regime. CP-HNSW is a system that eliminates this tuning burden by deriving all internal search and construction parameters from a single user-facing control: `recall_target`.RaBitQ binary quantization with theoretical error bounds and extend it with parent-relative edge encoding (SymphonyQG), where per-edge auxiliary terms enable bounded distance estimation during graph traversal without maintaining a separate codebook. Graph construction uses adaptive NN-Descent with delta-based convergence and pruning aggressiveness derived from sampled neighborhood statistics. At search time, FastScan SIMD kernels compute quantized distance estimates and lower bounds that drive beam expansion and candidate pruning, with exploration aggressiveness controlled by a gamma threshold derived directly from the recall target. The framework supports unified 1/2/4-bit code paths through a common graph and search infrastructure.

## Installation

```bash
pip install -e .
```

Requires: C++17, CMake >= 3.18, OpenMP, Python >= 3.8.

## Usage

```python
import cphnsw

index = cphnsw.Index(dim=128)
index.add(vectors)
index.finalize()

ids, dists = index.search(query, k=10, recall_target=0.95)
ids, dists = index.search_batch(queries, k=10, n_threads=8)
```

## SIFT-1M Results

Dataset: 10^6 base vectors, 10^4 queries, D=128, L2 distance. Single-threaded search, AVX2.

**Index Construction**

| Construction time | Throughput | Memory | Graph degree | NN-Descent rounds | Derived alpha |
|---:|---:|---:|---:|---:|---:|
| 1.29 min | 12,890 vec/s | 2.81 GiB | R=32, avg=32.0 | 11 (delta-converged) | 1.099 |

**Search Quality vs. Throughput**

| recall_target | gamma | epsilon | Recall@10 | QPS | p50 latency | p99 latency |
|---:|---:|---:|---:|---:|---:|---:|
| 0.80 | 1.609 | 1.79 | 0.9597 | 535 | 1,631 us | 6,137 us |
| 0.90 | 2.303 | 2.15 | 0.9735 | 330 | 2,652 us | 9,599 us |
| 0.95 | 2.996 | 2.45 | 0.9815 | 220 | 3,975 us | 13,656 us |
| 0.97 | 3.507 | 2.65 | 0.9849 | 176 | 5,047 us | 16,523 us |
| 0.99 | 4.605 | 3.03 | 0.9896 | 121 | 7,691 us | 23,228 us |

## Reproducing Results

```bash
# Single dataset
python -m cphnsw.bench.run_benchmark --dataset sift1m --k 100 --output-dir results

# All datasets (sift1m, gist1m, openai1536, msmarco10m)
python -m cphnsw.bench.run_benchmark --dataset all --k 100 --output-dir results

# Plot
python -m cphnsw.bench.plot_results results/sift1m_results.json
```

## References

1. Gao & Long. [RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search](https://arxiv.org/abs/2405.12497).
2. Gao & Long. [Practical and Asymptotically Optimal Quantization of High-Dimensional Vectors in Euclidean Space for Approximate Nearest Neighbor Search](https://arxiv.org/abs/2409.09913).
3. Chen et al. [SymphonyQG: Towards Symphonious Integration of Quantization and Graph for Approximate Nearest Neighbor Search](https://arxiv.org/abs/2411.12229).
4. Malkov & Yashunin. [Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs](https://arxiv.org/abs/1603.09320).
