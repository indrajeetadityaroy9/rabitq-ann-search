# Configuration-Parameterless HNSW (CP-HNSW): Zero-Configuration Quantization-Aware ANN Search

ANN system that combines RaBitQ-style quantization, parent-relative edge encoding, NN-Descent graph construction, and recall-target-driven search control.

The goal is to reduce parameter tuning burden while preserving competitive recall/throughput tradeoffs. Instead of exposing multiple coupled search/build knobs, the main user-facing control is `recall_target`; internal controls are derived from quantization-error and graph statistics.

## Contributions
1. `Recall-target` as primary search control:
   `gamma` and search error terms are derived from `recall_target` (`include/cphnsw/core/adaptive_defaults.hpp`) instead of manually tuning beam-style parameters.
2. Adaptive NN-Descent construction with auto convergence:
   graph refinement uses a delta-based stop criterion and derives pruning aggressiveness (`alpha`) from sampled neighborhood statistics (`include/cphnsw/graph/graph_refinement.hpp`).
3. Quantization-aware search with lower-bound pruning:
   FastScan-style distance estimation and lower bounds are used to prune candidates while preserving quality (`include/cphnsw/distance/fastscan_kernel.hpp`, `include/cphnsw/search/rabitq_search.hpp`).
4. Parent-relative edge auxiliary encoding:
   per-edge auxiliary terms support bounded distance estimation during traversal (`include/cphnsw/core/codes.hpp`, `include/cphnsw/encoder/rabitq_encoder.hpp`).
5. Unified multi-bit implementation:
   1/2/4-bit code paths share a common graph/search framework (`include/cphnsw/core/codes.hpp`, `include/cphnsw/api/rabitq_index.hpp`).

## Architecture
### Core C++ Modules (Novel Contributions)
- `include/cphnsw/api/rabitq_index.hpp`: **flat index API** (`RaBitQIndex`) - main research contribution.
- `include/cphnsw/encoder/rabitq_encoder.hpp`: centroid-relative quantization and query encoding.
- `include/cphnsw/graph/graph_refinement.hpp`: graph build pipeline (random init -> NN-Descent -> alpha-CG pruning -> reverse-edge pass -> hub entry).
- `include/cphnsw/distance/fastscan_kernel.hpp`: SIMD distance estimation and lower-bound kernels.
- `include/cphnsw/search/rabitq_search.hpp`: quantization-aware traversal/termination.
- `include/cphnsw/core/adaptive_defaults.hpp`: derived defaults from recall/error objectives.

### Optional Extensions (Standard Techniques)
- `include/cphnsw/api/hnsw_index.hpp`: **hierarchical index API** (`HNSWIndex`) - optional scalability extension using standard HNSW multi-layer routing (not a core research contribution). Uses exact L2 distances in upper layers for routing, quantized search only at layer 0.

### Python Layer
- `src/bindings.cpp`: pybind11 bridge exposing `Index` and `HNSWIndex`.
- `cphnsw/__init__.py`: package API surface.
- `cphnsw/bench/run_benchmark.py`: experiment runner across CP-HNSW/hnswlib/faiss baselines.
- `cphnsw/bench/plot_results.py`: result visualization (Recall-QPS, ADR-QPS, build time, memory).
- `cphnsw/datasets.py`: loaders/downloaders for all benchmark datasets.

### External Benchmark Integration
- `bench/vibe-module/module.py`, `bench/vibe-module/config.yml`: VIBE algorithm adapter/config.
- `bench/setup_vibe.sh`: setup/link script for VIBE integration.

## Benchmark Datasets

| Dataset | N | D | Queries | Metric | Source |
|---|---|---|---|---|---|
| `sift1m` | 1,000,000 | 128 | 10,000 | L2 | HuggingFace (`qbo-odp/sift1m`) |
| `gist1m` | 1,000,000 | 960 | 1,000 | L2 | HuggingFace (`fzliu/gist1m`) |
| `openai1536` | ~999,000 | 1,536 | 1,000 | L2 | HuggingFace (`Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M`) |
| `msmarco10m` | 10,000,000 | 1,024 | 1,000 | L2 | HuggingFace (`Cohere/msmarco-v2.1-embed-english-v3`) |

## Baseline Algorithms

| Algorithm | Family | Sweep Parameter |
|---|---|---|
| CP-HNSW flat 1/2/4-bit | cphnsw | `recall_target` [0.80-0.99] |
| CP-HNSW HNSW 1-bit | cphnsw | `recall_target` [0.80-0.99] |
| hnswlib M=16/32/64 | hnswlib | `ef` [100-2000] |
| FAISS HNSW M=32 | faiss-hnsw | `ef` [100-2000] |
| FAISS IVF-PQ | faiss-ivfpq | `nprobe` [1-256] |
| FAISS IVF-OPQ | faiss-ivfopq | `nprobe` [1-256] |

## Evaluation Metrics

- **Recall@1, Recall@10, Recall@100**: fraction of true nearest neighbors retrieved
- **QPS**: queries per second (single-threaded, median of 3 runs)
- **ADR (Average Distance Ratio)**: mean ratio of returned distances to ground truth distances (closer to 1.0 is better)
- **Build time**: index construction wall-clock time
- **Memory**: RSS delta before/after index build

## End-to-End Execution Flow
1. Data loading:
   benchmark script resolves dataset and metric (`cphnsw/datasets.py`, `cphnsw/bench/run_benchmark.py`).
2. Index build:
   vectors are encoded and inserted; graph refinement runs adaptive NN-Descent + pruning (`include/cphnsw/graph/graph_refinement.hpp`).
3. Query encoding and search:
   queries are quantized; FastScan estimates + lower bounds drive beam expansion/pruning (`include/cphnsw/search/rabitq_search.hpp`).
4. Adaptive termination:
   recall-target-derived thresholds determine exploration aggressiveness (`include/cphnsw/core/adaptive_defaults.hpp`).
5. Metrics + artifacts:
   structured logs, JSON result files, and plots are produced (`cphnsw/bench/run_benchmark.py`, `cphnsw/bench/plot_results.py`).

## Environment and Installation
### Requirements
- C++17 compiler
- CMake >= 3.18
- OpenMP
- Python >= 3.8

### Install package
```bash
pip install -e .
```

This installs the core package along with all benchmark dependencies (faiss, hnswlib, huggingface_hub, pandas, matplotlib, psutil, torch).

## Reproducibility Protocol
### Data layout
```text
data/
  sift1m/
  gist1m/
  openai1536/
  msmarco10m/
```

Supported benchmark identifiers in the Python runner:
- `sift1m`
- `gist1m`
- `openai1536`
- `msmarco10m`

### Main benchmark commands
```bash
python -m cphnsw.bench.run_benchmark --dataset sift1m --k 100 --base-dir data --output-dir results
python -m cphnsw.bench.run_benchmark --dataset all --k 100 --base-dir data --output-dir results
```

### Plotting
```bash
python -m cphnsw.bench.plot_results results/sift1m_results.json
```

### One-command reproduction
```bash
bash scripts/reproduce_results.sh
```

### Output artifacts
- `results/<dataset>_results.json`: benchmark metadata and per-algorithm sweeps.
- `results/*.log`: structured `event=...` runtime logs from scripts/runners.
- `results/*_recall10_qps.png`, `*_recall100_qps.png`, `*_adr_qps.png`, `*_build_time.png`, `*_memory.png`: generated plots.

### C++ executables (CMake)
- `benchmark`, `benchmark_d128`, `benchmark_d256`, `benchmark_d512`
- `test_nbit`
- `test_hnsw`

## SIFT-1M Benchmark Results
Dataset: SIFT-1M (`10^6` base vectors, `10^4` queries, `D=128`, L2 distance).
Hardware: 30-core server, AVX2, single-threaded search.

### Index Construction
| Metric | Value |
|---|---|
| Construction time | 1.29 min |
| Throughput | 12,890 vec/s |
| Memory footprint | 2.81 GiB |
| Graph degree | `R=32`, avg=32.0 |
| NN-Descent rounds | 11 (delta-converged) |
| Derived alpha | 1.099 |

### Search Quality vs. Throughput
| recall_target | gamma | epsilon | Recall@10 | QPS | p50 latency | p99 latency |
|---|---:|---:|---:|---:|---:|---:|
| 0.80 | 1.609 | 1.79 | 0.9597 | 535 | 1,631 us | 6,137 us |
| 0.90 | 2.303 | 2.15 | 0.9735 | 330 | 2,652 us | 9,599 us |
| 0.95 | 2.996 | 2.45 | 0.9815 | 220 | 3,975 us | 13,656 us |
| 0.97 | 3.507 | 2.65 | 0.9849 | 176 | 5,047 us | 16,523 us |
| 0.99 | 4.605 | 3.03 | 0.9896 | 121 | 7,691 us | 23,228 us |

## References
1. RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search (arXiv:2405.12497)
   https://arxiv.org/abs/2405.12497
   Quantization/error-bound mechanism that this implementation uses for ANN distance estimation.
2. Practical and Asymptotically Optimal Quantization of High-Dimensional Vectors in Euclidean Space for Approximate Nearest Neighbor Search (arXiv:2409.09913)
   https://arxiv.org/abs/2409.09913
   Practical multi-bit quantization design choices for higher-accuracy code paths.
3. SymphonyQG: Towards Symphonious Integration of Quantization and Graph for Approximate Nearest Neighbor Search (arXiv:2411.12229)
   https://arxiv.org/abs/2411.12229
   Parent-relative quantization/graph integration and in-graph quantized distance guidance used in this system.
4. Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs (arXiv:1603.09320)
   https://arxiv.org/abs/1603.09320
   Hierarchical routing structure
5. Fast-Convergent Proximity Graphs for Approximate Nearest Neighbor Search (arXiv:2510.05975)
   https://arxiv.org/abs/2510.05975
   Related to alpha-style pruning/convergence ideas
6. Empowering Graph-based Approximate Nearest Neighbor Search with Adaptive Awareness Capabilities (arXiv:2506.15986)
   https://arxiv.org/abs/2506.15986
   Adaptive graph-search
