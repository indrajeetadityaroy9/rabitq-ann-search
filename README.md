# CP-HNSW

High-performance approximate nearest neighbor search combining RaBitQ (SIGMOD 2024) binary quantization with SymphonyQG (SIGMOD 2025) FastScan SIMD graph search.

## Features

- **RaBitQ quantization**: Unbiased L2 distance estimator with O(1/sqrt(D)) error bounds
- **FastScan SIMD**: AVX2/AVX-512 batch distance computation via `vpshufb` 4-bit LUT
- **Implicit reranking**: Exact distances computed at vertex visit time (no separate reranking pass)
- **Fixed-degree graph**: R=32 neighbors per vertex, SIMD-aligned storage

## Quick Start (Python)

```python
import numpy as np
import cphnsw

# Build index
vectors = np.random.randn(10000, 128).astype(np.float32)
index = cphnsw.Index(dim=128, M=32, ef_construction=200)
index.add(vectors)
index.finalize()

# Search
query = np.random.randn(128).astype(np.float32)
ids, distances = index.search(query, k=10, ef=100)
```

## Quick Start (C++)

```cpp
#include <cphnsw/api/rabitq_index.hpp>
using namespace cphnsw;

RaBitQIndex<128, 32> index(IndexParams().set_dim(100).set_M(32));
index.add_batch(vectors, num_vectors);
index.finalize();

auto results = index.search(query, SearchParams().set_k(10).set_ef(100));
```

## Building

```bash
# C++ only
cmake -B build && cmake --build build
./build/test_rabitq

# With Python bindings
pip install -e ".[eval]"
```

### Requirements

- C++17 compiler (GCC 9+, Clang 10+)
- AVX2 or AVX-512 CPU
- OpenMP
- Python 3.8+ with pybind11 (for bindings)

## Reproducing Results

```bash
# Download SIFT-1M to data/sift1m/
python scripts/eval.py --config experiments/recall_qps/sift1m.yaml
# Or run all experiments:
./scripts/reproduce_results.sh
```

## Repository Structure

```
src/
  cpp/cphnsw/          C++ header-only library (15 headers)
    core/              types.hpp, codes.hpp, memory.hpp
    encoder/           rabitq_encoder.hpp, rotation.hpp, transform/fht.hpp
    distance/          fastscan_kernel.hpp, fastscan_layout.hpp
    graph/             rabitq_graph.hpp, graph_refinement.hpp, ...
    search/            rabitq_search.hpp
    api/               rabitq_index.hpp, params.hpp
  cpp/bindings.cpp     pybind11 Python bindings
  cphnsw/              Python package (datasets, metrics, evaluation)
scripts/               Thin entry points (train, eval, sweep)
configs/               Default YAML configuration
experiments/           Per-experiment configs (recall_qps, memory, construction, scalability)
tests/                 C++ and Python tests
docs/papers/           Paper notes
```

## References

- Gao & Long (2024): "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound" (SIGMOD 2024)
- SymphonyQG (2025): "RaBitQ Meets HNSW: FastScan SIMD Graph Search" (SIGMOD 2025)
- Andoni et al. (2015): "Practical and Optimal LSH for Angular Distance"
