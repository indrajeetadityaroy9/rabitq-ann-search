# CP-HNSW

Cross-Polytope Navigable Small World - a memory-efficient approximate nearest neighbor search algorithm optimized for angular/cosine similarity.

## Overview

CP-HNSW combines two state-of-the-art techniques:
- **Cross-Polytope LSH** (Andoni et al. 2015): Asymptotically optimal hashing for angular distance.
- **Navigable Small World (NSW)**: Memory-efficient graph for logarithmic-time navigation.

Key benefits:
- **Memory efficient**: ~32x compression vs standard HNSW (K bytes per vector vs 4d bytes) using "Flash" layout.
- **Flash Layout**: Neighbors and their quantized codes are stored inline to eliminate pointer-chasing and maximize cache efficiency.
- **Fast distance computation**: SIMD-optimized Hamming distance (XOR + PopCount) using AVX2 or AVX-512.
- **Unified Precision**: Supports Phase 1 (RaBitQ) and Phase 2 (Residual) quantization in a single unified API.

## Quick Start

```cpp
#include <cphnsw/api/index.hpp>

using namespace cphnsw;

// Create index for 128-dimensional vectors (using Phase 1 RaBitQ, 32-bit codes)
Index32 index(128);

// Or create with custom parameters
IndexParams params;
params.set_dim(128).set_M(32).set_ef_construction(200);
Index32 custom_index(params);

// Add vectors in parallel (OpenMP)
std::vector<float> vectors = ...;  // N x 128
index.add_batch(vectors.data(), N);

// Search with re-ranking for high recall
std::vector<float> query(128);
SearchParams search_params;
search_params.set_k(10).set_ef(100).set_rerank(true, 500);

auto results = index.search(query.data(), search_params);

for (const auto& r : results) {
    std::cout << "ID: " << r.id << " Distance: " << r.distance << "\n";
}
```

## Building

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

# Run master evaluation protocol
./eval_master --sift data/sift/ --output results/
```

### Build Requirements

- C++17 compiler (GCC 9+, Clang, or AppleClang)
- AVX2 or AVX-512 supported CPU
- OpenMP (Required for parallel construction)

## API Reference

### `CPHNSWIndex<K, R, Shift>`

Template parameters:
- `K`: Primary code bits (e.g., 32, 64)
- `R`: Residual code bits (0 for Phase 1, >0 for Phase 2)
- `Shift`: Bit-shift for residual weighting (default 2)

#### Type Aliases

- `Index32`: 32-bit RaBitQ
- `Index64`: 64-bit RaBitQ
- `Index64_32`: 64-bit Primary + 32-bit Residual

#### Methods

| Method | Description |
|--------|-------------|
| `add(const float* vec)` | Add single vector (thread-safe) |
| `add_batch(const float* vecs, size_t count)` | Add vectors in parallel using OpenMP |
| `search(const float* query, const SearchParams& params)` | K-NN search with CP distance and optional reranking |
| `size()` | Number of indexed vectors |
| `get_stats()` | Get graph connectivity and degree statistics |

## Architecture

```
include/cphnsw/
├── api/
│   └── index.hpp           # Unified public API (Index, Params)
├── core/
│   ├── codes.hpp           # Unified Primary/Residual code storage
│   ├── memory.hpp          # Aligned allocators and prefetch hints
│   └── types.hpp           # Base types (NodeId, DistanceType)
├── distance/
│   ├── detail/             # SIMD Kernels (AVX2, AVX-512)
│   └── metric_policy.hpp   # Unified distance interface
├── encoder/
│   ├── transform/
│   │   └── fht.hpp         # Optimized Fast Hadamard Transform
│   ├── cp_encoder.hpp      # Vector → Binary code encoding
│   └── rotation.hpp        # Pseudo-random rotation chains
├── graph/
│   ├── flat_graph.hpp      # Memory-efficient NSW graph storage
│   ├── neighbor_block.hpp  # Inline "Flash" neighbor storage
│   └── visitation_table.hpp# Epoch-based visitation tracking
└── search/
    └── search_engine.hpp   # Greedy beam search logic
```

## References

- Andoni et al. (2015): "Practical and Optimal LSH for Angular Distance"
- Malkov & Yashunin (2018): "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"
- SIGMOD (2024): "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound"