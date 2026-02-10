# Graph Construction: Why NN-Descent Fails for Navigability

## Problem
NN-Descent produces 77% recall on 10K SIFT despite:
- All nodes BFS-reachable from entry point
- High kNN approximation quality (converges in 3 iterations)
- RNG pruning + reverse edges + 2 refinement passes

Incremental insertion achieves 99.4% on same dataset.

## Root Cause (from literature)
NN-Descent optimizes **local kNN quality** (each node finds its K nearest neighbors).
Navigability requires **global routing structure** (monotonic paths from entry to all targets).

These are fundamentally different properties:
- kNN quality: for each node, how many true K-NN are found?
- Navigability: from entry point, can greedy search reach any target via monotonically decreasing distances?

## Why Incremental Insertion Works (HNSW/Vamana)
1. First nodes: small graph, perfect search → perfect neighborhoods
2. Each new node: searches existing high-quality graph → finds good neighbors
3. Quality propagates: good graph → good search → good new edges → better graph
4. "Warm start" - graph quality builds incrementally

## Why NN-Descent + Batch Refinement Fails
1. NN-Descent produces uniform local quality but no global routing
2. Batch refinement searches a uniformly poor graph → poor candidates everywhere
3. "Cold start" - no part of the graph is high-quality to bootstrap from
4. Circular: need navigable graph for refinement, need refinement for navigable graph

## Solutions from Literature

### 1. SymphonyQG (SIGMOD 2025)
- Iterative batch refinement from RANDOM graph (not NN-Descent)
- 3-4 passes: search+prune, quality improves each pass
- Uses quantized distances during construction for speed
- Adaptive pruning to fill exactly R slots

### 2. NSG (VLDB 2019)
- Build kNN via NN-Descent
- For each node: search FROM navigating node → collect ALL visited → MRNG prune
- DFS spanning tree ensures connectivity
- The search-from-navigating-node creates routing structure

### 3. Vamana/DiskANN (NeurIPS 2019)
- Two-pass incremental insertion
- Pass 1: alpha=1 (standard RNG, local edges)
- Pass 2: alpha>1 (relaxed pruning, long-range edges preserved)
- RobustPrune: prune q if alpha * dist(p*, q) <= dist(p, q)

### 4. ParlayANN (PPoPP 2024)
- Parallel incremental insertion via prefix-doubling
- Batches of exponentially increasing size (1, 2, 4, 8, ...)
- Early batches = sequential quality; later batches = parallel speed
- Lock-free, deterministic

## Recommended Implementation
Use **incremental insertion** (proven 99.4% recall) with:
1. `select_neighbors_heuristic_fill` for R-slot filling
2. Software prefetching in beam search
3. Vamana-style alpha>1 second pass for long-range edges
4. Optional: parallel batch insertion (ParlayANN-style) for speed
