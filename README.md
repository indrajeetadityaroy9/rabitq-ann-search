# CP-HNSW

Configuration-parameterless approximate nearest neighbor (ANN) search with a single public control knob: `recall_target`.

1. RaBitQ-style vector quantization and fast distance approximation.
2. HNSW-style hierarchical routing and graph traversal.
3. Adaptive graph refinement and memory-locality reordering.
4. EVT-calibrated risk control that converts `recall_target` into pruning and termination slack.

Execution path:
`Index.add(...) -> Index.finalize() -> Index.search(...) / Index.search_batch(...)`

## Contributions

1. A cohesive CP-HNSW pipeline where quantization, graph construction, calibration, and search are jointly wired.
2. Parent-relative edge encoding with bounded distance estimation in the search loop.
3. Calibration stage with affine correction plus GPD-tail EVT fit used directly at inference time.
4. Configuration-parameterless adaptive defaults for graph/search internals, with `recall_target` as the quality-speed control.

## Evaluation

The benchmark path evaluates three CP-HNSW variants:

1. `cphnsw-1bit`
2. `cphnsw-2bit`
3. `cphnsw-4bit`

For each variant, the sweep parameter is `recall_target` in:
`[0.80, 0.90, 0.95, 0.97, 0.99]`

Reported metrics per sweep point:

1. `recall_at_1`, `recall_at_10`, `recall_at_100`
2. `adr` (average distance ratio)
3. `qps`
4. `median_latency_us`

## References

1. **RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search**  
   https://arxiv.org/abs/2405.12497  

2. **SymphonyQG: Towards Symmetric Compression for Learning-Based Quantized Graph**  
   https://arxiv.org/abs/2409.09913  

3. **HNSW-PQ: Transitioning from Approximate Nearest Neighbor Benchmarks to Production Environments**  
   https://arxiv.org/abs/2411.12229  

4. **Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs**  
   https://arxiv.org/abs/1603.09320  