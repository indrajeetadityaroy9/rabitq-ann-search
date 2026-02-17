# CP-HNSW Research Spec

## Objectives

1. Expose a single control: `recall_target`.
2. Use parent-relative edge encoding for bounded distance estimation.
3. Derive pruning/termination slack from EVT calibration instead of manual tuning.

## Novel Mechanism

1. Parent-relative edge metadata: `nop`, `ip_qo`, `ip_cp`.
2. Calibrated estimator: ip_qo floor + affine correction.
3. EVT-CRC risk allocation from `recall_target`.

## Required Components

1. Index lifecycle: `include/cphnsw/api/hnsw_index.hpp`.
2. Encoding: `include/cphnsw/encoder/`.
3. FastScan kernels: `include/cphnsw/distance/`.
4. Graph build/refinement: `include/cphnsw/graph/`.
5. Search loop: `include/cphnsw/search/rabitq_search.hpp`.
6. EVT fit + calibration: `include/cphnsw/core/evt_crc.hpp`, `include/cphnsw/api/hnsw_index.hpp`.
7. Persistence: `include/cphnsw/io/serialization.hpp`.
8. Canonical evaluation: `cphnsw/eval.py`, `cphnsw/bench/run_benchmark.py`.
