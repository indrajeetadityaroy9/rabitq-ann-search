# Claim-to-Code Map

| Claim | Mechanism | Entry Path |
|---|---|---|
| Single user knob | `recall_target` controls risk budget | `Index.search`, `Index.search_batch` |
| Parent-relative estimation | `nop`, `ip_qo`, `ip_cp` | graph build + search kernels |
| EVT risk control | GPD fit + quantile slack | `Index.finalize` -> `Index.search*` |
| Unified bit-width paths | shared templated 1/2/4-bit stack | `Index(dim, bits=1|2|4)` |
| Persistent calibrated behavior | graph + calibration + EVT serialization | `Index.save`, `cphnsw.load` |
| Canonical evaluation path | CP-HNSW-only benchmark sweep | `python -m cphnsw.eval` |

## Primary Paths

1. Inference: `Index.add -> finalize -> search/search_batch`
2. Evaluation: `python -m cphnsw.eval`
3. Persistence: `Index.save` / `cphnsw.load`
