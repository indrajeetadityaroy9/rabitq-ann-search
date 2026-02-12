#!/usr/bin/env python3
"""Head-to-head ANN benchmark: CP-HNSW vs hnswlib vs faiss.

Usage:
    python bench/run_benchmark.py --dataset sift1m
    python bench/run_benchmark.py --dataset glove200
    python bench/run_benchmark.py --dataset all
"""

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Tuple, Optional, Callable

import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Ensure cphnsw is importable from build
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
if str(REPO_DIR / "python") not in sys.path:
    sys.path.insert(0, str(REPO_DIR / "python"))

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SweepPoint:
    param_name: str
    param_value: int
    recall_at_1: float
    recall_at_10: float
    qps: float
    median_latency_us: float

@dataclass
class AlgorithmResult:
    algorithm: str
    build_time_s: float
    memory_mb: float
    sweep: list  # List[SweepPoint] serialized

# ---------------------------------------------------------------------------
# Dataset loading (uses cphnsw.datasets for fvecs/ivecs/hdf5 parsing)
# ---------------------------------------------------------------------------

from cphnsw.datasets import load_fvecs, load_ivecs, load_hdf5_dataset


def normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    return (X / norms).astype(np.float32)


def load_sift1m() -> dict:
    base_dir = REPO_DIR / "data" / "sift1m"
    if not base_dir.exists():
        base_dir = REPO_DIR / "data" / "sift"
    base = load_fvecs(str(base_dir / "sift_base.fvecs"))
    queries = load_fvecs(str(base_dir / "sift_query.fvecs"))
    gt = load_ivecs(str(base_dir / "sift_groundtruth.ivecs"))
    # SIFT uses L2 on raw vectors — do NOT normalize (ground truth is L2-based)
    return {
        "name": "sift-1m", "dim": base.shape[1], "metric": "l2",
        "base": base.astype(np.float32), "queries": queries.astype(np.float32),
        "groundtruth": gt.astype(np.int64),
    }


def load_glove200() -> dict:
    path = REPO_DIR / "data" / "glove200" / "glove-200-angular.hdf5"
    if not path.exists():
        # fallback to old location
        path = REPO_DIR / "bench" / "vibe" / "data" / "glove-200-cosine.hdf5"
    ds = load_hdf5_dataset(str(path))
    return {
        "name": "glove-200", "dim": ds["dim"], "metric": "cosine",
        "base": ds["base"].astype(np.float32),
        "queries": ds["queries"].astype(np.float32),
        "groundtruth": ds["groundtruth"].astype(np.int64),
    }

# ---------------------------------------------------------------------------
# Recall computation
# ---------------------------------------------------------------------------

def recall_at_k(results: np.ndarray, gt: np.ndarray, k: int) -> float:
    """Vectorized batch recall@k."""
    res = results[:, :k].astype(np.int64)
    truth = gt[:, :k].astype(np.int64)
    hits = np.any(res[:, :, None] == truth[:, None, :], axis=2)
    return float(hits.sum(axis=1).mean()) / k

# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def timed_search(search_fn: Callable, queries: np.ndarray,
                 n_warmup: int = 1, n_runs: int = 3):
    for _ in range(n_warmup):
        search_fn(queries)

    times = []
    result = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        ids = search_fn(queries)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        if result is None:
            result = ids

    median_time = float(np.median(times))
    qps = len(queries) / median_time
    return result, qps, median_time

# ---------------------------------------------------------------------------
# Memory helper
# ---------------------------------------------------------------------------

def get_rss_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    except ImportError:
        return 0.0

# ---------------------------------------------------------------------------
# Algorithm runners
# ---------------------------------------------------------------------------

class BaseRunner:
    name: str = ""
    index = None

    def build(self, base, dim, metric):
        raise NotImplementedError

    def search_fn(self, k, param):
        """Return a callable: queries -> ids (np.ndarray)."""
        raise NotImplementedError

    def param_name(self) -> str:
        return "ef"

    def param_values(self) -> list:
        return [10, 20, 40, 80, 100, 200, 400, 800]

    def cleanup(self):
        self.index = None
        gc.collect()


class CPHNSWFlatRunner(BaseRunner):
    def __init__(self, bits=1, M=32, efc=200):
        self.bits = bits
        self.M = M
        self.efc = efc
        self.name = f"cphnsw-flat-{bits}bit-M{M}"

    def build(self, base, dim, metric):
        import cphnsw
        self.index = cphnsw.Index(dim=dim, M=self.M,
                                   ef_construction=self.efc, bits=self.bits)
        self.index.add(base)
        self.index.finalize()

    def search_fn(self, k, param):
        idx = self.index
        def _search(queries):
            ids, _ = idx.search_batch(queries, k=k, ef=param,
                                       n_threads=1, error_epsilon=1.9)
            return np.asarray(ids)
        return _search


class CPHNSWHnswRunner(BaseRunner):
    def __init__(self, bits=1, M=32, efc=200):
        self.bits = bits
        self.M = M
        self.efc = efc
        self.name = f"cphnsw-hnsw-{bits}bit-M{M}"

    def build(self, base, dim, metric):
        import cphnsw
        self.index = cphnsw.HNSWIndex(dim=dim, M=self.M,
                                       ef_construction=self.efc, bits=self.bits)
        self.index.add(base)
        self.index.finalize()

    def search_fn(self, k, param):
        idx = self.index
        def _search(queries):
            ids, _ = idx.search_batch(queries, k=k, ef=param,
                                       n_threads=1, error_epsilon=1.9)
            return np.asarray(ids)
        return _search


class HnswlibRunner(BaseRunner):
    def __init__(self, M=32, efc=200):
        self.M = M
        self.efc = efc
        self.name = f"hnswlib-M{M}"

    def build(self, base, dim, metric):
        import hnswlib
        space = "cosine" if metric == "cosine" else "l2"
        self.index = hnswlib.Index(space=space, dim=dim)
        self.index.init_index(max_elements=len(base),
                              ef_construction=self.efc, M=self.M)
        self.index.set_num_threads(1)
        self.index.add_items(base, np.arange(len(base)))

    def search_fn(self, k, param):
        idx = self.index
        def _search(queries):
            idx.set_ef(param)
            labels, _ = idx.knn_query(queries, k=k, num_threads=1)
            return labels
        return _search


class FaissHNSWRunner(BaseRunner):
    def __init__(self, M=32, efc=200):
        self.M = M
        self.efc = efc
        self.name = f"faiss-hnsw-M{M}"

    def build(self, base, dim, metric):
        import faiss
        faiss.omp_set_num_threads(1)
        self.index = faiss.IndexHNSWFlat(dim, self.M)
        self.index.hnsw.efConstruction = self.efc
        self.index.add(base)

    def search_fn(self, k, param):
        import faiss
        idx = self.index
        def _search(queries):
            faiss.omp_set_num_threads(1)
            idx.hnsw.efSearch = max(param, k)
            _, I = idx.search(queries, k)
            return I
        return _search


# ---------------------------------------------------------------------------
# Algorithm list builder
# ---------------------------------------------------------------------------

def build_algorithms(dataset_name: str, dim: int) -> list:
    algos = []

    # CP-HNSW variants
    try:
        import cphnsw  # noqa: F401
        algos.append(CPHNSWFlatRunner(bits=1))
        algos.append(CPHNSWFlatRunner(bits=2))
        algos.append(CPHNSWFlatRunner(bits=4))
        algos.append(CPHNSWHnswRunner(bits=1))
    except ImportError:
        print("WARNING: cphnsw not available, skipping CP-HNSW variants")

    # hnswlib
    try:
        import hnswlib  # noqa: F401
        algos.append(HnswlibRunner(M=16))
        algos.append(HnswlibRunner(M=32))
        algos.append(HnswlibRunner(M=64))
    except ImportError:
        print("WARNING: hnswlib not available")

    # faiss (graph-based only, for fair comparison)
    try:
        import faiss  # noqa: F401
        algos.append(FaissHNSWRunner(M=32))
    except ImportError:
        print("WARNING: faiss not available")

    return algos

# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_benchmark(dataset_name: str, k: int, n_runs: int, output_dir: str):
    print(f"\n{'='*70}")
    print(f"  Loading dataset: {dataset_name}")
    print(f"{'='*70}")

    if dataset_name == "sift1m":
        ds = load_sift1m()
    elif dataset_name == "glove200":
        ds = load_glove200()
    elif dataset_name.endswith(".hdf5") or dataset_name.endswith(".h5"):
        # Load custom HDF5 dataset by path
        raw = load_hdf5_dataset(dataset_name)
        ds = {
            "name": Path(dataset_name).stem,
            "dim": raw["dim"],
            "metric": "l2",
            "base": raw["base"].astype(np.float32),
            "queries": raw["queries"].astype(np.float32),
            "groundtruth": raw["groundtruth"].astype(np.int64),
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                         f"Use sift1m, glove200, or a path to an HDF5 file.")

    base = ds["base"]
    queries = ds["queries"]
    gt = ds["groundtruth"]
    dim = ds["dim"]
    metric = ds["metric"]

    # For cosine metric, normalize for L2-based libraries
    if metric == "cosine":
        base_norm = normalize(base)
        queries_norm = normalize(queries)
    else:
        base_norm = base
        queries_norm = queries

    print(f"  N={len(base):,}  D={dim}  queries={len(queries):,}  metric={metric}")
    print(f"  k={k}  n_runs={n_runs}")

    algorithms = build_algorithms(dataset_name, dim)
    results = []

    for runner in algorithms:
        print(f"\n{'─'*60}")
        print(f"  {runner.name}")
        print(f"{'─'*60}")

        # For hnswlib with cosine, use raw data (it handles normalization)
        if isinstance(runner, HnswlibRunner) and metric == "cosine":
            build_data = base
            search_data = queries
        else:
            build_data = base_norm
            search_data = queries_norm

        # Build with memory measurement
        gc.collect()
        rss_before = get_rss_mb()
        t0 = time.perf_counter()
        try:
            runner.build(build_data, dim, metric)
        except Exception as e:
            print(f"  BUILD FAILED: {e}")
            runner.cleanup()
            continue
        build_time = time.perf_counter() - t0
        gc.collect()
        rss_after = get_rss_mb()
        mem_mb = max(0.0, rss_after - rss_before)

        print(f"  Build: {build_time:.1f}s  Memory: ~{mem_mb:.0f} MB")

        # Sweep search parameters
        sweep = []
        for pval in runner.param_values():
            try:
                fn = runner.search_fn(k, pval)
                ids, qps_val, med_time = timed_search(fn, search_data,
                                                       n_warmup=1, n_runs=n_runs)
                r1 = recall_at_k(ids, gt, 1)
                r10 = recall_at_k(ids, gt, min(k, 10))
                lat_us = med_time / len(search_data) * 1e6

                sweep.append(asdict(SweepPoint(
                    param_name=runner.param_name(),
                    param_value=pval,
                    recall_at_1=round(r1, 4),
                    recall_at_10=round(r10, 4),
                    qps=round(qps_val, 1),
                    median_latency_us=round(lat_us, 2),
                )))

                print(f"  {runner.param_name()}={pval:>4d}  "
                      f"R@1={r1:.4f}  R@10={r10:.4f}  QPS={qps_val:>8.0f}")
            except Exception as e:
                print(f"  {runner.param_name()}={pval}: ERROR {e}")

        results.append(asdict(AlgorithmResult(
            algorithm=runner.name,
            build_time_s=round(build_time, 2),
            memory_mb=round(mem_mb, 1),
            sweep=sweep,
        )))

        runner.cleanup()

    # Write results
    output = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "dataset": ds["name"],
            "n_base": len(base),
            "n_queries": len(queries),
            "dim": dim,
            "metric": metric,
            "k": k,
            "n_runs": n_runs,
        },
        "results": results,
    }

    os.makedirs(output_dir, exist_ok=True)
    outfile = Path(output_dir) / f"{dataset_name}_results.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {outfile}")
    return str(outfile)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ANN Benchmark Suite")
    parser.add_argument("--dataset", default="sift1m",
                        help="Dataset name (sift1m, glove200, all) or path to HDF5 file")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "results"))
    args = parser.parse_args()

    datasets = ["sift1m", "glove200"] if args.dataset == "all" else [args.dataset]

    result_files = []
    for ds_name in datasets:
        f = run_benchmark(ds_name, args.k, args.n_runs, args.output_dir)
        result_files.append(f)

    print(f"\nAll benchmarks complete. Result files:")
    for f in result_files:
        print(f"  {f}")


if __name__ == "__main__":
    main()
