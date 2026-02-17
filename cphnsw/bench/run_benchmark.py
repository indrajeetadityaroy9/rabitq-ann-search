"""Run ANN benchmark sweeps and persist JSON results."""

import gc
import json
import os
import time
from pathlib import Path

import cphnsw
import numpy as np
import psutil

from cphnsw.datasets import load_dataset
from cphnsw.metrics import recall_at_k


def timed_search(search_fn, queries: np.ndarray, n_warmup: int = 1, n_runs: int = 3):
    for _ in range(n_warmup):
        search_fn(queries)

    times = []
    result = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        ids = search_fn(queries)
        times.append(time.perf_counter() - t0)
        if result is None:
            result = ids

    median_time = float(np.median(times))
    return result, len(queries) / median_time, median_time

CPHNSW_SPECS = [
    {
        "algorithm": "cphnsw-1bit",
        "bits": 1,
        "param_name": "recall_target",
        "param_values": [0.80, 0.90, 0.95, 0.97, 0.99],
    },
    {
        "algorithm": "cphnsw-2bit",
        "bits": 2,
        "param_name": "recall_target",
        "param_values": [0.80, 0.90, 0.95, 0.97, 0.99],
    },
    {
        "algorithm": "cphnsw-4bit",
        "bits": 4,
        "param_name": "recall_target",
        "param_values": [0.80, 0.90, 0.95, 0.97, 0.99],
    },
]

def run_benchmark(dataset_name: str, base_dir: Path,
                  k: int, n_runs: int, output_dir: Path):
    ds = load_dataset(dataset_name, base_dir=str(base_dir))
    base = ds["base"]
    queries = ds["queries"]
    gt = ds["groundtruth"].astype(np.int64)
    dim = ds["dim"]

    adr_k = min(k, 10)
    gt_ids = gt[:, :adr_k].astype(np.int64)
    gt_dists = np.sum((base[gt_ids] - queries[:, None, :]) ** 2, axis=2)

    results = []

    for spec in CPHNSW_SPECS:
        algorithm = spec["algorithm"]

        index = None
        gc.collect()
        rss_before = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
        t0 = time.perf_counter()

        index = cphnsw.Index(dim=dim, bits=spec["bits"])
        index.add(base)
        index.finalize()

        build_time = time.perf_counter() - t0
        gc.collect()
        rss_after = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
        mem_mb = max(0.0, rss_after - rss_before)

        sweep = []
        for pval in spec["param_values"]:
            def search_fn(batch, _rt=float(pval)):
                ids, _ = index.search_batch(batch, k=k, recall_target=_rt)
                return np.asarray(ids)

            ids, qps_val, med_time = timed_search(search_fn, queries, n_warmup=1, n_runs=n_runs)
            r1 = recall_at_k(ids, gt, 1)
            r10 = recall_at_k(ids, gt, min(k, 10))
            r100 = recall_at_k(ids, gt, min(k, 100))
            lat_us = med_time / len(queries) * 1e6

            res_ids = ids[:, :adr_k].astype(np.int64)
            res_dists = np.sum((base[res_ids] - queries[:, None, :]) ** 2, axis=2)
            adr = float(np.mean(res_dists / np.maximum(gt_dists, 1e-10)))

            sweep.append({
                "param_name": spec["param_name"],
                "param_value": float(pval),
                "recall_at_1": round(r1, 4),
                "recall_at_10": round(r10, 4),
                "recall_at_100": round(r100, 4),
                "adr": round(adr, 6),
                "qps": round(qps_val, 1),
                "median_latency_us": round(lat_us, 2),
            })

        results.append({
            "algorithm": algorithm,
            "build_time_s": round(build_time, 2),
            "memory_mb": round(mem_mb, 1),
            "sweep": sweep,
        })

        index = None
        gc.collect()

    output = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "dataset": dataset_name,
            "n_base": len(base),
            "n_queries": len(queries),
            "dim": dim,
            "metric": "l2",
            "k": k,
            "n_runs": n_runs,
            "base_dir": str(base_dir),
        },
        "results": results,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    outfile = output_dir / f"{dataset_name}_results.json"
    with outfile.open("w") as f:
        json.dump(output, f, indent=2)
    return str(outfile)
