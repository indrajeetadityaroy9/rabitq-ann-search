"""Benchmark evaluation library."""

import gc
import json
import time
from pathlib import Path

import numpy as np
import psutil

import cphnsw
from cphnsw.datasets import load_dataset

BIT_WIDTHS = [1, 2, 4]
ADR_K = 10
ADR_EPS = 1e-30
US_PER_SEC = 1e6
BYTES_PER_MB = 1024 ** 2
SECS_PER_MIN = 60.0
MB_PER_GIB = 1024.0


def recall_at_k(results: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    eval_k = min(k, results.shape[1], ground_truth.shape[1])
    res = results[:, :eval_k]
    gt = ground_truth[:, :eval_k]
    hits = np.any(res[:, :, None] == gt[:, None, :], axis=2)
    return float(hits.sum(axis=1).mean()) / eval_k


def run_benchmark(dataset_name: str, base_dir: Path,
                  k: int, n_runs: int, output_dir: Path) -> dict:
    ds = load_dataset(dataset_name, base_dir=base_dir)
    base = ds["base"]
    queries = ds["queries"]
    gt = ds["groundtruth"].astype(np.int64)
    dim = ds["dim"]

    adr_k = min(k, ADR_K, gt.shape[1])
    gt_ids = gt[:, :adr_k].astype(np.int64)
    gt_dists = np.sum((base[gt_ids] - queries[:, None, :]) ** 2, axis=2)

    results = []

    for bits in BIT_WIDTHS:
        algorithm = f"cphnsw-{bits}bit"

        gc.collect()
        rss_before = psutil.Process().memory_info().rss / BYTES_PER_MB
        t0 = time.perf_counter()

        index = cphnsw.CPIndex(dim=dim, bits=bits)
        index.build(base)
        index.finalize()

        build_time = time.perf_counter() - t0
        gc.collect()
        rss_after = psutil.Process().memory_info().rss / BYTES_PER_MB
        mem_mb = rss_after - rss_before

        def search_fn(batch):
            ids, _ = index.search_batch(batch, k=k)
            return np.asarray(ids)

        search_fn(queries)
        times = []
        t0 = time.perf_counter()
        ids = search_fn(queries)
        times.append(time.perf_counter() - t0)
        for _ in range(n_runs - 1):
            t0 = time.perf_counter()
            search_fn(queries)
            times.append(time.perf_counter() - t0)
        med_time = float(np.median(times))
        qps_val = len(queries) / med_time

        r1 = recall_at_k(ids, gt, 1)
        r10 = recall_at_k(ids, gt, min(k, 10))
        r100 = recall_at_k(ids, gt, min(k, 100))
        lat_us = med_time / len(queries) * US_PER_SEC

        res_ids = ids[:, :adr_k].astype(np.int64)
        res_dists = np.sum((base[res_ids] - queries[:, None, :]) ** 2, axis=2)
        adr = float(np.mean(res_dists / np.maximum(gt_dists, ADR_EPS)))

        results.append({
            "algorithm": algorithm,
            "build_time_s": round(build_time, 2),
            "memory_mb": round(mem_mb, 1),
            "recall_at_1": round(r1, 4),
            "recall_at_10": round(r10, 4),
            "recall_at_100": round(r100, 4),
            "adr": round(adr, 6),
            "qps": round(qps_val, 1),
            "median_latency_us": round(lat_us, 2),
        })

        del index
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
        },
        "results": results,
    }

    outfile = output_dir / f"{dataset_name}_results.json"
    with outfile.open("w") as f:
        json.dump(output, f, indent=2)

    return output
