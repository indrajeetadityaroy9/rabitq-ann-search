"""Mechanism-level evaluation functions for CP-HNSW.

Developer-facing tests for validating search sweep, serialization,
gamma tracking, and multi-threaded search.
"""

import gc
import os
import time

import numpy as np
import psutil

import cphnsw
from cphnsw.metrics import recall_at_k


def get_rss_gib():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)


def run_search_sweep(ds, bits_list, recall_targets, k=10, n_runs=3):
    """Build indices at each bit width and sweep recall targets.

    Returns dict mapping bit width to build stats and per-target results.
    """
    base = ds["base"]
    queries = ds["queries"]
    gt = ds["groundtruth"].astype(np.int64)
    dim = ds["dim"]

    results = {}

    for bits in bits_list:
        gc.collect()
        rss_before = get_rss_gib()

        t0 = time.perf_counter()
        idx = cphnsw.Index(dim=dim, bits=bits)
        idx.add(base)
        idx.finalize()
        build_s = time.perf_counter() - t0

        gc.collect()
        rss_after = get_rss_gib()
        index_mem_gib = max(0.0, rss_after - rss_before)

        sweep = []
        for rt in recall_targets:
            idx.search_batch(queries[:100], k=k, n_threads=1, recall_target=rt)

            times = []
            ids_np = None
            for _ in range(n_runs):
                t0 = time.perf_counter()
                ids, _ = idx.search_batch(queries, k=k, n_threads=1, recall_target=rt)
                times.append(time.perf_counter() - t0)
                if ids_np is None:
                    ids_np = np.asarray(ids)

            med_time = float(np.median(times))
            qps = len(queries) / med_time
            r_at_k = recall_at_k(ids_np, gt, k)
            lat_us = med_time / len(queries) * 1e6

            sweep.append({
                "recall_target": rt,
                "recall_at_k": round(r_at_k, 4),
                "qps": round(qps, 1),
                "latency_us": round(lat_us, 2),
            })

        results[bits] = {
            "build_time_s": round(build_s, 2),
            "memory_gib": round(index_mem_gib, 4),
            "sweep": sweep,
            "index": idx,
        }

    return results


def run_serialization_test(index, queries, k=10):
    """Test save/load round-trip and verify identical results."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        tmppath = f.name

    index.save(tmppath)
    loaded = cphnsw._core.load(tmppath)

    ids_orig, _ = index.search_batch(queries[:100], k=k, n_threads=1, recall_target=0.95)
    ids_load, _ = loaded.search_batch(queries[:100], k=k, n_threads=1, recall_target=0.95)
    match = bool(np.array_equal(np.asarray(ids_orig), np.asarray(ids_load)))

    os.unlink(tmppath)

    return {"roundtrip_match": match}


def run_gamma_tracking(index, ds, gamma_targets, k=10):
    """Validate that actual recall tracks the recall target."""
    queries = ds["queries"]
    gt = ds["groundtruth"].astype(np.int64)

    results = []
    for rt in gamma_targets:
        ids, _ = index.search_batch(queries, k=k, n_threads=1, recall_target=rt)
        r_at_k = recall_at_k(np.asarray(ids), gt, k)
        delta = r_at_k - rt
        results.append({
            "target": rt,
            "actual": round(r_at_k, 4),
            "delta": round(delta, 4),
            "tracks": abs(delta) < 0.10,
        })

    return results


def run_threading_test(index, queries, thread_counts, k=10, recall_target=0.95):
    """Measure QPS across different thread counts."""
    results = []
    for threads in thread_counts:
        t0 = time.perf_counter()
        index.search_batch(queries, k=k, n_threads=threads, recall_target=recall_target)
        search_s = time.perf_counter() - t0
        qps = len(queries) / search_s
        results.append({
            "threads": threads,
            "qps": round(qps, 0),
            "latency_us": round(search_s / len(queries) * 1e6, 1),
        })

    return results
