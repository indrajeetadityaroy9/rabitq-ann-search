"""Run ANN benchmark sweeps and persist JSON results."""

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Callable

import faiss
import hnswlib
import numpy as np
import psutil

from cphnsw.datasets import ALL_DATASETS, load_dataset
from cphnsw.metrics import recall_at_k

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


def emit(event: str, **fields):
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


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


def get_rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


ALGORITHM_SPECS = [
    {
        "algorithm": "cphnsw-flat-1bit",
        "family": "cphnsw",
        "bits": 1,
        "hierarchical": False,
        "param_name": "recall_target",
        "param_values": [0.80, 0.90, 0.95, 0.97, 0.99],
    },
    {
        "algorithm": "cphnsw-flat-2bit",
        "family": "cphnsw",
        "bits": 2,
        "hierarchical": False,
        "param_name": "recall_target",
        "param_values": [0.80, 0.90, 0.95, 0.97, 0.99],
    },
    {
        "algorithm": "cphnsw-flat-4bit",
        "family": "cphnsw",
        "bits": 4,
        "hierarchical": False,
        "param_name": "recall_target",
        "param_values": [0.80, 0.90, 0.95, 0.97, 0.99],
    },
    {
        "algorithm": "cphnsw-hnsw-1bit",
        "family": "cphnsw",
        "bits": 1,
        "hierarchical": True,
        "param_name": "recall_target",
        "param_values": [0.80, 0.90, 0.95, 0.97, 0.99],
    },
    {
        "algorithm": "hnswlib-M16",
        "family": "hnswlib",
        "M": 16,
        "ef_construction": 200,
        "param_name": "ef",
        "param_values": [100, 120, 160, 200, 400, 800, 1200, 2000],
    },
    {
        "algorithm": "hnswlib-M32",
        "family": "hnswlib",
        "M": 32,
        "ef_construction": 200,
        "param_name": "ef",
        "param_values": [100, 120, 160, 200, 400, 800, 1200, 2000],
    },
    {
        "algorithm": "hnswlib-M64",
        "family": "hnswlib",
        "M": 64,
        "ef_construction": 200,
        "param_name": "ef",
        "param_values": [100, 120, 160, 200, 400, 800, 1200, 2000],
    },
    {
        "algorithm": "faiss-hnsw-M32",
        "family": "faiss-hnsw",
        "M": 32,
        "ef_construction": 200,
        "param_name": "ef",
        "param_values": [100, 120, 160, 200, 400, 800, 1200, 2000],
    },
    {
        "algorithm": "faiss-ivfpq",
        "family": "faiss-ivfpq",
        "nlist": 4096,
        "M_pq": 32,
        "nbits_pq": 8,
        "param_name": "nprobe",
        "param_values": [1, 2, 4, 8, 16, 32, 64, 128, 256],
    },
    {
        "algorithm": "faiss-ivfopq",
        "family": "faiss-ivfopq",
        "nlist": 4096,
        "M_pq": 32,
        "nbits_pq": 8,
        "param_name": "nprobe",
        "param_values": [1, 2, 4, 8, 16, 32, 64, 128, 256],
    },
]


def run_benchmark(dataset_name: str, base_dir: Path,
                  k: int, n_runs: int, output_dir: Path):
    emit("benchmark_dataset_start", dataset=dataset_name)

    ds = load_dataset(dataset_name, base_dir=str(base_dir))

    base = ds["base"]
    queries = ds["queries"]
    gt = ds["groundtruth"].astype(np.int64)
    dim = ds["dim"]

    emit(
        "benchmark_dataset_loaded",
        dataset=dataset_name,
        n_base=int(len(base)),
        n_queries=int(len(queries)),
        dim=int(dim),
        metric="l2",
        k=int(k),
        n_runs=int(n_runs),
    )

    # Precompute ground truth distances once (constant across all algorithms/sweeps)
    adr_k = min(k, 10)
    gt_ids = gt[:, :adr_k].astype(np.int64)
    gt_dists = np.sum((base[gt_ids] - queries[:, None, :]) ** 2, axis=2)

    results = []

    for spec in ALGORITHM_SPECS:
        family = spec["family"]
        algorithm = spec["algorithm"]
        emit("benchmark_algorithm_start", dataset=dataset_name, algorithm=algorithm)

        index = None
        gc.collect()
        rss_before = get_rss_mb()
        t0 = time.perf_counter()

        if family == "cphnsw":
            import cphnsw

            if spec["hierarchical"]:
                index = cphnsw.HNSWIndex(dim=dim, bits=spec["bits"])
            else:
                index = cphnsw.Index(dim=dim, bits=spec["bits"])
            index.add(base)
            index.finalize()
        elif family == "hnswlib":
            index = hnswlib.Index(space="l2", dim=dim)
            index.init_index(
                max_elements=len(base),
                ef_construction=spec["ef_construction"],
                M=spec["M"],
            )
            index.set_num_threads(1)
            index.add_items(base, np.arange(len(base)))
        elif family == "faiss-hnsw":
            faiss.omp_set_num_threads(1)
            index = faiss.IndexHNSWFlat(dim, spec["M"])
            index.hnsw.efConstruction = spec["ef_construction"]
            index.add(base)
        elif family == "faiss-ivfpq":
            faiss.omp_set_num_threads(1)
            m_pq = min(spec["M_pq"], dim)
            while dim % m_pq != 0 and m_pq > 1:
                m_pq -= 1
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFPQ(quantizer, dim, spec["nlist"], m_pq, spec["nbits_pq"])
            index.train(base)
            index.add(base)
        elif family == "faiss-ivfopq":
            faiss.omp_set_num_threads(1)
            m_pq = min(spec["M_pq"], dim)
            while dim % m_pq != 0 and m_pq > 1:
                m_pq -= 1
            index = faiss.index_factory(
                dim, f"OPQ{m_pq},IVF{spec['nlist']},PQ{m_pq}x{spec['nbits_pq']}"
            )
            index.train(base)
            index.add(base)
            ivf_sub = faiss.extract_index_ivf(index)
        else:
            raise ValueError(f"Unsupported algorithm family: {family}")

        build_time = time.perf_counter() - t0
        gc.collect()
        rss_after = get_rss_mb()
        mem_mb = max(0.0, rss_after - rss_before)

        emit(
            "benchmark_algorithm_built",
            dataset=dataset_name,
            algorithm=algorithm,
            build_time_s=round(build_time, 3),
            memory_mb=round(mem_mb, 3),
        )

        sweep = []
        for pval in spec["param_values"]:
            if family == "cphnsw":
                def search_fn(batch, _rt=float(pval)):
                    ids, _ = index.search_batch(
                        batch, k=k, n_threads=1, recall_target=_rt
                    )
                    return np.asarray(ids)
            elif family == "hnswlib":
                def search_fn(batch, _ef=int(pval)):
                    index.set_ef(max(_ef, k))
                    labels, _ = index.knn_query(batch, k=k, num_threads=1)
                    return labels
            elif family == "faiss-hnsw":
                def search_fn(batch, _ef=int(pval)):
                    faiss.omp_set_num_threads(1)
                    index.hnsw.efSearch = max(_ef, k)
                    _, ids = index.search(batch, k)
                    return ids
            elif family == "faiss-ivfpq":
                def search_fn(batch, _nprobe=int(pval)):
                    faiss.omp_set_num_threads(1)
                    index.nprobe = _nprobe
                    _, ids = index.search(batch, k)
                    return ids
            elif family == "faiss-ivfopq":
                def search_fn(batch, _nprobe=int(pval)):
                    faiss.omp_set_num_threads(1)
                    ivf_sub.nprobe = _nprobe
                    _, ids = index.search(batch, k)
                    return ids
            else:
                raise ValueError(f"Unsupported algorithm family: {family}")

            ids, qps_val, med_time = timed_search(search_fn, queries, n_warmup=1, n_runs=n_runs)
            r1 = recall_at_k(ids, gt, 1)
            r10 = recall_at_k(ids, gt, min(k, 10))
            r100 = recall_at_k(ids, gt, min(k, 100))
            lat_us = med_time / len(queries) * 1e6

            # ADR: exact L2 for result neighbors, divided by precomputed GT distances
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

            emit(
                "benchmark_point",
                dataset=dataset_name,
                algorithm=algorithm,
                param_name=spec["param_name"],
                param_value=float(pval),
                recall_at_1=round(r1, 4),
                recall_at_10=round(r10, 4),
                recall_at_100=round(r100, 4),
                adr=round(adr, 6),
                qps=round(qps_val, 3),
                median_latency_us=round(lat_us, 3),
            )

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
    emit("benchmark_dataset_done", dataset=dataset_name, output=str(outfile))
    return str(outfile)


def parse_args():
    parser = argparse.ArgumentParser(description="ANN Benchmark Suite")
    parser.add_argument(
        "--dataset",
        default="sift1m",
        help=f"Dataset name ({', '.join(ALL_DATASETS)}, all)",
    )
    parser.add_argument(
        "--base-dir",
        default="data",
        help="Base directory containing named datasets",
    )
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--output-dir", default="results")
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    datasets = ALL_DATASETS if args.dataset == "all" else [args.dataset]

    result_files = []
    for ds_name in datasets:
        f = run_benchmark(ds_name, base_dir, args.k, args.n_runs, output_dir)
        result_files.append(f)

    for f in result_files:
        emit("benchmark_run_output", output=f)


if __name__ == "__main__":
    main()
