#!/usr/bin/env python3
"""Evaluate an ANN index on recall-vs-QPS.

Usage:
    python scripts/eval.py --config configs/default.yaml
    python scripts/eval.py --config experiments/recall_qps/sift1m.yaml --output results/sift1m/
"""

import argparse
import sys
import time
from pathlib import Path

import yaml
import numpy as np

import cphnsw
from cphnsw.evaluator import Evaluator

def main():
    parser = argparse.ArgumentParser(description="Evaluate CP-HNSW index")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--output", default="results/", help="Output directory")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    idx_cfg = config.get("index", {})
    data_cfg = config.get("data", {})
    search_cfg = config.get("search", {})
    eval_cfg = config.get("eval", {})

    # Load dataset
    print(f"Loading dataset: {data_cfg['dataset']}")
    dataset = cphnsw.load_dataset(data_cfg["dataset"], data_cfg["base_dir"])

    # Instantiate Index
    print("Initializing CP-HNSW index...")
    index = cphnsw.Index(
        dim=dataset["dim"],
        M=idx_cfg.get("M", 32),
        ef_construction=idx_cfg.get("ef_construction", 200),
        seed=idx_cfg.get("seed", 42),
    )

    # Build
    print("Building index...")
    t0 = time.perf_counter()
    index.add(dataset["base"])
    index.finalize()
    build_time = time.perf_counter() - t0
    print(f"  Built in {build_time:.2f}s")

    # Evaluate
    sweep_param = "ef"
    sweep_values = search_cfg.get("ef_values", [10, 20, 40, 80, 100, 200, 400])
    
    print(f"Running evaluation sweep on {sweep_param}...")
    results = []
    
    warmup = eval_cfg.get("warmup_queries", 100)
    k = search_cfg.get("k", 10)
    queries = dataset["queries"]
    groundtruth = dataset["groundtruth"]
    n_queries = len(queries)

    from cphnsw.metrics import recall_at_k, qps

    for val in sweep_values:
        search_kwargs = {sweep_param: val}
        
        # Warmup
        for i in range(min(warmup, n_queries)):
            index.search(queries[i], k=k, **search_kwargs)

        # Timed evaluation
        latencies = []
        all_ids = []
        for i in range(n_queries):
            t0 = time.perf_counter()
            ids, dists = index.search(queries[i], k=k, **search_kwargs)
            latencies.append(time.perf_counter() - t0)
            all_ids.append(ids)

        latencies = np.array(latencies)
        recall_sum = 0.0
        for i in range(n_queries):
            recall_sum += recall_at_k(all_ids[i], groundtruth[i], k)
        mean_recall = recall_sum / n_queries

        res = {
            "param": sweep_param,
            "value": val,
            "recall": mean_recall,
            "qps": qps(latencies),
            "p50_us": np.percentile(latencies, 50) * 1e6,
            "p99_us": np.percentile(latencies, 99) * 1e6,
        }
        results.append(res)
        print(f"{sweep_param}={val:<4} Recall={res['recall']:.4f} QPS={res['qps']:<6.0f} p99={res['p99_us']:.0f}us")

    # Save
    import csv
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / f"{data_cfg['dataset']}_cphnsw_results.csv"
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["param", "value", "recall", "qps", "p50_us", "p99_us"])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to {csv_path}")

if __name__ == "__main__":
    main()