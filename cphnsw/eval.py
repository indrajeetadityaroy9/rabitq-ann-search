"""Canonical evaluation entrypoint."""

import argparse
import json
from pathlib import Path

import yaml

from cphnsw.bench.run_benchmark import run_benchmark
from cphnsw.datasets import ALL_DATASETS
from cphnsw.paths import resolve_project_path

SECS_PER_MIN = 60.0
MB_PER_GIB = 1024.0
SUMMARY_RECALL_THRESH = 0.95  # recall@10 threshold for QPS summary


def main():
    parser = argparse.ArgumentParser(description="Run CP-HNSW benchmark evaluation.")
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to benchmark config YAML (required).")
    args = parser.parse_args()

    config_path = resolve_project_path(args.config)
    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    run_cfg = cfg["run"]
    data_cfg = cfg["data"]
    eval_cfg = cfg["eval"]

    output_dir = resolve_project_path(run_cfg["output_dir"])
    dataset = data_cfg["dataset"]
    base_dir = resolve_project_path(data_cfg["base_dir"])
    datasets = ALL_DATASETS if dataset == "all" else [dataset]

    k = eval_cfg["k"]
    n_runs = eval_cfg["n_runs"]

    for ds_name in datasets:
        print(json.dumps({"event": "benchmark_start", "dataset": ds_name}), flush=True)
        outfile = run_benchmark(ds_name, base_dir, k, n_runs, output_dir)
        print(json.dumps({"event": "benchmark_done", "dataset": ds_name, "result_file": outfile}), flush=True)

    for result_file in sorted(output_dir.glob("*_results.json")):
        with result_file.open() as f:
            data = json.load(f)

        dataset_name = data["metadata"]["dataset"]
        for algo in data["results"]:
            build_min = round(algo["build_time_s"] / SECS_PER_MIN, 4)
            mem_gib = round(algo["memory_mb"] / MB_PER_GIB, 4)

            qps_95 = None
            for point in algo["sweep"]:
                if point["recall_at_10"] >= SUMMARY_RECALL_THRESH:
                    qps_95 = point["qps"]
                    break

            print(json.dumps({
                "event": "summary",
                "dataset": dataset_name,
                "algorithm": algo["algorithm"],
                "build_time_min": build_min,
                "memory_gib": mem_gib,
                "qps_at_95_recall_k10": qps_95,
            }), flush=True)


if __name__ == "__main__":
    main()
