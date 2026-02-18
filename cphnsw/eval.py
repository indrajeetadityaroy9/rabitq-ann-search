"""Canonical evaluation entrypoint."""

import json
from pathlib import Path

import yaml

from cphnsw.bench.run_benchmark import run_benchmark
from cphnsw.datasets import ALL_DATASETS


REQUIRED_SECTIONS = ["run", "data", "eval"]
DEFAULT_CONFIG_PATH = Path("configs/benchmark.yaml")

SECS_PER_MIN = 60.0
MB_PER_GIB = 1024.0
SUMMARY_RECALL_THRESH = 0.95  # recall@10 threshold for QPS summary


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict:
    with path.open() as f:
        cfg = yaml.safe_load(f)

    for section in REQUIRED_SECTIONS:
        if section not in cfg:
            raise ValueError(f"Missing required config section '{section}' in {path}")

    return cfg


def _extract_summary(results_dir: Path) -> list:
    """Extract summary rows from per-dataset result files."""
    summary = []
    for result_file in sorted(results_dir.glob("*_results.json")):
        with open(result_file) as f:
            data = json.load(f)

        dataset = data["metadata"]["dataset"]

        for algo in data["results"]:
            build_min = algo["build_time_s"] / SECS_PER_MIN
            mem_gib = algo["memory_mb"] / MB_PER_GIB

            qps_95 = None
            for point in algo["sweep"]:
                if point["recall_at_10"] >= SUMMARY_RECALL_THRESH:
                    qps_95 = point["qps"]
                    break

            summary.append({
                "dataset": dataset,
                "algorithm": algo["algorithm"],
                "build_time_min": round(build_min, 4),
                "memory_gib": round(mem_gib, 4),
                "qps_at_95_recall_k10": qps_95,
            })

    return summary


def main():
    cfg = load_config()

    run_cfg = cfg["run"]
    data_cfg = cfg["data"]
    eval_cfg = cfg["eval"]

    output_dir = Path(run_cfg["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    dataset = data_cfg["dataset"]
    base_dir = Path(data_cfg["base_dir"]).resolve()
    datasets = ALL_DATASETS if dataset == "all" else [dataset]

    k = eval_cfg["k"]
    n_runs = eval_cfg["n_runs"]

    for ds_name in datasets:
        print(json.dumps({"event": "benchmark_start", "dataset": ds_name}), flush=True)
        outfile = run_benchmark(ds_name, base_dir, k, n_runs, output_dir)
        print(json.dumps({"event": "benchmark_done", "dataset": ds_name, "result_file": outfile}), flush=True)

    summary = _extract_summary(output_dir)

    for entry in summary:
        print(json.dumps({
            "event": "summary",
            "dataset": entry["dataset"],
            "algorithm": entry["algorithm"],
            "build_time_min": entry["build_time_min"],
            "memory_gib": entry["memory_gib"],
            "qps_at_95_recall_k10": entry["qps_at_95_recall_k10"],
        }), flush=True)


if __name__ == "__main__":
    main()
