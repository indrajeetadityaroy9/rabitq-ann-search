"""Canonical evaluation entrypoint."""

import json
from pathlib import Path

import yaml

from cphnsw.bench.run_benchmark import run_benchmark
from cphnsw.datasets import ALL_DATASETS


REQUIRED_SECTIONS = ["run", "data", "eval"]
DEFAULT_CONFIG_PATH = Path("configs/benchmark.yaml")


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
            build_min = algo["build_time_s"] / 60.0
            mem_gib = algo["memory_mb"] / 1024.0

            qps_95 = None
            for point in algo["sweep"]:
                if point.get("recall_at_10", 0) >= 0.95:
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

    output_dir = Path(run_cfg.get("output_dir", "results")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = data_cfg.get("dataset", "sift1m")
    base_dir = Path(data_cfg.get("base_dir", "data")).resolve()
    datasets = ALL_DATASETS if dataset == "all" else [dataset]

    k = eval_cfg.get("k", 100)
    n_runs = eval_cfg.get("n_runs", 3)

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
