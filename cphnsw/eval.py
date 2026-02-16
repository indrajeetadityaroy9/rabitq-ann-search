"""Config-only evaluation entrypoint.

Usage:
    python -m cphnsw.eval --config configs/benchmark.yaml
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import yaml

from cphnsw.bench.run_benchmark import ALGORITHM_SPECS, run_benchmark
from cphnsw.datasets import ALL_DATASETS


REQUIRED_SECTIONS = ["run", "data", "eval"]


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)

    for section in REQUIRED_SECTIONS:
        if section not in cfg:
            print(f"Error: missing required config section '{section}'", file=sys.stderr)
            sys.exit(1)

    return cfg


def _pin_threads(n_threads: int):
    val = str(n_threads)
    os.environ["OMP_NUM_THREADS"] = val
    os.environ["MKL_NUM_THREADS"] = val
    os.environ["OPENBLAS_NUM_THREADS"] = val


def _extract_summary(results_dir: Path) -> list:
    """Extract tracked metrics from per-dataset result files.

    Returns list of dicts with: dataset, algorithm, build_time_min,
    memory_gib, qps_at_95_recall_k10.
    """
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
    parser = argparse.ArgumentParser(description="CP-HNSW Evaluation")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    run_cfg = cfg["run"]
    data_cfg = cfg["data"]
    eval_cfg = cfg["eval"]

    output_dir = Path(run_cfg.get("output_dir", "results")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _pin_threads(eval_cfg.get("n_threads", 1))

    shutil.copy2(args.config, output_dir / "config.yaml")

    dataset = data_cfg.get("dataset", "sift1m")
    base_dir = Path(data_cfg.get("base_dir", "data")).resolve()
    datasets = ALL_DATASETS if dataset == "all" else [dataset]

    k = eval_cfg.get("k", 100)
    n_runs = eval_cfg.get("n_runs", 3)

    for ds_name in datasets:
        print(f"Running benchmark: {ds_name}", flush=True)
        outfile = run_benchmark(ds_name, base_dir, k, n_runs, output_dir)
        print(f"  -> {outfile}", flush=True)

    summary = _extract_summary(output_dir)
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {summary_path}", flush=True)

    for entry in summary:
        qps_str = f"{entry['qps_at_95_recall_k10']:.1f}" if entry["qps_at_95_recall_k10"] else "N/A"
        print(f"  {entry['dataset']}/{entry['algorithm']}: "
              f"build={entry['build_time_min']:.4f}min "
              f"mem={entry['memory_gib']:.4f}GiB "
              f"qps@95%={qps_str}")


if __name__ == "__main__":
    main()
