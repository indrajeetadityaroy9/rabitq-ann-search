"""CLI entrypoint for ``python -m cphnsw``."""

import argparse
import json
from pathlib import Path

import yaml

from cphnsw.datasets import ALL_DATASETS
from cphnsw.eval import (
    MB_PER_GIB,
    SECS_PER_MIN,
    run_benchmark,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="cphnsw",
        description="Run CP-HNSW benchmark evaluation.",
    )
    parser.add_argument(
        "--config", type=Path, required=True,
        help="Path to benchmark config YAML.",
    )
    args = parser.parse_args(argv)

    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg["run"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = cfg["data"]["dataset"]
    base_dir = Path(cfg["data"]["base_dir"])
    datasets = ALL_DATASETS if dataset == "all" else [dataset]

    k = cfg["eval"]["k"]
    n_runs = cfg["eval"]["n_runs"]

    all_outputs = []
    for ds_name in datasets:
        print(
            json.dumps({"event": "benchmark_start", "dataset": ds_name}),
            flush=True,
        )
        output = run_benchmark(ds_name, base_dir, k, n_runs, output_dir)
        all_outputs.append(output)

    for output in all_outputs:
        dataset_name = output["metadata"]["dataset"]
        for algo in output["results"]:
            build_min = round(algo["build_time_s"] / SECS_PER_MIN, 4)
            mem_gib = round(algo["memory_mb"] / MB_PER_GIB, 4)

            summary: dict = {
                "event": "summary",
                "dataset": dataset_name,
                "algorithm": algo["algorithm"],
                "build_time_min": build_min,
                "memory_gib": mem_gib,
                "recall_at_10": algo["recall_at_10"],
                "qps": algo["qps"],
            }
            print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
