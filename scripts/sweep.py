#!/usr/bin/env python3
"""Parameter sweep: varies one index parameter and evaluates each setting.

Usage:
    python scripts/sweep.py --config configs/default.yaml --param index.M --values 8 16 32 64
"""

import argparse
import copy
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cphnsw
from cphnsw.evaluator import Evaluator

import time


def set_nested(d: dict, key: str, value):
    """Set a nested dict value by dot-separated key."""
    parts = key.split(".")
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = value


def main():
    parser = argparse.ArgumentParser(description="Parameter sweep for CP-HNSW")
    parser.add_argument("--config", required=True, help="Base YAML config file")
    parser.add_argument("--param", required=True, help="Dot-separated parameter to sweep (e.g., index.M)")
    parser.add_argument("--values", nargs="+", required=True, help="Values to sweep over")
    parser.add_argument("--output", default="results/sweep/", help="Output directory")
    args = parser.parse_args()

    with open(args.config) as f:
        base_config = yaml.safe_load(f)

    if cphnsw.Index is None:
        print("ERROR: C++ bindings not available.")
        sys.exit(1)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    for val_str in args.values:
        # Auto-detect type
        try:
            val = int(val_str)
        except ValueError:
            try:
                val = float(val_str)
            except ValueError:
                val = val_str

        config = copy.deepcopy(base_config)
        set_nested(config, args.param, val)

        idx_cfg = config.get("index", {})
        data_cfg = config.get("data", {})
        search_cfg = config.get("search", {})
        eval_cfg = config.get("eval", {})

        print(f"\n=== {args.param}={val} ===")

        dataset = cphnsw.load_dataset(data_cfg["dataset"], data_cfg["base_dir"])

        index = cphnsw.Index(
            dim=dataset["dim"],
            M=idx_cfg.get("M", 32),
            ef_construction=idx_cfg.get("ef_construction", 200),
            seed=idx_cfg.get("seed", 42),
        )

        t0 = time.perf_counter()
        index.add(dataset["base"])
        index.finalize()
        print(f"  Built in {time.perf_counter() - t0:.2f}s")

        evaluator = Evaluator(index, dataset, {
            "k": search_cfg.get("k", 10),
            "ef_values": search_cfg.get("ef_values", [10, 20, 40, 80, 100, 200, 400]),
            "warmup_queries": eval_cfg.get("warmup_queries", 100),
        })

        results = evaluator.run()
        csv_path = output / f"{args.param.replace('.', '_')}_{val}.csv"
        evaluator.save(str(csv_path))

        best = max(results, key=lambda r: r["recall"])
        print(f"  Best: recall={best['recall']:.4f} at ef={best['ef']}, QPS={best['qps']:.0f}")

    print(f"\nAll results saved to {output}/")


if __name__ == "__main__":
    main()
