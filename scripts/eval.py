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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

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

    if cphnsw.Index is None:
        print("ERROR: C++ bindings not available. Build with CPHNSW_BUILD_PYTHON=ON.")
        sys.exit(1)

    # Load dataset
    print(f"Loading dataset: {data_cfg['dataset']}")
    dataset = cphnsw.load_dataset(data_cfg["dataset"], data_cfg["base_dir"])

    # Build index
    index = cphnsw.Index(
        dim=dataset["dim"],
        M=idx_cfg.get("M", 32),
        ef_construction=idx_cfg.get("ef_construction", 200),
        seed=idx_cfg.get("seed", 42),
    )

    print("Building index...")
    t0 = time.perf_counter()
    index.add(dataset["base"])
    index.finalize()
    print(f"  Built in {time.perf_counter() - t0:.2f}s")

    # Evaluate
    evaluator = Evaluator(index, dataset, {
        "k": search_cfg.get("k", 10),
        "ef_values": search_cfg.get("ef_values", [10, 20, 40, 80, 100, 200, 400]),
        "warmup_queries": eval_cfg.get("warmup_queries", 100),
    })

    print("Running evaluation sweep...")
    results = evaluator.run()

    # Print results
    print(f"\n{'ef':>6} {'Recall@k':>10} {'QPS':>10} {'p50 (us)':>10} {'p99 (us)':>10}")
    print("-" * 50)
    for r in results:
        print(f"{r['ef']:>6} {r['recall']:>10.4f} {r['qps']:>10.0f} {r['p50_us']:>10.1f} {r['p99_us']:>10.1f}")

    # Save
    output = Path(args.output)
    csv_path = output / f"{data_cfg['dataset']}_recall_qps.csv"
    evaluator.save(str(csv_path))
    print(f"\nResults saved to {csv_path}")

    # Plot if matplotlib available
    try:
        from cphnsw.plotting import plot_recall_qps
        fig_path = output / f"{data_cfg['dataset']}_recall_qps.png"
        plot_recall_qps(str(csv_path), str(fig_path), label="CP-HNSW")
        print(f"Plot saved to {fig_path}")
    except ImportError:
        print("(Install matplotlib for automatic plots: pip install cphnsw[eval])")


if __name__ == "__main__":
    main()
