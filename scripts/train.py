#!/usr/bin/env python3
"""Build an ANN index from a dataset.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config experiments/recall_qps/sift1m.yaml
"""

import argparse
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cphnsw


def main():
    parser = argparse.ArgumentParser(description="Build CP-HNSW index")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--output", default="results/", help="Output directory")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    idx_cfg = config.get("index", {})
    data_cfg = config.get("data", {})

    # Load dataset
    print(f"Loading dataset: {data_cfg['dataset']} from {data_cfg['base_dir']}")
    dataset = cphnsw.load_dataset(data_cfg["dataset"], data_cfg["base_dir"])
    dim = dataset["dim"]
    n = len(dataset["base"])
    print(f"  {n} vectors, dim={dim}")

    # Build index
    if cphnsw.Index is None:
        print("ERROR: C++ bindings not available. Build with CPHNSW_BUILD_PYTHON=ON.")
        sys.exit(1)

    index = cphnsw.Index(
        dim=dim,
        M=idx_cfg.get("M", 32),
        ef_construction=idx_cfg.get("ef_construction", 200),
        seed=idx_cfg.get("seed", 42),
    )

    print(f"Building index (M={idx_cfg.get('M', 32)}, ef_c={idx_cfg.get('ef_construction', 200)})...")
    t0 = time.perf_counter()
    index.add(dataset["base"])
    index.finalize(verbose=True)
    build_time = time.perf_counter() - t0

    print(f"  Built {index.size} vectors in {build_time:.2f}s ({index.size / build_time:.0f} vec/s)")

    # Save build stats
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    stats_path = output / "build_stats.yaml"
    with open(stats_path, "w") as f:
        yaml.dump({
            "dataset": data_cfg["dataset"],
            "n_vectors": n,
            "dim": dim,
            "M": idx_cfg.get("M", 32),
            "ef_construction": idx_cfg.get("ef_construction", 200),
            "build_time_s": round(build_time, 3),
            "throughput_vecs_per_s": round(index.size / build_time, 1),
        }, f)
    print(f"  Stats saved to {stats_path}")


if __name__ == "__main__":
    main()
