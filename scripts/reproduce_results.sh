#!/bin/bash
set -euo pipefail

# Reproduce CP-HNSW benchmark results.
# Prerequisites:
#   pip install -e ".[eval]"
#   Download datasets to data/ (see README.md)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

echo "=== Reproducing CP-HNSW Benchmark Results ==="
echo ""

mkdir -p results

# 1. SIFT-1M recall-QPS benchmark
echo "[1/2] Recall-QPS: SIFT-1M"
python bench/run_benchmark.py --dataset sift1m 2>&1 | tee results/sift1m.log

# 2. Full comparison across all datasets
echo "[2/2] Full benchmark: all datasets"
python bench/run_benchmark.py --dataset all 2>&1 | tee results/all_datasets.log

echo ""
echo "=== Results saved to results/ ==="
