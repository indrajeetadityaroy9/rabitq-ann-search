#!/bin/bash
set -euo pipefail

# Reproduce all paper results.
# Prerequisites:
#   pip install -e ".[eval]"
#   Download datasets to data/ (see README.md)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

echo "=== Reproducing CP-HNSW Paper Results ==="
echo ""

# 1. Recall-QPS on SIFT-1M
echo "[1/4] Recall-QPS: SIFT-1M"
python scripts/eval.py --config experiments/recall_qps/sift1m.yaml --output results/recall_qps/

# 2. Memory scaling across dimensions
echo "[2/4] Memory: Dimension Scaling"
python scripts/sweep.py --config experiments/memory/dimensions.yaml \
    --param index.M --values 16 32 64 --output results/memory/

# 3. Construction scalability
echo "[3/4] Construction: Dataset Size Scaling"
python scripts/eval.py --config experiments/construction/scaling.yaml --output results/construction/

# 4. Thread scalability
echo "[4/4] Construction: Thread Scaling"
python scripts/sweep.py --config experiments/construction/threads.yaml \
    --param eval.num_threads --values 1 2 4 8 16 --output results/construction_threads/

echo ""
echo "=== All results saved to results/ ==="
echo "Generate plots with: python -c \"from cphnsw.plotting import *; plot_recall_qps('results/recall_qps/sift1m_recall_qps.csv', 'results/recall_qps.png')\""
