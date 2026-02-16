#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

python -m cphnsw.eval --config configs/benchmark.yaml
python -m cphnsw.bench.plot_results results/*_results.json
