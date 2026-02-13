#!/bin/bash
set -euo pipefail


SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$ROOT_DIR/results"
DATA_DIR="$ROOT_DIR/data"
cd "$ROOT_DIR"

mkdir -p "$RESULTS_DIR"

printf 'event=reproduce_step step=all_datasets output=%s\n' "$RESULTS_DIR/all_datasets.log"
python -m cphnsw.bench.run_benchmark \
  --dataset all \
  --k 100 \
  --base-dir "$DATA_DIR" \
  --output-dir "$RESULTS_DIR" 2>&1 | tee "$RESULTS_DIR/all_datasets.log"

printf 'event=reproduce_step step=plot_results\n'
python -m cphnsw.bench.plot_results "$RESULTS_DIR"/*_results.json

printf 'event=reproduce_done results_dir=%s\n' "$RESULTS_DIR"
