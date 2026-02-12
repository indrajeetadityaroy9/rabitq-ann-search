#!/bin/bash
set -euo pipefail


SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$ROOT_DIR/results"
DATA_DIR="$ROOT_DIR/data"
cd "$ROOT_DIR"

mkdir -p "$RESULTS_DIR"

printf 'event=reproduce_step step=sift1m output=%s\n' "$RESULTS_DIR/sift1m.log"
python -m cphnsw.bench.run_benchmark \
  --dataset sift1m \
  --base-dir "$DATA_DIR" \
  --output-dir "$RESULTS_DIR" 2>&1 | tee "$RESULTS_DIR/sift1m.log"

printf 'event=reproduce_step step=all output=%s\n' "$RESULTS_DIR/all_datasets.log"
python -m cphnsw.bench.run_benchmark \
  --dataset all \
  --base-dir "$DATA_DIR" \
  --output-dir "$RESULTS_DIR" 2>&1 | tee "$RESULTS_DIR/all_datasets.log"

printf 'event=reproduce_done results_dir=%s\n' "$RESULTS_DIR"
