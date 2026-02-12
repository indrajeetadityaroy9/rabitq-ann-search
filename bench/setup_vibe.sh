#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
VIBE_DIR="$SCRIPT_DIR/vibe"
MODULE_SRC="$SCRIPT_DIR/vibe-module"
MODULE_DST="$VIBE_DIR/vibe/algorithms/cphnsw"

if [ ! -d "$VIBE_DIR" ]; then
    printf 'event=vibe_clone path=%s\n' "$VIBE_DIR"
    git clone https://github.com/vector-index-bench/vibe "$VIBE_DIR"
fi

printf 'event=vibe_install_deps path=%s\n' "$VIBE_DIR"
pip install -e "$VIBE_DIR" 2>/dev/null || pip install -r "$VIBE_DIR/requirements.txt" 2>/dev/null || true

mkdir -p "$MODULE_DST"
for f in module.py config.yml image.def; do
    ln -sf "$MODULE_SRC/$f" "$MODULE_DST/$f"
done
touch "$MODULE_DST/__init__.py"

printf 'event=vibe_module_linked path=%s\n' "$MODULE_DST"
printf 'event=vibe_ready run_cmd=%q\n' "cd $VIBE_DIR && python run.py --algorithm cphnsw --dataset <dataset>"
