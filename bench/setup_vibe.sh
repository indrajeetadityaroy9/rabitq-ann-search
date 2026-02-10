#!/bin/bash
# Setup VIBE benchmark framework with CP-HNSW algorithm module.
#
# Usage:
#   ./bench/setup_vibe.sh
#
# This script:
#   1. Clones VIBE if not already present
#   2. Symlinks the CP-HNSW algorithm module into VIBE's algorithm directory

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
VIBE_DIR="$SCRIPT_DIR/vibe"
MODULE_SRC="$SCRIPT_DIR/vibe-module"
MODULE_DST="$VIBE_DIR/vibe/algorithms/cphnsw"

# Clone VIBE if needed
if [ ! -d "$VIBE_DIR" ]; then
    echo "Cloning VIBE benchmark framework..."
    git clone https://github.com/vector-index-bench/vibe "$VIBE_DIR"
else
    echo "VIBE already cloned at $VIBE_DIR"
fi

# Install VIBE dependencies
echo "Installing VIBE dependencies..."
pip install -e "$VIBE_DIR" 2>/dev/null || pip install -r "$VIBE_DIR/requirements.txt" 2>/dev/null || true

# Link CP-HNSW module
mkdir -p "$MODULE_DST"
for f in module.py config.yml image.def; do
    ln -sf "$MODULE_SRC/$f" "$MODULE_DST/$f"
done
touch "$MODULE_DST/__init__.py"

echo "CP-HNSW algorithm module linked into VIBE at:"
echo "  $MODULE_DST"
echo ""
echo "To run benchmarks:"
echo "  cd $VIBE_DIR"
echo "  python run.py --algorithm cphnsw --dataset <dataset>"
