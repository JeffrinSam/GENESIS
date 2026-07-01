#!/bin/bash
# GENESIS — Quick start demo
# Runs a single navigation inference using the FlowDiT V2 model
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENESIS_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== GENESIS Quick Start ==="
echo ""

# Check checkpoint
CHECKPOINT="$GENESIS_ROOT/part2_navigation/flow_constrained_v2/checkpoints/best.pth"
if [ ! -f "$CHECKPOINT" ]; then
    echo "Checkpoint not found. Run: bash scripts/download_checkpoints.sh"
    exit 1
fi

# Run FlowDiT V2 test inference
echo "Running FlowDiT V2 validation..."
conda run -n genesis-navigation python \
    "$GENESIS_ROOT/part2_navigation/flow_constrained_v2/test_inference.py" \
    --checkpoint "$CHECKPOINT"

echo ""
echo "Quick start complete. See part2_navigation/README.md for full usage."
