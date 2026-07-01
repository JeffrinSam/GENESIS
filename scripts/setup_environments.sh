#!/bin/bash
# GENESIS — Create all conda environments
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENESIS_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_DIR="$GENESIS_ROOT/environments"

echo "=== GENESIS Environment Setup ==="
echo "GENESIS root: $GENESIS_ROOT"
echo ""

check_conda() {
    if ! command -v conda &> /dev/null; then
        echo "ERROR: conda not found. Install Anaconda or Miniconda first."
        exit 1
    fi
}

create_env() {
    local name=$1
    local yml=$2
    if conda info --envs | grep -q "^$name "; then
        echo "  [SKIP] $name already exists (use --force to recreate)"
    else
        echo "  Creating $name ..."
        conda env create -f "$yml" -q
        echo "  [OK] $name"
    fi
}

check_conda

echo "Creating environments (this may take 5-10 minutes)..."
create_env "genesis-generation"  "$ENV_DIR/environment-generation.yml"
create_env "genesis-navigation"  "$ENV_DIR/environment-navigation.yml"
create_env "genesis-simulation"  "$ENV_DIR/environment-simulation.yml"

echo ""
echo "=== Done ==="
echo "Activate with:"
echo "  conda activate genesis-generation   # Part 1 — video generation"
echo "  conda activate genesis-navigation   # Part 2a — FlowDiT navigation"
echo "  conda activate genesis-simulation   # Simulator — Isaac Sim evaluation"
echo ""
echo "See docs/SETUP.md for upstream model setup (WAN 2.2, Cosmos 2.5, Qwen3-VL)."
