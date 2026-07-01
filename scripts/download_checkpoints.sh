#!/bin/bash
# GENESIS — Download pretrained checkpoints from HuggingFace Hub
# HuggingFace model repos: https://huggingface.co/JeffrinSam
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENESIS_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== GENESIS Checkpoint Downloader ==="

if ! command -v huggingface-cli &> /dev/null; then
    echo "Installing huggingface-hub..."
    pip install huggingface-hub -q
fi

# FlowDiT V2 — navigation model (Part 2a)
# Repo: https://huggingface.co/JeffrinSam/genesis-flowdit-v2
echo ""
echo "Downloading FlowDiT V2 checkpoint (Part 2a — navigation)..."
huggingface-cli download JeffrinSam/genesis-flowdit-v2 \
    best.pth \
    --repo-type model \
    --local-dir "$GENESIS_ROOT/part2_navigation/flow_constrained_v2/checkpoints"

# FlowDiT V3 Humanoid — inference-only
# Repo: https://huggingface.co/JeffrinSam/genesis-flowdit-v3-humanoid
echo ""
echo "Downloading FlowDiT V3 Humanoid checkpoint..."
huggingface-cli download JeffrinSam/genesis-flowdit-v3-humanoid \
    flowdit_v3_humanoid_best.pt \
    --repo-type model \
    --local-dir "$GENESIS_ROOT/part2_navigation/flow_constrained_v3_humanoid/checkpoints"

# DC-GR00T LoRA adapter — manipulation model (Part 2b) [UNDER DEVELOPMENT]
# Repo: https://huggingface.co/JeffrinSam/genesis-dc-groot-adapter
echo ""
echo "Downloading DC-GR00T LoRA adapter (Part 2b — manipulation, research preview)..."
huggingface-cli download JeffrinSam/genesis-dc-groot-adapter \
    adapter_model.safetensors adapter_config.json \
    --repo-type model \
    --local-dir "$GENESIS_ROOT/part2_manipulation/checkpoints/dc_groot_adapter"

echo ""
echo "=== Checkpoints downloaded ==="
echo ""
echo "NOTE: DC-GR00T (Part 2b) requires NVIDIA GR00T-N1.6-3B as base model."
echo "Download separately:"
echo "  huggingface-cli download nvidia/GR00T-N1.6-3B --repo-type model"
