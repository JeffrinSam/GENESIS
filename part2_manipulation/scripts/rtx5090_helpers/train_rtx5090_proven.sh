#!/bin/bash
#
# PROVEN RTX 5090 Configuration for GR00T N1.6 Fine-tuning
# Based on successful training by RX-02333 (GitHub Issue #101)
#
# Hardware: RTX 5090 32GB
# Memory Usage: ~31GB (confirmed working)
# Batch Size: 8 (confirmed working)
#

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  GR00T N1.6 Fine-tuning - RTX 5090 Proven Configuration   ║"
echo "║  Based on GitHub Issue #101 Success Report                ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Activate environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dc_groot

# Set PYTHONPATH
GENESIS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export PYTHONPATH="$GENESIS_ROOT/part2_manipulation:$PYTHONPATH"

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Kill any processes that might be using GPU memory
echo "Checking for GPU processes..."
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
echo ""

# Verify PyTorch setup
echo "Verifying PyTorch configuration..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
echo ""

# Check available GPU memory
echo "GPU Memory Status:"
nvidia-smi --query-gpu=memory.total,memory.free,memory.used --format=csv
echo ""

echo "Starting training with PROVEN RTX 5090 configuration..."
echo "Batch size: 8 (uses ~31GB of 32GB)"
echo ""

# Run training with proven RTX 5090 settings
python scripts/demo_conditioned/train_dc_groot.py \
  --dataset_path ./data/dc_groot_teleop \
  --pretrained_groot nvidia/GR00T-N1.6-3B \
  --output_dir ./checkpoints/dc_groot_rtx5090 \
  --batch_size 8 \
  --gradient_accumulation_steps 2 \
  --max_steps 5000 \
  --bf16 \
  --gradient_checkpointing \
  --run_name dc_groot_rtx5090_batch8 \
  --learning_rate 1e-4 \
  --warmup_steps 500 \
  --save_steps 500 \
  --logging_steps 10 \
  --dataloader_num_workers 4

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Training completed successfully!                          ║"
echo "╚════════════════════════════════════════════════════════════╝"
