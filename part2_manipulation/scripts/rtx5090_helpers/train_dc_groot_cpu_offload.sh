#!/bin/bash
#
# DC-GR00T Training with CPU Offloading
# Uses GPU (32GB) + System RAM (62GB) = 94GB total
#

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  DC-GR00T Training with CPU Offloading                     ║"
echo "║  GPU: 32GB + System RAM: 62GB = 94GB Total Resources      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Activate environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dc_groot

# Set PYTHONPATH
export PYTHONPATH=/mnt/Thesis/JeffrinSam/Part2/vidtomani/Isaac-GR00T:$PYTHONPATH

# Memory optimizations
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ACCELERATE_MIXED_PRECISION=bf16

# Show resources
echo "System Resources:"
free -h | grep "Mem:"
nvidia-smi --query-gpu=memory.total,memory.free --format=csv
echo ""

echo "Starting training with CPU offloading..."
echo "This will use both GPU and system RAM"
echo ""

# Use accelerate for automatic device placement and CPU offloading
accelerate launch \
  --mixed_precision=bf16 \
  --num_processes=1 \
  --num_cpu_threads_per_process=8 \
  scripts/demo_conditioned/train_dc_groot.py \
  --dataset_path ./data/dc_groot_teleop \
  --pretrained_groot nvidia/GR00T-N1.6-3B \
  --output_dir ./checkpoints/dc_groot_cpu_offload \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --max_steps 5000 \
  --bf16 \
  --run_name dc_groot_cpu_offload \
  --learning_rate 1e-4 \
  --warmup_steps 500 \
  --save_steps 500 \
  --logging_steps 10

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Training completed!                                        ║"
echo "╚════════════════════════════════════════════════════════════╝"
