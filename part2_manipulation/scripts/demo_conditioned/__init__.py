# Demo-Conditioned GR00T Scripts
#
# Training:
#   python train_dc_groot.py --dataset_path /path/to/data --output_dir ./checkpoints
#
# Data Preparation:
#   python prepare_dc_dataset.py --robot_data /path/to/lerobot --output_dir ./dc_data
#   python convert_lerobot_to_dc.py --input_dir /path/to/lerobot --output_dir ./dc_data
#
# Inference:
#   python run_dc_inference.py --checkpoint ./checkpoints/final --demo_video ./demo.mp4
#
# Deployment:
#   python deploy_dc_groot_g1.py --checkpoint ./checkpoints/final --demo_video ./demo.mp4
