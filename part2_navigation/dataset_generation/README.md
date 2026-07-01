# Dataset Generation for VideotoNav

**Author**: Jeffrin Sam
**Institution**: Skoltech
**Year**: 2025

This directory contains tools to generate and convert navigation datasets for the VideotoNav project.

## 🎉 **STATUS: ALL DATASETS READY**

✅ **Dataset generation complete!** All 9,078 clips (16.2 hours) are converted and ready for training.

| Metric | Value |
|--------|-------|
| **Total Clips** | 9,078 |
| **Total Duration** | 16.2 hours |
| **Total Size** | 25.6 GB |
| **Format** | MP4 (16fps) + NPY actions |
| **Datasets** | RECON (8,948) + TartanAir (130) + Habitat (50) |

**Location**: `${GENESIS_ROOT}/part2_navigation/dataset/`

---

## 📁 Directory Structure

```
dataset_generation/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
│
├── Scripts (7 files):
│   ├── generate_habitat_dataset.py    # Generate synthetic data (Habitat-Sim)
│   ├── convert_recon.py               # Convert RECON HDF5 → VideotoNav
│   ├── convert_tartanair.py           # Convert TartanAir → VideotoNav
│   ├── download_tartanair.py          # Download TartanAir scenes
│   ├── split_dataset.py               # Split into train/val/test
│   ├── verify_dataset.py              # Validate dataset integrity
│   └── visualize_samples.py           # Visualize dataset samples
│
├── data/                          # Downloaded datasets (88GB total)
│   ├── recon/                     # 71GB - 11,836 HDF5 trajectories ✅
│   ├── tartanair_land/            # 17GB - 10 trajectories ✅
│   └── versioned_data/            # 106MB - 3 Habitat scenes ✅
│
└── ../dataset/                    # Output directory (25.6GB total)
    ├── recon/                     # 8,948 clips (15.8 hours) ✅ READY
    ├── tartanair/                 # 130 clips (0.3 hours) ✅ READY
    └── habitat/                   # 50 clips ✅ READY
```

---

## 🔧 Environment Setup

### Prerequisites
- **Conda** (Anaconda or Miniconda)
- **GPU** (Recommended: RTX 3060+ with 8GB+ VRAM)
- **Disk Space**: 150GB+ (88GB raw data + 100GB for converted datasets)

### Two Environments Needed

#### 1️⃣ Habitat Environment (For Habitat Generation Only)
```bash
# Create environment
conda create -n habitat python=3.9 -y
conda activate habitat

# Install habitat-sim (use v0.2.5 to avoid shader bugs)
conda install -c aihabitat -c conda-forge habitat-sim=0.2.5 -y

# Install other dependencies
pip install opencv-python==4.8.1.78 numpy==1.26.4 tqdm pyyaml h5py scipy pillow==10.4.0
```

#### 2️⃣ Base Environment (For RECON, TartanAir, and Utils)
```bash
# Use your existing Python 3.9+ environment or create new one
conda create -n videonav python=3.9 -y
conda activate videonav

# Install dependencies
pip install opencv-python numpy tqdm pyyaml h5py scipy matplotlib pillow
```

**Why two environments?**
- `habitat-sim` has strict dependency requirements (numpy==1.26.4, pillow==10.4.0)
- Other scripts work with any recent versions
- Only `generate_habitat_dataset.py` needs the habitat environment

---

## 📊 Dataset Status

### ✅ All Datasets Downloaded, Converted & Ready:

| Dataset | Raw Size | Converted Size | Clips | Duration | Status |
|---------|----------|----------------|-------|----------|--------|
| **RECON** | 71GB (11,836 HDF5) | 25GB | 8,948 clips | 15.8 hours | ✅ **Converted & Ready** |
| **TartanAir** | 17GB (10 trajectories) | 652MB | 130 clips | 0.3 hours | ✅ **Converted & Ready** |
| **Habitat** | 106MB (3 scenes) | N/A | 50 clips | 0.1 hours | ✅ **Generated & Ready** |
| **TOTAL** | 88GB | 25.6GB | 9,078 clips | 16.2 hours | ✅ **Ready to Train** |

### 📦 RECON Dataset (UC Berkeley)
- **Source**: UC Berkeley RECON Navigation Dataset
- **Robot**: Jackal (wheeled ground robot)
- **Total**: 11,836 HDF5 files (~23 hours of navigation)
- **Per trajectory**: ~70 frames (7 seconds @ 10 FPS)
- **Data**: RGB images (640×480) + linear/angular velocities + GPS/IMU/LIDAR
- **Environment**: Indoor + outdoor real-world

### 📦 TartanAir Dataset (CMU)
- **Source**: CMU TartanAir
- **Environment**: ConstructionSite (10 trajectories)
- **Per trajectory**: ~1,056 frames
- **Data**: RGB images + depth maps + camera poses
- **Robot**: Aerial drone / ground robot

### 📦 Habitat Scenes
- **Scenes**: apartment_1, van-gogh-room, skokloster-castle
- **Already Generated**: 50 clips (8 seconds @ 16 FPS, 720p)
- **Location**: `../dataset/habitat/`
- **Status**: ✅ **Ready to train**

---

## 🚀 How to Run Each Script

### 1️⃣ `generate_habitat_dataset.py`

**Environment**: `habitat` (requires habitat-sim)

**Purpose**: Generate synthetic navigation clips using Habitat-Sim simulator

**Usage**:
```bash
# Activate habitat environment
conda activate habitat

# Generate 100 clips
python generate_habitat_dataset.py \
    --num_clips 100 \
    --duration 8.0 \
    --fps 16 \
    --embodiment wheeled \
    --output_dir ../dataset/habitat_extra \
    --resolution 720p

# Generate aerial robot clips
python generate_habitat_dataset.py \
    --num_clips 50 \
    --embodiment aerial \
    --output_dir ../dataset/habitat_aerial
```

**Parameters**:
- `--num_clips`: Number of clips to generate (default: 100)
- `--duration`: Clip duration in seconds (default: 8.0)
- `--fps`: Frames per second (default: 16)
- `--embodiment`: Robot type (`wheeled`, `aerial`, `legged`, `humanoid`)
- `--resolution`: Video resolution (`480p`, `720p`, `1080p`)
- `--output_dir`: Output directory
- `--min_quality`: Minimum quality 0-1 (default: 0.7)
- `--seed`: Random seed (default: 42)

**Output**: MP4 videos + NPY actions + metadata.json

**Time**: ~1 minute for 100 clips

---

### 2️⃣ `convert_recon.py`

**Environment**: `videonav` or base Python environment

**Purpose**: Convert RECON HDF5 files to VideotoNav format

**Status**: ✅ **Updated for HDF5 format - COMPLETED**

**Usage**:
```bash
# Activate environment
conda activate videonav  # or your base environment

# Convert all RECON data (ALREADY DONE ✅)
python convert_recon.py \
    --recon_dir data/recon/recon_release \
    --output_dir ../dataset/recon \
    --min_duration 2.0 \
    --max_duration 12.0 \
    --fps 16

# Convert with custom settings
python convert_recon.py \
    --recon_dir data/recon/recon_release \
    --output_dir ../dataset/recon_custom \
    --min_duration 4.0 \
    --fps 16 \
    --source_fps 10
```

**Parameters**:
- `--recon_dir`: Path to RECON HDF5 files
- `--output_dir`: Output directory
- `--min_duration`: Minimum clip duration (default: 2.0s)
- `--max_duration`: Maximum clip duration (default: 12.0s)
- `--fps`: Target FPS (default: 16)
- `--source_fps`: RECON source FPS (default: 10)

**Output**: MP4 videos + NPY actions (differential drive: vx, vy=0, yaw_rate) + metadata.json

**Actual Results**: ✅ **8,948 clips from 11,836 trajectories (15.8 hours, 25GB)**

**Time**: ~50 minutes (3-6 files/second)

---

### 3️⃣ `convert_tartanair.py`

**Environment**: `videonav` or base Python environment

**Purpose**: Convert TartanAir dataset to VideotoNav format

**Status**: ✅ **COMPLETED**

**Usage**:
```bash
# Activate environment
conda activate videonav  # or your base environment

# Convert TartanAir for wheeled robot (ALREADY DONE ✅)
python convert_tartanair.py \
    --tartanair_dir data/tartanair_land \
    --output_dir ../dataset/tartanair \
    --duration 8.0 \
    --fps 16 \
    --embodiment wheeled

# Convert for aerial drone (if needed)
python convert_tartanair.py \
    --tartanair_dir data/tartanair_land \
    --output_dir ../dataset/tartanair_aerial \
    --embodiment aerial \
    --fps 16
```

**Parameters**:
- `--tartanair_dir`: Path to TartanAir data
- `--output_dir`: Output directory
- `--duration`: Clip duration in seconds (default: 8.0)
- `--fps`: Target FPS (default: 16)
- `--embodiment`: Robot type (`wheeled` or `aerial`)
  - `wheeled`: 3D actions (vx, vy, yaw_rate)
  - `aerial`: 4D actions (vx, vy, vz, yaw_rate)

**Output**: MP4 videos + NPY actions + metadata.json

**Actual Results**: ✅ **130 clips from 10 trajectories (0.3 hours, 652MB)**

**Time**: ~1.5 minutes

---

### 4️⃣ `download_tartanair.py`

**Environment**: Any Python 3.9+ environment

**Purpose**: Download additional TartanAir environments

**Usage**:
```bash
# List available environments
python download_tartanair.py --list

# Download specific environments
python download_tartanair.py \
    --env abandonedfactory hospital office \
    --difficulty easy \
    --output_dir data/tartanair_extra

# Download with specific modalities
python download_tartanair.py \
    --env seasidetown \
    --difficulty easy hard \
    --modality image depth pose \
    --output_dir data/tartanair
```

**Parameters**:
- `--list`: Show available environments
- `--env`: Environment names (space-separated)
- `--difficulty`: Difficulty levels (`easy`, `hard`)
- `--modality`: Data types (`image`, `depth`, `pose`, `seg`)
- `--output_dir`: Output directory

**Time**: Varies (5-20 GB per environment, 1-3 hours each)

---

### 5️⃣ `split_dataset.py`

**Environment**: Any Python 3.9+ environment

**Purpose**: Split dataset into train/val/test sets

**Usage**:
```bash
# Split with default ratios (70/15/15)
python split_dataset.py --dataset_dir ../dataset/habitat

# Custom split ratios
python split_dataset.py \
    --dataset_dir ../dataset/recon \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1

# Sequential split (not random)
python split_dataset.py \
    --dataset_dir ../dataset/tartanair \
    --mode sequential
```

**Parameters**:
- `--dataset_dir`: Dataset directory (must contain metadata.json)
- `--train_ratio`: Training set ratio (default: 0.7)
- `--val_ratio`: Validation set ratio (default: 0.15)
- `--test_ratio`: Test set ratio (default: 0.15)
- `--mode`: Split mode (`random`, `sequential`)
- `--seed`: Random seed (default: 42)

**Output**: Updates metadata.json with split labels

**Time**: < 1 second

---

### 6️⃣ `verify_dataset.py`

**Environment**: Any Python 3.9+ environment

**Purpose**: Validate dataset integrity and check for corrupted files

**Usage**:
```bash
# Verify single dataset
python verify_dataset.py --dataset_dir ../dataset/habitat

# Verify multiple datasets
python verify_dataset.py --dataset_dir ../dataset/recon
python verify_dataset.py --dataset_dir ../dataset/tartanair

# Detailed verification
python verify_dataset.py \
    --dataset_dir ../dataset/habitat \
    --check_videos \
    --check_actions \
    --check_metadata
```

**Parameters**:
- `--dataset_dir`: Dataset directory to verify
- `--check_videos`: Verify all videos are playable (default: True)
- `--check_actions`: Verify all action files loadable (default: True)
- `--check_metadata`: Verify metadata consistency (default: True)

**Output**: Prints verification report with any errors

**Time**: 1-5 minutes depending on dataset size

---

### 7️⃣ `visualize_samples.py`

**Environment**: Any Python 3.9+ environment (needs matplotlib)

**Purpose**: Visualize random samples from dataset

**Usage**:
```bash
# Visualize 5 random samples
python visualize_samples.py \
    --dataset_dir ../dataset/habitat \
    --num_samples 5

# Save visualizations
python visualize_samples.py \
    --dataset_dir ../dataset/recon \
    --num_samples 10 \
    --output_dir visualizations \
    --save_images

# Show specific clip
python visualize_samples.py \
    --dataset_dir ../dataset/habitat \
    --clip_id habitat_clip_0000
```

**Parameters**:
- `--dataset_dir`: Dataset directory
- `--num_samples`: Number of samples to visualize (default: 5)
- `--output_dir`: Directory to save images (optional)
- `--save_images`: Save visualizations to files
- `--clip_id`: Visualize specific clip ID

**Output**: Shows/saves video frames with action overlays

**Time**: < 1 minute

---

## 📋 Complete Workflow

### Option 1: Quick Start (Use Existing Data)
```bash
# You already have 50 Habitat clips ready!
cd ../flow_constrained
python training/train.py --dataset_dir ../dataset/habitat
```

### Option 2: Full Dataset Pipeline
```bash
# 1. Convert RECON (real-world data) - RECOMMENDED
conda activate videonav
python convert_recon.py \
    --recon_dir data/recon/recon_release \
    --output_dir ../dataset/recon \
    --fps 16

# 2. Convert TartanAir (diverse environments)
python convert_tartanair.py \
    --tartanair_dir data/tartanair_land \
    --output_dir ../dataset/tartanair \
    --embodiment wheeled

# 3. Verify all datasets
python verify_dataset.py --dataset_dir ../dataset/habitat
python verify_dataset.py --dataset_dir ../dataset/recon
python verify_dataset.py --dataset_dir ../dataset/tartanair

# 4. Split datasets
python split_dataset.py --dataset_dir ../dataset/habitat
python split_dataset.py --dataset_dir ../dataset/recon
python split_dataset.py --dataset_dir ../dataset/tartanair

# 5. Visualize samples
python visualize_samples.py --dataset_dir ../dataset/recon --num_samples 5

# 6. Train model with combined data
cd ../flow_constrained
python training/train.py \
    --dataset_dirs ../dataset/habitat ../dataset/recon ../dataset/tartanair \
    --batch_size 32 \
    --epochs 50
```

---

## 🎯 Environment Quick Reference

| Script | Environment | Install Command |
|--------|-------------|-----------------|
| `generate_habitat_dataset.py` | **habitat** | `conda activate habitat` |
| `convert_recon.py` | videonav | `conda activate videonav` |
| `convert_tartanair.py` | videonav | `conda activate videonav` |
| `download_tartanair.py` | videonav | `conda activate videonav` |
| `split_dataset.py` | videonav | `conda activate videonav` |
| `verify_dataset.py` | videonav | `conda activate videonav` |
| `visualize_samples.py` | videonav | `conda activate videonav` |

---

## 📐 Output Format Specification

All datasets are converted to a unified format:

### Directory Structure:
```
dataset/
├── videos/
│   ├── clip_0000.mp4
│   ├── clip_0001.mp4
│   └── ...
├── actions/
│   ├── clip_0000.npy
│   ├── clip_0001.npy
│   └── ...
└── metadata.json
```

### Video Format:
- **Codec**: MP4V or H.264
- **FPS**: 16 (default)
- **Resolution**: 720p (1280×720) or 480p (854×480)
- **Duration**: 4-12 seconds per clip

### Action Format:
- **File**: NumPy .npy array
- **Shape**: `[T, action_dim]`
  - Wheeled: `[T, 3]` → (vx, vy, yaw_rate)
  - Aerial: `[T, 4]` → (vx, vy, vz, yaw_rate)
- **Units**: m/s for velocities, rad/s for angular
- **Dtype**: float32

### Metadata Format (JSON):
```json
{
  "clip_0000": {
    "source": "habitat|recon|tartanair",
    "embodiment": "wheeled|aerial",
    "duration": 8.0,
    "fps": 16,
    "num_frames": 128,
    "resolution": [720, 1280],
    "split": "train|val|test",
    "video_path": "videos/clip_0000.mp4",
    "action_path": "actions/clip_0000.npy"
  }
}
```

---

## 🔧 Troubleshooting

### Habitat-sim shader errors
**Problem**: "only four-component swizzles are supported"
**Solution**: Use habitat-sim 0.2.5 instead of 0.3.3:
```bash
conda activate habitat
pip uninstall habitat-sim -y
conda install -c aihabitat habitat-sim=0.2.5 -y
```

### RECON conversion fails
**Problem**: Script expects PNG frames but RECON has HDF5
**Solution**: Update `convert_recon.py` to handle HDF5 format (see script header)

### Import errors in habitat environment
**Problem**: `ModuleNotFoundError: No module named 'h5py'`
**Solution**:
```bash
conda activate habitat
pip install h5py
```

### Out of disk space
**Current usage**: 88GB for raw data
**After conversion**: Will need ~100-150GB more
**Solution**: Clean up archives after extraction, or use external drive

### Slow conversion
**Problem**: Converting 11,836 RECON files takes hours
**Solution**: This is normal. Run overnight or use `--max_clips` parameter to limit

### Video playback issues
**Problem**: Generated videos won't play
**Solution**: Install codecs or use different player (VLC recommended)

---

## 💡 Tips & Best Practices

1. **Start Small**: Test with `--max_clips 100` before full conversion
2. **Verify Early**: Run `verify_dataset.py` after each conversion
3. **Save Disk Space**: Delete extracted archives after conversion
4. **Use GPU**: Habitat generation is faster with CUDA support
5. **Parallel Processing**: Run RECON and TartanAir conversions in parallel
6. **Monitor Progress**: Use `tqdm` progress bars (enabled by default)
7. **Backup Metadata**: Save `metadata.json` files separately

---

## 📊 Actual Dataset Sizes (Conversion Complete ✅)

| Dataset | Input Size | Output Size | Clips | Duration |
|---------|-----------|-------------|-------|----------|
| Habitat (generated) | 106MB scenes | N/A | 50 | 0.1 hours |
| RECON (converted) | 71GB HDF5 | 25GB | 8,948 | 15.8 hours |
| TartanAir (converted) | 17GB raw | 652MB | 130 | 0.3 hours |
| **Total** | **88GB** | **25.6GB** | **9,078** | **16.2 hours** |

**Note**: Output size is much smaller than expected because:
- Video compression (MP4 with H.264 codec)
- Many short trajectories were filtered out
- Efficient storage of action arrays (NPY format)

---

## 📞 Support

For issues or questions:
1. Check troubleshooting section above
2. Verify dataset with `verify_dataset.py`
3. Check script help: `python <script>.py --help`
4. Check logs in output directories

---

## 📄 License

MIT License - See main repository LICENSE file

---

**Last Updated**: January 6, 2025

**Status**: ✅ **ALL DATASETS READY TO TRAIN**
- ✅ Habitat: Complete (50 clips, 0.1 hours)
- ✅ RECON: Converted (8,948 clips, 15.8 hours, 25GB)
- ✅ TartanAir: Converted (130 clips, 0.3 hours, 652MB)
- ✅ **TOTAL: 9,078 clips, 16.2 hours of navigation data**

**Next Steps**:
1. Verify datasets: `python verify_dataset.py --dataset_dir ../dataset/recon`
2. Split datasets: `python split_dataset.py --dataset_dir ../dataset --train_ratio 0.8`
3. Start training with combined 16.2 hours of multi-environment navigation data
