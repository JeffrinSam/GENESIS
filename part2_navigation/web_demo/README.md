# Video-to-Navigation Web Application

A web interface for running video-to-navigation inference with drag-and-drop video upload, model selection, visualization, and data export.

## Features

- 🎥 **Drag & Drop Video Upload** - Easy video file upload with instant preview
- 🤖 **Dual Model Support** - Choose between Model 1 (Flow-Constrained) or Model 2 (FlowDiT V2)
- 👁️ **Video Preview** - See thumbnail of uploaded video before inference
- 📊 **Interactive Visualizations** - View trajectory plots and velocity vectors
- 📥 **Data Export** - Download predictions in CSV, Excel, or TXT format
- ⚡ **Real-time Processing** - See inference progress and results
- 🔄 **Automatic Environment Switching** - Each model runs in its own conda environment

## Setup

### 1. Environment Setup

**IMPORTANT**: This web app uses **two different conda environments** - one for each model:

- **Model 1 (Flow-Constrained)**: Uses `flow_training` conda environment
- **Model 2 (FlowDiT V2)**: Uses `flowdit_v2_py310` conda environment

The app automatically handles environment switching via wrapper scripts. Make sure both environments exist:

```bash
# Check Model 1 environment
conda activate flow_training
python -c "import torch; print('Model 1 env OK')"

# Check Model 2 environment  
conda activate genesis-navigation
python -c "import torch; print('Model 2 env OK')"
```

### 2. Install Web App Dependencies

**Option A: Use Setup Script (Recommended)**
```bash
cd web_app
./setup.sh
```

**Option B: Manual Setup**
```bash
cd web_app
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Option C: Use Existing Conda Environment**
```bash
cd web_app
conda activate flow_training  # or any Python 3.8+ environment
pip install -r requirements.txt
```

### 3. Ensure Model Checkpoints Exist

- **Model 1**: `../flow_constrained/checkpoints/wheeled/best_model.pth`
- **Model 2**: `../flow_constrained_v2/checkpoints/best.pth`

### 4. Update Environment Paths (if needed)

If your conda environments are in a different location, edit:
- `run_inference_model1.sh` - Update conda path and environment name
- `run_inference_model2.sh` - Update conda path and environment name

### 5. Run the Application

**Using startup script:**
```bash
./start.sh
```

**Or manually:**
```bash
# If using virtual environment
source venv/bin/activate

# Run the app
python3 app.py
```

The application will start on `http://localhost:5000`

**Note**: 
- The Flask app runs in your current Python environment (or venv)
- Each model inference runs in its own conda environment automatically via wrapper scripts

## Usage

1. **Upload Video**: Drag and drop a video file or click to browse
   - Video preview will appear automatically
   - Shows file name and size
2. **Select Model**: Choose Model 1 or Model 2
   - Settings will change based on selected model
3. **Configure Settings**:
   - **Model 1**: Robot embodiment (wheeled/legged/aerial/humanoid), trajectory mode, device (CUDA/CPU)
   - **Model 2**: Current observation mode (first/middle/last), device (CUDA/CPU)
4. **Run Inference**: Click "Run Inference" button
   - App will activate the correct conda environment automatically
   - Shows loading spinner during processing
5. **View Results**: 
   - 3D trajectory visualization
   - Velocity component plots
   - Trajectory statistics (num frames, FPS, predictions)
6. **Download Data**: Export predictions in CSV, Excel, or TXT format
   - Each file includes velocity vectors (vx, vy, yaw_rate)
   - Trajectory coordinates if available (x, y, heading)

## API Endpoints

### POST `/api/upload`
Upload a video file.

**Request**: `multipart/form-data` with `file` field

**Response**:
```json
{
  "success": true,
  "filename": "video.mp4",
  "filepath": "/path/to/file"
}
```

### POST `/api/preview_video`
Generate a preview thumbnail of uploaded video.

**Request**: `multipart/form-data` with `video` field

**Response**:
```json
{
  "success": true,
  "preview": "data:image/jpeg;base64,..."
}
```

### POST `/api/inference`
Run inference on uploaded video.

**Request**:
```json
{
  "filename": "video.mp4",
  "model_type": "model1",
  "trajectory_mode": true,
  "embodiment": "wheeled",
  "current_obs_mode": "middle",
  "device": "cuda"
}
```

**Response**:
```json
{
  "success": true,
  "predictions": [[vx, vy, yaw], ...],
  "trajectory": [[x, y, theta], ...],
  "num_frames": 100,
  "visualization": "base64_encoded_image",
  "viz_filename": "video_model1_viz.png"
}
```

### POST `/api/download/<format>`
Download predictions in specified format (csv, excel, txt).

**Request**:
```json
{
  "predictions": [[vx, vy, yaw], ...],
  "trajectory": [[x, y, theta], ...],
  "filename": "output"
}
```

## File Structure

```
web_app/
├── app.py                           # Flask backend with all endpoints
├── run_inference_model1.sh          # Activates flow_training conda env
├── run_inference_model2.sh          # Activates flowdit_v2_py310 conda env
├── templates/
│   └── index.html                  # Frontend interface
├── inference_wrappers/
│   ├── inference_model1.py          # Model 1 inference (silent mode)
│   └── inference_model2.py          # Model 2 inference (silent mode)
├── uploads/                         # Uploaded videos (auto-created)
├── outputs/                         # Generated visualizations (auto-created)
├── venv/                           # Virtual environment
├── requirements.txt                # Python dependencies
├── setup.sh                        # Auto setup script
├── start.sh                        # Startup script
├── README.md                       # This file
├── QUICK_START.md                 # Quick start guide
├── ENVIRONMENT_SETUP.md           # Detailed environment setup
└── PACKAGE_INFO.md                # Package information
```

## Troubleshooting

### CUDA Out of Memory
- Set device to "CPU" in the Device dropdown
- Or reduce video resolution before uploading

### Model Not Found
- Ensure checkpoints exist in the specified paths:
  - Model 1: `../flow_constrained/checkpoints/wheeled/best_model.pth`
  - Model 2: `../flow_constrained_v2/checkpoints/best.pth`
- Check file permissions

### Inference Error (Model 1)
- Check that `flow_training` conda environment exists
- Verify environment path in `run_inference_model1.sh`
- Ensure model dependencies are installed in that environment

### Inference Error (Model 2)
- Check that `flowdit_v2_py310` conda environment exists
- Verify environment path in `run_inference_model2.sh`
- Model 2 automatically resizes video frames to compatible dimensions
- If frame dimensions are incompatible, error will be shown

### Import/Syntax Errors
- Make sure you're in the correct virtual environment: `source venv/bin/activate`
- Install all requirements: `pip install -r requirements.txt`
- Check that both conda environments are properly set up

### Port Already in Use
- The app runs on port 5000 by default
- To use a different port, edit the last line in `app.py`:
  ```python
  app.run(host='0.0.0.0', port=5001, debug=True)
  ```

## Notes

- Maximum file size: 500MB
- Supported formats: MP4, AVI, MOV, MKV, WEBM
- Processing time depends on video length and model complexity
- Visualizations are generated server-side and displayed in browser

## Author

Jeffrin Sam - Skoltech 2026

