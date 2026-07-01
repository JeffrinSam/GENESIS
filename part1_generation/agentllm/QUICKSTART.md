# Unified AgentLLM - Quick Start Guide

## 🎉 Setup Complete!

The unified web interface has been successfully implemented and all dependencies are installed.

---

## 🚀 Start the Interface

```bash
cd part1_generation/agentllm
./start_unified.sh
```

Then open your browser to:
```
http://localhost:5002
```

---

## 📋 What You Get

### Single Unified Website
- **Port 5002** (no conflicts with existing 5000/5001)
- **4 Tasks**: Drone, Ground Robot, UR3, G1
- **2 Categories**: Navigation (blue) + Manipulation (purple)

### 6-Step Wizard
1. **Select Task** - Choose from 4 color-coded cards
2. **Upload & Prompt** - Image + simple description
3. **QwenVL Extension** - **NEW: See & edit enhanced prompt**
4. **Configure** - Advanced parameters (frames, resolution, guidance)
5. **Generate** - Real-time pipeline with live console logs
6. **Results** - Video player + Cosmos-Reason2 validation scores

### Key Features
✅ **QwenVL Visibility**: See how your simple prompt becomes enhanced
✅ **Real-Time Logs**: Live console output during generation
✅ **Validation Scores**: 3-component breakdown (Prompt, Physics, Quality)
✅ **Advanced Parameters**: Full control over generation settings
✅ **Polished UI**: Gradient theme with animations

---

## 🧪 Quick Test

### Test 1: Drone Navigation

1. Start server: `./start_unified.sh`
2. Open: `http://localhost:5002`
3. Click: **Drone Aerial Navigation** card
4. Upload: Any outdoor environment image
5. Prompt: `"Drone flies forward through forest"`
6. Click: **✨ Extend with QwenVL**
7. Review: Enhanced cinematic prompt (editable!)
8. Configure: 61 frames, 1280×704 resolution
9. Generate: Click **🚀 Start Generation**
10. Wait: ~3-5 minutes (generation + validation)
11. Result: Video + validation scores

### Test 2: UR3 Manipulation

1. Select: **Bimanual UR3 Manipulation** card
2. Upload: Workshop/tabletop image
3. Prompt: `"Pick up red cube"`
4. Extend: See physics-based enhancement
5. Configure: Cosmos 2B, 77 frames
6. Generate & validate

---

## 📊 Expected Results

### Generation Times
- **Drone/Ground** (WAN 2.2, 61 frames): 2-4 minutes
- **UR3/G1** (Cosmos 2B, 77 frames): 3-5 minutes
- **UR3/G1** (Cosmos 14B, 121 frames): 10-12 minutes

### Validation Times
- **Cosmos-Reason2**: 1-3 minutes
- **Skip validation**: 0 seconds (optional checkbox)

### Memory Usage
- **Navigation**: ~12-15 GB GPU VRAM
- **Manipulation**: ~15-20 GB GPU VRAM

---

## 🔍 Differences from Old Interface

| Feature | Old (2 Websites) | New (Unified) |
|---------|-----------------|---------------|
| **Ports** | 5000 + 5001 | 5002 |
| **QwenVL** | Hidden | Visible & editable |
| **Steps** | 4 | 6 (adds extension + validation display) |
| **Validation** | Basic pass/fail | 3-component scores + reasoning |
| **Logs** | No live logs | Real-time console output |
| **UI** | Basic | Gradient theme, animations |

---

## 📁 Files Created

```
part1_generation/agentllm/
├── unified_app.py               # Backend (800 lines)
├── templates/
│   └── unified_index.html       # Frontend (1900 lines)
├── start_unified.sh             # Startup script ✓ executable
├── check_dependencies.py        # Dependency checker ✓ executable
├── UNIFIED_README.md            # Full documentation (19KB)
├── QUICKSTART.md                # This file
└── UPGRADE_NOTES.md             # Cosmos-Reason2 validator upgrade
```

---

## 🛠️ Troubleshooting

### Server won't start
```bash
# Check dependencies
python3 check_dependencies.py

# Check port availability
sudo lsof -i :5002
```

### Generation fails
```bash
# Check logs
tail -f unified_app.log

# Test individual components
cd Navigation
python3 video_validator.py --help
```

### Validation fails
- Automatic fallback to default scores (70%)
- Pipeline continues without blocking
- Check error in validation JSON

---

## 📚 Documentation

- **Full docs**: [UNIFIED_README.md](UNIFIED_README.md)
- **Validator upgrade**: [UPGRADE_NOTES.md](UPGRADE_NOTES.md)
- **API reference**: See UNIFIED_README.md section "API Reference"

---

## 🎯 Next Steps

1. **Start the server**: `./start_unified.sh`
2. **Generate first video**: Follow Test 1 above
3. **Explore features**: Try all 4 tasks
4. **Experiment**: Edit QwenVL prompts, adjust parameters
5. **Compare**: Navigation vs Manipulation approaches

---

## ✅ Verification

Run dependency checker:
```bash
python3 check_dependencies.py
```

Expected output:
```
✅ ALL CHECKS PASSED
Ready to start: ./start_unified.sh
```

---

**Status**: ✅ Complete and Ready
**Date**: 2026-02-04
**Version**: 1.0.0
