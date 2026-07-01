#!/bin/bash
# Engineered by Jeffrin Sam
# Contact: jeffrinsam.a@gmail.com
# Part of: Self-Tuning Robotics Video Generation System
# Test Complete Navigation Pipeline
# Tests drone and ground robot with full validation loop

echo "=========================================="
echo "TESTING NAVIGATION PIPELINE WITH VALIDATION"
echo "=========================================="
echo ""
echo "This will test:"
echo "  1. Drone navigation (with validation)"
echo "  2. Ground robot navigation (with validation)"
echo ""
echo "Total time: ~8-15 minutes"
echo ""

# Check environment
if ! python3 -c "import torch" 2>/dev/null; then
    echo "ERROR: torch not found. Did you activate wan2.2?"
    echo "Run: conda activate wan2.2"
    exit 1
fi

# Create outputs directory
mkdir -p outputs

# Create test images
echo "Creating test images..."

# Drone aerial view
python3 -c "
from PIL import Image, ImageDraw
img = Image.new('RGB', (512, 512), color=(135, 206, 235))
draw = ImageDraw.Draw(img)
# Clouds
draw.ellipse([50, 50, 150, 100], fill=(255, 255, 255, 200))
draw.ellipse([200, 80, 320, 130], fill=(255, 255, 255, 200))
draw.ellipse([350, 40, 470, 90], fill=(255, 255, 255, 200))
# Mountains
draw.polygon([(0, 400), (100, 350), (200, 380), (300, 340), (400, 370), (512, 360), (512, 512), (0, 512)],
             fill=(100, 150, 100))
img.save('test_drone.jpg')
print('✓ Created test_drone.jpg')
"

# Ground robot corridor
python3 -c "
from PIL import Image, ImageDraw
img = Image.new('RGB', (512, 512), color=(220, 220, 220))
draw = ImageDraw.Draw(img)
# Floor
draw.rectangle([0, 300, 512, 512], fill=(180, 180, 180))
# Walls (perspective)
draw.polygon([(0, 0), (512, 0), (400, 300), (112, 300)], fill=(200, 200, 200))
# Door
draw.rectangle([200, 100, 312, 280], fill=(139, 90, 60), outline=(100, 60, 40), width=3)
img.save('test_ground.jpg')
print('✓ Created test_ground.jpg')
"

echo "✓ Test images created"
echo ""

# Test 1: Drone with validation
echo "=========================================="
echo "TEST 1/2: DRONE NAVIGATION + VALIDATION"
echo "=========================================="
echo ""

python3 navigation_pipeline.py \
  --task drone \
  --image test_drone.jpg \
  --prompt "Flying over snowy mountain peaks at sunrise" \
  --output outputs/test_drone.mp4

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ DRONE TEST PASSED"
    echo ""
else
    echo ""
    echo "❌ DRONE TEST FAILED"
    exit 1
fi

# Clear GPU memory
echo "Clearing GPU memory..."
python3 -c "
import torch
import gc
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()
print('✓ GPU cache cleared')
"
sleep 2
echo ""

# Test 2: Ground robot with validation
echo "=========================================="
echo "TEST 2/2: GROUND ROBOT + VALIDATION"
echo "=========================================="
echo ""

python3 navigation_pipeline.py \
  --task ground \
  --image test_ground.jpg \
  --prompt "Navigating through modern office hallway" \
  --output outputs/test_ground.mp4

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ GROUND ROBOT TEST PASSED"
    echo ""
else
    echo ""
    echo "❌ GROUND ROBOT TEST FAILED"
    exit 1
fi

# Summary
echo "=========================================="
echo "✅ ALL NAVIGATION TESTS COMPLETED!"
echo "=========================================="
echo ""
echo "Generated videos:"
ls -lh outputs/test_*.mp4
echo ""
echo "Validation reports:"
ls -lh outputs/validation_*.json
echo ""
echo "Test images:"
ls -lh test_*.jpg
echo ""
echo "To view:"
echo "  vlc outputs/test_drone.mp4"
echo "  vlc outputs/test_ground.mp4"
echo ""
echo "To check validation:"
echo "  cat outputs/validation_test_drone.json | jq"
echo "  cat outputs/validation_test_ground.json | jq"
echo ""
echo "✅ Complete navigation pipeline verified!"
echo ""
