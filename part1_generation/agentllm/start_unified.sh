#!/bin/bash
# Engineered by Jeffrin Sam (jeffrinsam.a@gmail.com)
# Part of: Self-Tuning Robotics Video Generation System
#
# Unified AgentLLM Web Interface Startup Script
# Port: 5002
#

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  AgentLLM Unified Web Interface${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Activate wan2.2 conda environment (needed for torch, transformers, etc.)
echo -e "${YELLOW}[1/5]${NC} Activating wan2.2 conda environment..."
eval "$(conda shell.bash hook)"
conda activate wan2.2 || { echo -e "${RED}ERROR: Failed to activate wan2.2 conda env${NC}"; exit 1; }
echo -e "${GREEN}✓ wan2.2 conda env active${NC}"
python3 --version

# Check required directories
echo -e "${YELLOW}[2/5]${NC} Checking directories..."
BASE_DIR="/mnt/Thesis/JeffrinSam/Part1"

# WAN 2.2
if [ ! -d "$BASE_DIR/Wan2.2" ]; then
    echo -e "${RED}ERROR: WAN 2.2 not found at $BASE_DIR/Wan2.2${NC}"
    exit 1
fi

# Qwen3.5 VL
if [ ! -d "$BASE_DIR/Qwen3-VL" ]; then
    echo -e "${RED}ERROR: Qwen3-VL not found at $BASE_DIR/Qwen3-VL${NC}"
    exit 1
fi

# Cosmos-Reason2
if [ ! -d "$BASE_DIR/cosmos-reason2" ]; then
    echo -e "${RED}ERROR: Cosmos-Reason2 not found at $BASE_DIR/cosmos-reason2${NC}"
    exit 1
fi

echo -e "${GREEN}✓ All directories found${NC}"

# Check Cosmos-Reason2 venv
echo -e "${YELLOW}[3/5]${NC} Checking Cosmos-Reason2 venv..."
COSMOS_VENV="$BASE_DIR/cosmos-reason2/.venv/bin/python3"
if [ ! -f "$COSMOS_VENV" ]; then
    echo -e "${RED}ERROR: Cosmos-Reason2 venv not found at $COSMOS_VENV${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Cosmos-Reason2 venv found${NC}"

# Check Qwen3.5 venv
QWEN_VENV="$BASE_DIR/Qwen3-VL/.venv/bin/python"
if [ ! -f "$QWEN_VENV" ]; then
    echo -e "${RED}ERROR: Qwen3.5 venv not found at $QWEN_VENV${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Qwen3.5 venv found${NC}"

# Check Cosmos-Predict2.5 venv
COSMOS_PREDICT_VENV="$BASE_DIR/cosmos-predict2.5/.venv/bin/python"
if [ ! -f "$COSMOS_PREDICT_VENV" ]; then
    echo -e "${YELLOW}WARNING: Cosmos-Predict2.5 venv not found (manipulation tasks will be unavailable)${NC}"
else
    echo -e "${GREEN}✓ Cosmos-Predict2.5 venv found${NC}"
fi

# Check Flask installation
echo -e "${YELLOW}[4/5]${NC} Checking Flask..."
python3 -c "import flask" 2>/dev/null || {
    echo -e "${YELLOW}WARNING: Flask not found, installing...${NC}"
    pip3 install flask --quiet
}
echo -e "${GREEN}✓ Flask installed${NC}"

# Create uploads and outputs directories
echo -e "${YELLOW}[5/5]${NC} Creating upload/output directories..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$SCRIPT_DIR/uploads"
mkdir -p "$SCRIPT_DIR/outputs"
echo -e "${GREEN}✓ Directories ready${NC}"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Starting Unified Interface...${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "URL: ${YELLOW}http://localhost:5002${NC}"
echo -e "Uploads: ${YELLOW}$SCRIPT_DIR/uploads${NC}"
echo -e "Outputs: ${YELLOW}$SCRIPT_DIR/outputs${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
echo ""

# Start Flask app
cd "$SCRIPT_DIR"
python3 unified_app.py --port 5002 --host 0.0.0.0
