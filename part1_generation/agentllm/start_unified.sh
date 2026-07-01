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

# Check required environment variables
echo -e "${YELLOW}[1/5]${NC} Checking environment variables..."
if [ -f "$(dirname "$0")/../../.env" ]; then
    export $(grep -v '^#' "$(dirname "$0")/../../.env" | xargs) 2>/dev/null || true
fi

WAN_ROOT="${WAN_ROOT:-}"
COSMOS_ROOT="${COSMOS_ROOT:-}"
COSMOS_REASON2_ROOT="${COSMOS_REASON2_ROOT:-}"

if [ -z "$WAN_ROOT" ]; then
    echo -e "${RED}ERROR: WAN_ROOT not set. Set it in .env or: export WAN_ROOT=/path/to/Wan2.2${NC}"
    exit 1
fi
if [ -z "$COSMOS_REASON2_ROOT" ]; then
    echo -e "${RED}ERROR: COSMOS_REASON2_ROOT not set. Set it in .env or: export COSMOS_REASON2_ROOT=/path/to/cosmos-reason2${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Environment variables set${NC}"

# Check required directories
echo -e "${YELLOW}[2/5]${NC} Checking directories..."
if [ ! -d "$WAN_ROOT" ]; then
    echo -e "${RED}ERROR: WAN_ROOT directory not found: $WAN_ROOT${NC}"
    exit 1
fi
if [ ! -d "$COSMOS_REASON2_ROOT" ]; then
    echo -e "${RED}ERROR: COSMOS_REASON2_ROOT directory not found: $COSMOS_REASON2_ROOT${NC}"
    exit 1
fi
echo -e "${GREEN}✓ All directories found${NC}"

# Check Cosmos-Reason2 venv
echo -e "${YELLOW}[3/5]${NC} Checking Cosmos-Reason2 venv..."
COSMOS_VENV="$COSMOS_REASON2_ROOT/.venv/bin/python3"
if [ ! -f "$COSMOS_VENV" ]; then
    echo -e "${RED}ERROR: Cosmos-Reason2 venv not found at $COSMOS_VENV${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Cosmos-Reason2 venv found${NC}"

# Check Cosmos-Predict2.5 venv (optional)
if [ -n "$COSMOS_ROOT" ] && [ -f "$COSMOS_ROOT/.venv/bin/python" ]; then
    echo -e "${GREEN}✓ Cosmos-Predict2.5 venv found${NC}"
else
    echo -e "${YELLOW}WARNING: COSMOS_ROOT not set or venv missing — manipulation tasks will be unavailable${NC}"
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
