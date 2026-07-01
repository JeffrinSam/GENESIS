#!/bin/bash
# Startup script for Main Pipeline Website
# Part of GENESIS — https://github.com/JeffrinSam/GENESIS

echo "========================================"
echo "Main Pipeline - Robotics Video Generation"
echo "========================================"
echo ""

# Check required env vars
if [ -z "$WAN_ROOT" ] || [ -z "$COSMOS_ROOT" ]; then
    echo "Warning: WAN_ROOT or COSMOS_ROOT not set. Set them in .env or export before running."
    echo "Example: export WAN_ROOT=/path/to/Wan2.2 && export COSMOS_ROOT=/path/to/cosmos-predict2.5"
fi

# Check if Flask is installed
if ! python3 -c "import flask" 2>/dev/null; then
    echo "Flask not found. Installing..."
    pip3 install flask werkzeug
fi

# Create necessary directories
mkdir -p uploads outputs

# Start server
echo "Starting Flask server..."
echo ""
echo "Access the website at: http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo ""

python3 app.py
