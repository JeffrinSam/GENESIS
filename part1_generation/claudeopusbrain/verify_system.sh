#!/bin/bash
# Engineered by Jeffrin Sam
# Contact: jeffrinsam.a@gmail.com
# Part of: Self-Tuning Robotics Video Generation System
# System Health Check for IROS 2026 Experiments
# Run this before starting your 100-task experiments

echo "========================================================================"
echo "🔍 SYSTEM HEALTH CHECK — Claudeopusbrain v2.1"
echo "========================================================================"
echo ""

ERRORS=0
WARNINGS=0

# Test 1: Conda environment
echo "1. Checking conda environment..."
if conda env list | grep -q "wan2.2"; then
    echo "   ✅ wan2.2 conda environment exists"
else
    echo "   ❌ wan2.2 conda environment not found"
    ERRORS=$((ERRORS+1))
fi
echo ""

# Test 2: Claude API key
echo "2. Checking Claude API key..."
if [ -n "$ANTHROPIC_API_KEY" ]; then
    KEY_PREFIX="${ANTHROPIC_API_KEY:0:7}"
    KEY_SUFFIX="${ANTHROPIC_API_KEY: -4}"
    echo "   ✅ API key set: ${KEY_PREFIX}...${KEY_SUFFIX}"
else
    echo "   ⚠️  ANTHROPIC_API_KEY not set"
    echo "      Set with: export ANTHROPIC_API_KEY='your-key-here'"
    echo "      Get key from: https://console.anthropic.com/"
    echo "      (Only needed for Claude Opus/Sonnet models)"
    WARNINGS=$((WARNINGS+1))
fi
echo ""

# Test 3: Ollama server (for free models)
echo "3. Checking Ollama server (for Llama/Qwen)..."
if curl -s -f "${OLLAMA_SERVER:-http://localhost:11434}/health" > /dev/null 2>&1; then
    echo "   ✅ Ollama server is running"
else
    echo "   ⚠️  Ollama server not reachable at ${OLLAMA_SERVER:-http://localhost:11434}"
    echo "      (Only needed for Llama/Qwen models, not Claude)"
    WARNINGS=$((WARNINGS+1))
fi
echo ""

# Test 4: GPU check
echo "4. Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -n "$GPU_INFO" ]; then
        GPU_NAME=$(echo "$GPU_INFO" | cut -d',' -f1)
        GPU_MEM=$(echo "$GPU_INFO" | cut -d',' -f2)
        echo "   ✅ GPU: $GPU_NAME"
        echo "      Free VRAM: ${GPU_MEM} MB"

        if [ "$GPU_MEM" -lt 10000 ]; then
            echo "      ⚠️  Low VRAM warning (< 10GB free)"
            echo "         Consider closing other GPU processes"
            WARNINGS=$((WARNINGS+1))
        fi
    else
        echo "   ❌ No GPU detected"
        ERRORS=$((ERRORS+1))
    fi
else
    echo "   ❌ nvidia-smi not found"
    ERRORS=$((ERRORS+1))
fi
echo ""

# Test 5: Python dependencies
echo "5. Checking Python dependencies..."
MISSING_DEPS=0

# Check in wan2.2 environment
if conda run -n wan2.2 python3 -c "import anthropic" 2>/dev/null; then
    echo "   ✅ anthropic package installed"
else
    echo "   ❌ anthropic package missing"
    echo "      Install with: conda run -n wan2.2 pip install anthropic"
    MISSING_DEPS=$((MISSING_DEPS+1))
fi

if conda run -n wan2.2 python3 -c "import requests" 2>/dev/null; then
    echo "   ✅ requests package installed"
else
    echo "   ❌ requests package missing"
    MISSING_DEPS=$((MISSING_DEPS+1))
fi

if [ $MISSING_DEPS -gt 0 ]; then
    ERRORS=$((ERRORS+1))
fi
echo ""

# Test 6: AgentLLM components
echo "6. Checking AgentLLM components..."
AGENTLLM_BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../agentllm" 2>/dev/null && pwd)"

if [ -d "$AGENTLLM_BASE" ]; then
    echo "   ✅ AgentLLM directory exists"
else
    echo "   ❌ AgentLLM directory not found: $AGENTLLM_BASE"
    ERRORS=$((ERRORS+1))
fi

if [ -f "$AGENTLLM_BASE/Manipulation/cosmos_generate.py" ]; then
    echo "   ✅ Manipulation pipeline found"
else
    echo "   ❌ Manipulation pipeline not found"
    ERRORS=$((ERRORS+1))
fi

if [ -d "${QWEN_ROOT:-/opt/qwen3-vl}" ] || [ -d "$(dirname "${BASH_SOURCE[0]}")/../qwen_prompt_expansion" ]; then
    echo "   ✅ Qwen3-VL found"
else
    echo "   ❌ Qwen3-VL not found"
    ERRORS=$((ERRORS+1))
fi
echo ""

# Test 7: Core scripts
echo "7. Checking core scripts..."
if [ -f "./run_self_tuning.py" ]; then
    echo "   ✅ run_self_tuning.py exists"
else
    echo "   ❌ run_self_tuning.py not found"
    ERRORS=$((ERRORS+1))
fi

if [ -f "./run_batch_experiments.py" ]; then
    echo "   ✅ run_batch_experiments.py exists"
else
    echo "   ❌ run_batch_experiments.py not found"
    ERRORS=$((ERRORS+1))
fi

if [ -f "./src/claude_brain.py" ]; then
    echo "   ✅ claude_brain.py exists"
else
    echo "   ❌ claude_brain.py not found"
    ERRORS=$((ERRORS+1))
fi
echo ""

# Test 8: Disk space
echo "8. Checking disk space..."
DISK_AVAILABLE=$(df -h . | tail -1 | awk '{print $4}' | sed 's/G//')
DISK_AVAILABLE_INT=${DISK_AVAILABLE%.*}

if [ "$DISK_AVAILABLE_INT" -gt 100 ]; then
    echo "   ✅ Disk space: ${DISK_AVAILABLE}G available"
else
    echo "   ⚠️  Low disk space: ${DISK_AVAILABLE}G available"
    echo "      100 tasks × 5 iterations × 5 sec videos ≈ 50GB needed"
    WARNINGS=$((WARNINGS+1))
fi
echo ""

# Summary
echo "========================================================================"
echo "📊 SUMMARY"
echo "========================================================================"

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo "✅ ALL CHECKS PASSED — System is ready for 100-task experiments!"
    echo ""
    echo "Next steps:"
    echo "  1. Fill in tasks: cp tasks_template.json my_100_tasks.json"
    echo "  2. Quick test: python3 run_self_tuning.py --task '...' --task-type g1 --image x.jpg --model opus --max-iterations 1"
    echo "  3. Full run: python3 run_batch_experiments.py --tasks my_100_tasks.json --model opus --cost-budget 200"
    echo ""
elif [ $ERRORS -eq 0 ]; then
    echo "⚠️  ${WARNINGS} WARNING(S) — System mostly ready, but check warnings above"
    echo ""
    echo "You can proceed, but some features may not work:"
    echo "  - No API key → Can't use Claude models (use --model llama or --model qwen)"
    echo "  - No Ollama server → Can't use Llama/Qwen (use --model opus or --model sonnet)"
    echo "  - Low disk space → May run out during long experiments"
    echo ""
else
    echo "❌ ${ERRORS} ERROR(S), ${WARNINGS} WARNING(S) — Fix errors before proceeding"
    echo ""
    echo "Common fixes:"
    echo "  - Install anthropic: conda run -n wan2.2 pip install anthropic"
    echo "  - Activate environment: conda activate wan2.2"
    echo "  - Set API key: export ANTHROPIC_API_KEY='your-key-here'"
    echo "  - Check GPU: nvidia-smi"
    echo ""
fi

echo "========================================================================"

exit $ERRORS
