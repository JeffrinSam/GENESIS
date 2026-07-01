#!/bin/bash
# =============================================================================
# Test all robots with CSV velocity data (headless)
# =============================================================================
# Usage:
#   ./test_all.sh                                    # Use default test CSV
#   ./test_all.sh /path/to/custom_velocities.csv     # Use custom CSV
#
# Tests:
#   1. Limo (mobile robot) — real physics, DifferentialController
#   2. Drone (Crazyflie)   — kinematic, 4-axis velocity control
#   3. Humanoid (G1)       — WBC controller, Isaac Lab
#
# Each test runs headless and reports final position.
# =============================================================================
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CSV_PATH="${1:-${TEST_CSV:-$SCRIPT_DIR/test_velocities.csv}}"
ISAAC_PYTHON="${ISAAC_SIM_PYTHON:-/opt/isaacsim/python.sh}"

if [ ! -f "$CSV_PATH" ]; then
    echo "ERROR: Test CSV not found: $CSV_PATH"
    exit 1
fi

echo "================================================================="
echo " Simulation Validator — Test Suite"
echo "================================================================="
echo " CSV: $CSV_PATH"
echo " Date: $(date)"
echo "================================================================="
echo ""

PASS=0
FAIL=0

# ── Test 1: Limo (mobile robot) ─────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " TEST 1: Limo (mobile robot, real physics)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -f "$ISAAC_PYTHON" ]; then
    echo "Running Limo CSV replay headless..."
    if timeout 300 "$ISAAC_PYTHON" "$SCRIPT_DIR/mobile_robot/limo_sim.py" \
        --env warehouse --headless \
        --csv-replay "$CSV_PATH" 2>&1 | tail -20; then
        echo "  [PASS] Limo test completed"
        PASS=$((PASS + 1))
    else
        echo "  [FAIL] Limo test failed (exit code: $?)"
        FAIL=$((FAIL + 1))
    fi
else
    echo "  [SKIP] Isaac Sim python.sh not found"
fi
echo ""

# ── Test 2: Drone ────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " TEST 2: Drone (kinematic, Crazyflie CF2X)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -f "$ISAAC_PYTHON" ]; then
    echo "Running Drone CSV replay headless..."
    if timeout 300 "$ISAAC_PYTHON" "$SCRIPT_DIR/drone/drone_sim.py" \
        --env warehouse --headless \
        --csv-replay "$CSV_PATH" 2>&1 | tail -20; then
        echo "  [PASS] Drone test completed"
        PASS=$((PASS + 1))
    else
        echo "  [FAIL] Drone test failed (exit code: $?)"
        FAIL=$((FAIL + 1))
    fi
else
    echo "  [SKIP] Isaac Sim python.sh not found"
fi
echo ""

# ── Test 3: Humanoid (G1) ───────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " TEST 3: Humanoid G1 (WBC, Isaac Lab)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -f "$SCRIPT_DIR/humanoid/run_humanoid.sh" ]; then
    echo "Running Humanoid CSV replay headless..."
    if timeout 300 "$SCRIPT_DIR/humanoid/run_humanoid.sh" \
        --mode csv --csv-path "$CSV_PATH" \
        --headless --num-steps 121 2>&1 | tail -20; then
        echo "  [PASS] Humanoid test completed"
        PASS=$((PASS + 1))
    else
        echo "  [FAIL] Humanoid test failed (exit code: $?)"
        FAIL=$((FAIL + 1))
    fi
else
    echo "  [SKIP] run_humanoid.sh not found"
fi
echo ""

# ── Summary ──────────────────────────────────────────────────────────────────
echo "================================================================="
echo " TEST SUMMARY"
echo "================================================================="
echo " Passed: $PASS"
echo " Failed: $FAIL"
echo " Total:  $((PASS + FAIL))"
echo "================================================================="

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
