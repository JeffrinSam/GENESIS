#!/bin/bash
# =============================================================================
# Simulation Validator — Launch Script
# =============================================================================
# Supports: limo (mobile robot), drone, humanoid
#
# Usage:
#   ./run.sh --env warehouse --record                          # Limo (default)
#   ./run.sh --robot drone --env warehouse --record            # Drone
#   ./run.sh --robot humanoid --mode keyboard                  # Humanoid (via Isaac Lab)
#   ./run.sh --robot limo --csv-replay /path/to/data.csv --headless
#
# The humanoid uses a different Python environment (env_isaaclab).
# For direct humanoid control, use: ./run_humanoid.sh [args]
# =============================================================================

ISAAC_PYTHON="${ISAAC_SIM_PYTHON:-/opt/isaacsim/python.sh}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Parse --robot flag (before passing to Python) ────────────────────────────
ROBOT="limo"
NEXT_IS_ROBOT=0
REMAINING_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--robot" ]]; then
        NEXT_IS_ROBOT=1
        continue
    fi
    if [[ "$NEXT_IS_ROBOT" == "1" ]]; then
        ROBOT="$arg"
        NEXT_IS_ROBOT=0
        continue
    fi
    REMAINING_ARGS+=("$arg")
done

# ── Route to correct script ─────────────────────────────────────────────────
case "$ROBOT" in
    limo)
        if [ ! -f "$ISAAC_PYTHON" ]; then
            echo "ERROR: Isaac Sim python.sh not found at $ISAAC_PYTHON"
            exit 1
        fi
        echo "[run.sh] Launching Limo simulation..."
        exec "$ISAAC_PYTHON" "$SCRIPT_DIR/mobile_robot/limo_sim.py" "${REMAINING_ARGS[@]}"
        ;;
    drone)
        if [ ! -f "$ISAAC_PYTHON" ]; then
            echo "ERROR: Isaac Sim python.sh not found at $ISAAC_PYTHON"
            exit 1
        fi
        echo "[run.sh] Launching Drone simulation..."
        exec "$ISAAC_PYTHON" "$SCRIPT_DIR/drone/drone_sim.py" "${REMAINING_ARGS[@]}"
        ;;
    humanoid)
        echo "[run.sh] Launching Humanoid simulation (via run_humanoid.sh)..."
        exec "$SCRIPT_DIR/humanoid/run_humanoid.sh" "${REMAINING_ARGS[@]}"
        ;;
    *)
        echo "ERROR: Unknown robot '$ROBOT'. Use: limo, drone, or humanoid"
        exit 1
        ;;
esac
