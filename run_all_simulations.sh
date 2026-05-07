#!/usr/bin/env bash
set -euo pipefail

# Run all NESS and non-stationary simulations, then regenerate figures.
# Usage:
#   ./run_all_simulations.sh
#   ./run_all_simulations.sh --force

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
FORCE_ARG=""

for arg in "$@"; do
  case "$arg" in
    --force)
      FORCE_ARG="--force"
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      echo "Usage: ./run_all_simulations.sh [--force]" >&2
      exit 1
      ;;
  esac
done

mkdir -p data figures

echo "=== Run NESS simulations ==="
echo "--- NESS width=0.05, lambda in [-50, 0] ---"
"$PYTHON_BIN" simulation_NESS.py --broad_band 0.05 --lambda_min -50 --lambda_max 0 ${FORCE_ARG}

echo "--- NESS width=0.5, lambda in [-150, 150] ---"
"$PYTHON_BIN" simulation_NESS.py --broad_band 0.5 --lambda_min -150 --lambda_max 150 ${FORCE_ARG}

echo "--- NESS width=1.0, lambda in [-150, 150] ---"
"$PYTHON_BIN" simulation_NESS.py --broad_band 1.0 --lambda_min -150 --lambda_max 150 ${FORCE_ARG}

echo "=== Plot NESS figures ==="
"$PYTHON_BIN" plot_NESS.py --widths 0.05 0.5 1.0

echo "=== Run non-stationary simulation ==="
"$PYTHON_BIN" simulation_nonstationary.py --band_width 0.1 ${FORCE_ARG}

echo "=== Plot non-stationary figures ==="
"$PYTHON_BIN" plot_nonstationary.py --band_width 0.1

echo "All simulations and figures are done."
