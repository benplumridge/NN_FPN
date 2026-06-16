#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
CONFIG="${CONFIG:-configs/run_2d_training_simulation.yaml}"
ANSATZ="${ANSATZ:-both}"
PHASE="${PHASE:-all}"
DRY_RUN="${DRY_RUN:-0}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

usage() {
  cat <<'EOF'
Usage: scripts/run_2d_training_simulation.sh

Runs the 2D training + simulation workflow through main.py.

Environment overrides:
  PYTHON=python3                         Python executable
  CONFIG=configs/run_2d_training_simulation.yaml
  ANSATZ=both                            both | nn | constant
  PHASE=all                              train | simulate | all
  DRY_RUN=0                              set to 1 to print planned work only
  MPLBACKEND=Agg                         matplotlib backend for headless runs

Examples:
  scripts/run_2d_training_simulation.sh
  ANSATZ=nn scripts/run_2d_training_simulation.sh
  PHASE=train scripts/run_2d_training_simulation.sh
  DRY_RUN=1 scripts/run_2d_training_simulation.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

args=(-c "$CONFIG" --target 2d --ansatz "$ANSATZ" --phase "$PHASE")
if [[ "$DRY_RUN" == "1" ]]; then
  args+=(--dry-run)
fi

cd "$ROOT"
exec "$PYTHON_BIN" main.py "${args[@]}"
