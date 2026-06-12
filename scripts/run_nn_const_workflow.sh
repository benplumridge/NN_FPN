#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
TARGET="${1:-all}"
TRAIN_NN="${TRAIN_NN:-1}"
TRAIN_CONST="${TRAIN_CONST:-1}"
RUN_SIM="${RUN_SIM:-1}"
NUM_1D_TRAINS="${NUM_1D_TRAINS:-10}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

usage() {
  cat <<'EOF'
Usage: scripts/run_nn_const_workflow.sh [1d|2d|all]

Environment overrides:
  PYTHON=python3          Python executable to use
  TRAIN_NN=1             Train neural-network ansatz checkpoints
  RUN_SIM=1              Run simulations/figure generation after each training phase
  TRAIN_CONST=1          Train constant-ansatz checkpoints
  NUM_1D_TRAINS=10       Number of 1D model replicas per N
  TRAINING_DATA_1D=paper 1D training-data mode used for default checkpoint tag
  MODEL_TAG_1D=paper     Explicit 1D checkpoint tag, overriding TRAINING_DATA_1D
  MPLBACKEND=Agg         Matplotlib backend for headless runs

Outputs:
  1D/trained_models/             NN checkpoints
  1D/trained_models_const/       constant-ansatz checkpoints
  1D/results_nn/                 NN figures from scripts/test_all.py
  1D/results_const/              constant-ansatz figures from scripts/test_all.py
  1D/error_reduction_table*.txt  NN/constant summary tables
  2D/trained_models/             NN checkpoints
  2D/trained_models_const/       constant-ansatz checkpoints
  2D/results_nn/                 NN figures
  2D/results_const/              constant-ansatz figures
  2D/error_reduction_*           NN/constant summary tables when paper refs are available
EOF
}

run_with_filter() {
  local dim="$1"
  local script_path="$2"
  local filter_type="$3"

  (cd "$ROOT/$dim" && DIM="$dim" FILTER_TYPE_OVERRIDE="$filter_type" SCRIPT_PATH="$script_path" "$PYTHON" - <<'PY'
import os
import runpy
import sys

sys.path.insert(0, os.path.abspath("src"))
from params_common import params

params["filter_type"] = int(os.environ["FILTER_TYPE_OVERRIDE"])
if os.environ.get("DIM") == "1D":
    from params_common import model_tag_from_params

    params["training_data"] = os.environ.get(
        "TRAINING_DATA_1D", params.get("training_data", "paper")
    )
    params["model_tag"] = model_tag_from_params(
        os.environ.get("MODEL_TAG_1D", params["training_data"])
    )
runpy.run_path(os.environ["SCRIPT_PATH"], run_name="__main__")
PY
  )
}

train_1d() {
  local filter_type="$1"
  local label="$2"

  echo "==> 1D: training ${label} ansatz"
  (cd "$ROOT/1D" && FILTER_TYPE="$filter_type" NUM_TRAINS="$NUM_1D_TRAINS" "$PYTHON" - <<'PY'
import os
import sys
import torch

sys.path.insert(0, os.path.abspath("src"))
from params_common import model_tag_from_params, params, tagged_model_path
from train_model import training

filter_type = int(os.environ["FILTER_TYPE"])
num_trains = int(os.environ["NUM_TRAINS"])

params["filter_type"] = filter_type
params["training_data"] = os.environ.get("TRAINING_DATA_1D", params.get("training_data", "paper"))
params["model_tag"] = model_tag_from_params(os.environ.get("MODEL_TAG_1D", params["training_data"]))
params["num_IC"] = 4
params["batch_size"] = 64
params["num_epochs"] = 200
params["learning_rate"] = 1e-1
params["momentum_factor"] = 0.9
params["sigs_max"] = 1
params["GD_optimizer"] = "Adam"
params["tt_flag"] = 0
params["IC_idx"] = 0
params["device"] = os.environ.get("DEVICE_1D", "cpu")
params["ablation_idx"] = 0
params["obj_idx"] = 0

for j in range(num_trains):
    for N in [3, 7, 9]:
        params["N"] = N
        if filter_type in (1, 2):
            params["num_features"] = 2 * N + 4
            params["num_hidden"] = 50
            params["weight_decay"] = 1e-5
            output_path = tagged_model_path(N, j, filter_type, params["model_tag"])
        elif filter_type == 3:
            params["num_features"] = 0
            params["num_hidden"] = 0
            params["weight_decay"] = 1e-6
            output_path = tagged_model_path(N, j, filter_type, params["model_tag"])
        else:
            raise ValueError(f"Unsupported 1D filter_type={filter_type}")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        model = training(params)
        torch.save(model, output_path)
        print(f"saved {output_path}")
PY
  )
}

run_1d_sim() {
  local filter_type="$1"
  local label="$2"

  echo "==> 1D: running ${label} simulations and tables"
  (cd "$ROOT/1D" && rm -rf results "results_${label}")
  run_with_filter "1D" "scripts/test_iters.py" "$filter_type"
  run_with_filter "1D" "scripts/test_iters_reeds.py" "$filter_type"
  run_with_filter "1D" "scripts/test_all.py" "$filter_type"
  (cd "$ROOT/1D" && if [[ -d results ]]; then mv results "results_${label}"; fi)
}

train_2d() {
  local filter_type="$1"
  local label="$2"

  echo "==> 2D: training ${label} ansatz"
  (cd "$ROOT/2D" && FILTER_TYPE="$filter_type" "$PYTHON" - <<'PY'
import os
import sys
import torch

sys.path.insert(0, os.path.abspath("src"))
from params_common import params
from train_model import training

filter_type = int(os.environ["FILTER_TYPE"])
init_bias_by_N = {
    0: {3: 5.0, 5: 5.0, 7: 5.0, 9: 5.0},
    1: {3: 31.75, 5: 27.27, 7: 16.52, 9: 13.62},
}

params["filter_type"] = filter_type
params["num_IC"] = 5
params["batch_size"] = 5
params["num_epochs"] = 200
params["learning_rate"] = 1e-2
params["momentum_factor"] = 0.9
params["sigs_max"] = 1
params["GD_optimizer"] = "Adam"
params["tt_flag"] = 0
params["IC_idx"] = 0
params["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for N in [3, 5, 7, 9]:
    params["N"] = N
    params["num_features"] = 2 * (N + 1) + 2
    params["num_hidden"] = params["num_features"] // 2
    params["num_basis"] = (N + 1) * (N + 2) // 2
    params["init_bias"] = init_bias_by_N[filter_type][N]

    if filter_type == 0:
        output_path = f"trained_models/model_N{N}.pth"
    elif filter_type == 1:
        output_path = f"trained_models_const/model_N{N}.pth"
    else:
        raise ValueError(f"Unsupported 2D filter_type={filter_type}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model = training(params)
    torch.save(model, output_path)
    print(f"saved {output_path}")
PY
  )
}

have_2d_paper_refs() {
  [[ -f "$ROOT/2D/exact_solns/linesource_37.npy" \
    && -f "$ROOT/2D/exact_solns/lattice_37_T16.npy" \
    && -f "$ROOT/2D/exact_solns/lattice_37_T32.npy" ]]
}

run_2d_sim() {
  local filter_type="$1"
  local label="$2"

  echo "==> 2D: running ${label} simulations"
  (cd "$ROOT/2D" && rm -rf results "results_${label}")

  if have_2d_paper_refs; then
    run_with_filter "2D" "scripts/test_all.py" "$filter_type"
  else
    echo "    Missing 2D paper reference .npy files; running scripts/test_driver.py fallback instead."
    echo "    To reproduce paper line-source/lattice figures, provide exact_solns/linesource_37.npy, lattice_37_T16.npy, and lattice_37_T32.npy."
    run_with_filter "2D" "scripts/test_driver.py" "$filter_type"
  fi

  (cd "$ROOT/2D" && if [[ -d results ]]; then mv results "results_${label}"; fi)
}

run_1d_workflow() {
  if [[ "$TRAIN_NN" == "1" ]]; then train_1d 1 nn; fi
  if [[ "$RUN_SIM" == "1" ]]; then run_1d_sim 1 nn; fi
  if [[ "$TRAIN_CONST" == "1" ]]; then train_1d 3 const; fi
  if [[ "$RUN_SIM" == "1" ]]; then run_1d_sim 3 const; fi
}

run_2d_workflow() {
  if [[ "$TRAIN_NN" == "1" ]]; then train_2d 0 nn; fi
  if [[ "$RUN_SIM" == "1" ]]; then run_2d_sim 0 nn; fi
  if [[ "$TRAIN_CONST" == "1" ]]; then train_2d 1 const; fi
  if [[ "$RUN_SIM" == "1" ]]; then run_2d_sim 1 const; fi
}

case "$TARGET" in
  1d) run_1d_workflow ;;
  2d) run_2d_workflow ;;
  all) run_1d_workflow; run_2d_workflow ;;
  -h|--help|help) usage ;;
  *) usage; exit 2 ;;
esac

echo "==> workflow complete"
