# Neural Network-Based Adaptive Filtering of the Spherical Harmonic Method

Benjamin Plumridge, Cory Hauck, Steffen Schotthöfer  
_To appear in the Journal of Scientific Computing, September 2025_

This repository trains and evaluates data-driven filtered spherical harmonics (`FPN`) solvers for the radiation transport equation. It supports both the neural-network filter ansatz and the trained constant-filter baseline.

## Installation

Use [`uv`](https://docs.astral.sh/uv/) for environment creation and package installation.

```bash
git clone https://github.com/benplumridge/NN_FPN.git
cd NN_FPN

uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

Check the install:

```bash
uv run python - <<'PY'
import torch, numpy, matplotlib, yaml
print("torch", torch.__version__)
PY
```

For a CUDA-enabled PyTorch install, install the PyTorch wheel appropriate for your system with `uv pip install ...` before running the experiments.

## Main Workflow

The recommended entry point is the root-level `main.py`, driven by YAML configs in `configs/`.

```bash
uv run python main.py -c configs/reproduce_all.yaml
```

Common variants:

```bash
# Full 1D reproduction: NN and constant ansatz, training and simulation
uv run python main.py -c configs/reproduce_1d.yaml

# Full 2D reproduction workflow
uv run python main.py -c configs/reproduce_2d.yaml

# Quick smoke run for wiring checks
uv run python main.py -c configs/smoke_1d.yaml

# Print planned work without running training/simulation
uv run python main.py -c configs/reproduce_all.yaml --dry-run
```

You can override the config from the command line:

```bash
uv run python main.py -c configs/reproduce_all.yaml --target 1d --ansatz nn --phase train
uv run python main.py -c configs/reproduce_all.yaml --target 1d --ansatz both --phase simulate
uv run python main.py -c configs/reproduce_all.yaml --target 2d --ansatz constant --phase all
```

CLI options:

```text
--target   1d | 2d | all
--ansatz   nn | constant | const | both
--phase    train | simulate | all
--dry-run  show the planned commands/config without executing
```

## Config Files

- `configs/reproduce_all.yaml`: run 1D and 2D, neural-network and constant ansatz.
- `configs/reproduce_1d.yaml`: run only 1D experiments.
- `configs/reproduce_2d.yaml`: run only 2D experiments.
- `configs/smoke_1d.yaml`: small one-epoch 1D run for checking the installation and workflow.

Each config controls:

- moment orders `N`
- number of epochs
- batch sizes
- optimizer settings
- ansatz selection
- simulation scripts
- output result directories


## Weights & Biases Logging

W&B logging is available for training and is disabled by default in most YAML configs. To enable online logging, edit the `workflow.wandb` block in a config:

```yaml
workflow:
  wandb:
    enabled: true
    project: NN_FPN
    mode: online
    group: null
    tags: []
    log_interval: 10
    name_template: "{dimension}-{ansatz}-N{N}-run{replicate}"
```

Then run normally:

```bash
uv run python main.py -c configs/reproduce_1d.yaml --target 1d --ansatz nn --phase train
```

Authenticate once before online W&B logging:

```bash
uv run wandb login
```

The training logger records loss, square-root loss, sampled cross-section means, and filter-strength min/mean/max at `log_interval` epochs. Larger intervals reduce host/GPU synchronization during training. Runs are named with the dimension, ansatz, moment order, and replica index.

## Outputs

The workflow writes checkpoints and generated figures/tables into the corresponding dimension directory.

1D outputs:

```text
1D/trained_models/             neural-network checkpoints
1D/trained_models_const/       constant-ansatz checkpoints
1D/results_nn/                 NN simulation figures
1D/results_const/              constant simulation figures
1D/error_reduction_table*.txt  summary tables
```

2D outputs:

```text
2D/trained_models/             neural-network checkpoints
2D/trained_models_const/       constant-ansatz checkpoints
2D/results_nn/                 NN simulation figures
2D/results_const/              constant simulation figures
2D/error_reduction_*           summary tables when reference files are available
```

## Figure Generation

The 1D plotting scripts are in place and generate scalar-flux/filter-strength figures under `1D/results_*`.

The 2D plotting path is also in place in `2D/src/test_model.py`, but full paper-style line-source and lattice reproduction expects reference files that are not currently committed:

```text
2D/exact_solns/linesource_37.npy
2D/exact_solns/lattice_37_T16.npy
2D/exact_solns/lattice_37_T32.npy
```

When those files are missing, `main.py` falls back to `2D/scripts/test_driver.py` so the workflow still produces available 2D diagnostic figures.

## Legacy Scripts

The original direct scripts remain available:

```bash
cd 1D
uv run python scripts/train_all.py
uv run python scripts/test_all.py

cd ../2D
uv run python scripts/train_all.py
uv run python scripts/test_all.py
```

For reproducible NN-vs-constant runs, prefer the root `main.py` YAML workflow.

## Validation

Run the lightweight validation tests with:

```bash
uv run python -B -m unittest discover -s tests
```

These tests check the known physics fixes, script wiring, YAML configs, and the main CLI entry point without running long training jobs.
