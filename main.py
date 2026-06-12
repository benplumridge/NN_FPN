#!/usr/bin/env python3
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "reproduce_all.yaml"


TRAIN_1D_CODE = r"""
import json
import os
import sys

import torch

cfg = json.loads(os.environ["NN_FPN_RUN_CONFIG"])
sys.path.insert(0, os.path.abspath("src"))
from params_common import model_tag_from_params, params, resolve_device, tagged_model_path
from funcs_common import nn_feature_count
from train_model import training

filter_type = int(cfg["filter_type"])
train_cfg = cfg["train"]
base_wandb = cfg.get("wandb", {}) or {}
params.update(train_cfg.get("params", {}))
params["device"] = resolve_device(params.get("device"))
params["filter_type"] = filter_type
params["model_tag"] = model_tag_from_params(params)

for j in range(int(train_cfg.get("num_replicates", 10))):
    for N in train_cfg.get("Ns", [3, 7, 9]):
        N = int(N)
        params["N"] = N
        if filter_type in (1, 2):
            params["num_features"] = nn_feature_count(N, params)
            params["num_hidden"] = int(train_cfg.get("hidden_width", 50))
            params["weight_decay"] = float(train_cfg.get("nn_weight_decay", 1e-5))
            output_path = tagged_model_path(N, j, filter_type, params["model_tag"])
        elif filter_type == 3:
            params["num_features"] = 0
            params["num_hidden"] = 0
            params["weight_decay"] = float(train_cfg.get("constant_weight_decay", 1e-6))
            output_path = tagged_model_path(N, j, filter_type, params["model_tag"])
        else:
            raise ValueError(f"Unsupported 1D filter_type={filter_type}")

        wandb_cfg = dict(base_wandb)
        context = dict(wandb_cfg.get("context", {}))
        context.update(
            {
                "dimension": "1d",
                "ansatz": cfg["ansatz"],
                "replicate": j,
                "N": N,
                "filter_type": filter_type,
                "model_tag": params["model_tag"],
                "training_data": params.get("training_data", "paper"),
                "feature_variant": params.get("feature_variant", "baseline_norm"),
            }
        )
        wandb_cfg["context"] = context
        params["wandb"] = wandb_cfg

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        model = training(params)
        torch.save(model, output_path)
        print(f"saved {output_path}")
"""


TRAIN_2D_CODE = r"""
import json
import os
import sys

import torch

cfg = json.loads(os.environ["NN_FPN_RUN_CONFIG"])
sys.path.insert(0, os.path.abspath("src"))
from params_common import params, resolve_device
from funcs_common import nn_feature_count
from train_model import training

filter_type = int(cfg["filter_type"])
train_cfg = cfg["train"]
base_wandb = cfg.get("wandb", {}) or {}
params.update(train_cfg.get("params", {}))
params["device"] = resolve_device(params.get("device"))
params["filter_type"] = filter_type

init_bias = train_cfg.get("init_bias", {})
ansatz_key = cfg["ansatz"]
for N in train_cfg.get("Ns", [3, 5, 7, 9]):
    N = int(N)
    params["N"] = N
    params["num_features"] = nn_feature_count(N, params)
    hidden_width = train_cfg.get("hidden_width", "half_features")
    if hidden_width == "half_features":
        params["num_hidden"] = params["num_features"] // 2
    else:
        params["num_hidden"] = int(hidden_width)
    params["num_basis"] = (N + 1) * (N + 2) // 2

    bias_by_n = init_bias.get(ansatz_key, {})
    params["init_bias"] = float(bias_by_n.get(str(N), bias_by_n.get(N, 0.0)))

    if filter_type == 0:
        output_path = f"trained_models/model_N{N}.pth"
    elif filter_type == 1:
        output_path = f"trained_models_const/model_N{N}.pth"
    else:
        raise ValueError(f"Unsupported 2D filter_type={filter_type}")

    wandb_cfg = dict(base_wandb)
    context = dict(wandb_cfg.get("context", {}))
    context.update(
        {
            "dimension": "2d",
            "ansatz": ansatz_key,
            "replicate": 0,
            "N": N,
            "filter_type": filter_type,
            "feature_variant": params.get("feature_variant", "baseline_norm"),
        }
    )
    wandb_cfg["context"] = context
    params["wandb"] = wandb_cfg

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model = training(params)
    torch.save(model, output_path)
    print(f"saved {output_path}")
"""


RUN_SCRIPT_CODE = r"""
import json
import os
import runpy
import sys

cfg = json.loads(os.environ["NN_FPN_RUN_CONFIG"])
sys.path.insert(0, os.path.abspath("src"))
from params_common import params, resolve_device

params.update(cfg.get("params", {}))
params["device"] = resolve_device(params.get("device"))
params["filter_type"] = int(cfg["filter_type"])
runpy.run_path(cfg["script"], run_name="__main__")
"""


def load_config(path):
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(
            "PyYAML is required for YAML configs. Install it with: python -m pip install PyYAML"
        ) from exc

    with Path(path).open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Config {path} must contain a YAML mapping at the top level")
    return data


def normalize_ansatz(value):
    if value == "const":
        return "constant"
    return value


def selected_ansatzes(value):
    value = normalize_ansatz(value or "both")
    if value == "both":
        return ["nn", "constant"]
    if value not in {"nn", "constant"}:
        raise SystemExit(f"Unsupported ansatz: {value}")
    return [value]


def selected_targets(value):
    value = (value or "all").lower()
    if value == "all":
        return ["1d", "2d"]
    if value not in {"1d", "2d"}:
        raise SystemExit(f"Unsupported target: {value}")
    return [value]


def phase_flags(value, workflow):
    value = (value or workflow.get("phase") or "all").lower()
    if value == "all":
        return bool(workflow.get("train", True)), bool(workflow.get("simulate", True))
    if value == "train":
        return True, False
    if value == "simulate":
        return False, True
    raise SystemExit(f"Unsupported phase: {value}")


def run_subprocess(code, cwd, payload, env_overrides, dry_run):
    env = os.environ.copy()
    env.update(env_overrides)
    env["NN_FPN_RUN_CONFIG"] = json.dumps(payload)
    command = [sys.executable, "-c", code]

    if dry_run:
        print(f"DRY RUN: cd {cwd} && {sys.executable} -c <runner>")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    subprocess.run(command, cwd=str(cwd), env=env, check=True)


def merged_dict(*configs):
    merged = {}
    for config in configs:
        if config:
            merged.update(config)
    return merged


def filter_type_for(dim_cfg, ansatz):
    mapping = dim_cfg.get("filter_types", {})
    default_map = {
        "1d": {"nn": 1, "constant": 3},
        "2d": {"nn": 0, "constant": 1},
    }
    dim_key = dim_cfg["dimension"]
    return int(mapping.get(ansatz, default_map[dim_key][ansatz]))


def default_1d_model_tag(dim_cfg):
    train_params = dim_cfg.get("train", {}).get("params", {}) or {}
    return (
        train_params.get("model_tag")
        or train_params.get("training_data")
        or train_params.get("training_setup")
        or "paper"
    )


def train_dimension(target, ansatz, dim_cfg, workflow, dry_run):
    dim_dir = ROOT / target.upper()
    filter_type = filter_type_for(dim_cfg, ansatz)
    train_cfg = dim_cfg.get("train", {})
    payload = {
        "ansatz": ansatz,
        "filter_type": filter_type,
        "train": train_cfg,
        "wandb": merged_dict(
            workflow.get("wandb", {}),
            dim_cfg.get("wandb", {}),
            train_cfg.get("wandb", {}),
        ),
    }
    env = {"MPLBACKEND": str(workflow.get("mpl_backend", "Agg"))}
    code = TRAIN_1D_CODE if target == "1d" else TRAIN_2D_CODE
    print(f"==> {target.upper()}: training {ansatz} ansatz")
    run_subprocess(code, dim_dir, payload, env, dry_run)


def refs_available(dim_dir, required_refs):
    return all((dim_dir / ref).exists() for ref in required_refs)


def simulation_scripts(target, dim_cfg):
    sim_cfg = dim_cfg.get("simulate", {})
    if "scripts" in sim_cfg:
        return list(sim_cfg["scripts"])

    if target == "2d":
        required_refs = sim_cfg.get("required_refs", [])
        if required_refs and not refs_available(ROOT / "2D", required_refs):
            if sim_cfg.get("fallback_without_refs", True):
                missing = [
                    ref for ref in required_refs if not (ROOT / "2D" / ref).exists()
                ]
                print("    Missing 2D reference files; using fallback scripts:")
                for ref in missing:
                    print(f"      - 2D/{ref}")
                return list(sim_cfg.get("fallback_scripts", ["scripts/test_driver.py"]))
            raise SystemExit("Missing 2D reference files: " + ", ".join(required_refs))
        return list(sim_cfg.get("paper_scripts", ["scripts/test_all.py"]))

    return [
        "scripts/test_iters.py",
        "scripts/test_iters_reeds.py",
        "scripts/test_all.py",
    ]


def prepare_results_dir(dim_dir, result_dir, workflow, dry_run):
    scratch = dim_dir / "results"
    output = dim_dir / result_dir
    overwrite = bool(workflow.get("overwrite_results", True))

    if dry_run:
        print(f"DRY RUN: prepare results scratch={scratch} output={output}")
        return scratch, output

    if scratch.exists():
        shutil.rmtree(scratch)
    if output.exists():
        if overwrite:
            shutil.rmtree(output)
        else:
            raise SystemExit(f"Output directory already exists: {output}")
    return scratch, output


def finalize_results_dir(scratch, output, dry_run):
    if dry_run:
        print(f"DRY RUN: move {scratch} -> {output} if generated")
        return
    if scratch.exists():
        shutil.move(str(scratch), str(output))


def _enabled_1d_summary_specs(dim_cfg):
    scripts = {str(script).replace("\\", "/") for script in simulation_scripts("1d", dim_cfg)}
    specs = []
    if "scripts/test_iters.py" in scripts:
        specs.append(("error_reduction_table.csv", "error_reduction_table_const.csv"))
    if "scripts/test_iters_reeds.py" in scripts:
        specs.append(
            ("error_reduction_table_reeds.csv", "error_reduction_table_reeds_const.csv")
        )
    return specs


def _read_summary_csv(path):
    with path.open("r", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def _problem_label(problem):
    return problem.replace("_", " ")


def _latex_cell(row, best_value):
    if row is None:
        return r"\textemdash"
    mean_value = row["mean"]
    cell = rf"${mean_value:.4f} \pm {row['std']:.4f}$"
    if best_value is not None and abs(mean_value - best_value) < 1e-12:
        return rf"\bm{{{cell}}}"
    return cell


def _terminal_cell(row, best_value):
    if row is None:
        return "n/a"
    marker = "*" if best_value is not None and abs(row["mean"] - best_value) < 1e-12 else ""
    return f"{row['mean']:.4f}+/-{row['std']:.4f}{marker}"


def _print_1d_latex_table(table_key, times, Ns, data):
    _, problem = table_key
    print()
    print(_problem_label(problem))
    print(r"\begin{tabular}{c c c c c c c c}")
    print(
        r"   & \multicolumn{3}{c}{Neural Network ($r_N^{\text{nn}}$)} "
        r"& \phantom{abc} & \multicolumn{3}{c}{Constant ($r_N^{\text{const}}$)} \\"
    )
    print(r"        \cmidrule{2-4} \cmidrule{6-8}")
    print(r"        $\tf$  & $N=3$ & $N=7$ & $N=9$ & & $N=3$ & $N=7$ & $N=9$ \\")
    print(r"        \midrule")

    for time_value in times:
        nn_cells = []
        const_cells = []
        for N in Ns:
            nn_row = data.get((*table_key, time_value, N, "nn"))
            const_row = data.get((*table_key, time_value, N, "constant"))
            available = [row["mean"] for row in (nn_row, const_row) if row is not None]
            best_value = min(available) if available else None
            nn_cells.append(_latex_cell(nn_row, best_value))
            const_cells.append(_latex_cell(const_row, best_value))
        print(
            f"        {time_value:.1f} & "
            + " & ".join(nn_cells)
            + " & & "
            + " & ".join(const_cells)
            + r" \\"
        )

    print(r"        \bottomrule")
    print(r"\end{tabular}")


def _print_1d_terminal_table(table_key, times, Ns, data):
    _, problem = table_key
    cell_width = 18
    print()
    print(_problem_label(problem))
    nn_header = " ".join(f"{f'NN N={N}':>{cell_width}}" for N in Ns)
    const_header = " ".join(f"{f'Const N={N}':>{cell_width}}" for N in Ns)
    print(f"{'T':>5} {nn_header} | {const_header}")
    print("-" * (6 + len(nn_header) + 3 + len(const_header)))

    for time_value in times:
        nn_cells = []
        const_cells = []
        for N in Ns:
            nn_row = data.get((*table_key, time_value, N, "nn"))
            const_row = data.get((*table_key, time_value, N, "constant"))
            available = [row["mean"] for row in (nn_row, const_row) if row is not None]
            best_value = min(available) if available else None
            nn_cells.append(_terminal_cell(nn_row, best_value))
            const_cells.append(_terminal_cell(const_row, best_value))

        nn_text = " ".join(f"{cell:>{cell_width}}" for cell in nn_cells)
        const_text = " ".join(f"{cell:>{cell_width}}" for cell in const_cells)
        print(f"{time_value:>5.1f} {nn_text} | {const_text}")


def print_1d_paper_summary_tables(dim_cfg, simulated_ansatzes, dry_run):
    if dry_run or {"nn", "constant"} - set(simulated_ansatzes):
        return

    dim_dir = ROOT / "1D"
    specs = _enabled_1d_summary_specs(dim_cfg)
    if not specs:
        return

    missing = []
    csv_paths = []
    for nn_file, const_file in specs:
        for filename in (nn_file, const_file):
            path = dim_dir / filename
            if path.exists():
                csv_paths.append(path)
            else:
                missing.append(path.relative_to(ROOT))

    if missing:
        print("\nPaper-style summary tables skipped; missing CSV files:")
        for path in missing:
            print(f"  - {path}")
        return

    data = {}
    table_order = []
    for csv_path in csv_paths:
        for row in _read_summary_csv(csv_path):
            table_key = (row["table"], row["problem"])
            if table_key not in table_order:
                table_order.append(table_key)
            key = (
                row["table"],
                row["problem"],
                float(row["final_time"]),
                int(row["N"]),
                row["ansatz"],
            )
            data[key] = {
                "mean": float(row["mean_flux_error_reduction"]),
                "std": float(row["std_flux_error_reduction"]),
            }

    if not data:
        return

    table_shapes = []
    print("\nPaper-style summary tables")
    for table_key in table_order:
        times = sorted({key[2] for key in data if key[:2] == table_key})
        Ns = sorted({key[3] for key in data if key[:2] == table_key})
        table_shapes.append((table_key, times, Ns))
        _print_1d_latex_table(table_key, times, Ns, data)

    print("\nTerminal-readable summary tables")
    print("* = lower mean error reduction for that T,N comparison")
    for table_key, times, Ns in table_shapes:
        _print_1d_terminal_table(table_key, times, Ns, data)


def simulate_dimension(target, ansatz, dim_cfg, workflow, dry_run):
    dim_dir = ROOT / target.upper()
    sim_cfg = dim_cfg.get("simulate", {})
    filter_type = filter_type_for(dim_cfg, ansatz)
    label = "const" if ansatz == "constant" else "nn"
    result_template = sim_cfg.get("result_dir_template", "results_{label}")
    result_dir = result_template.format(ansatz=ansatz, label=label)
    scratch, output = prepare_results_dir(dim_dir, result_dir, workflow, dry_run)

    print(f"==> {target.upper()}: running {ansatz} simulations")
    scripts = simulation_scripts(target, dim_cfg)
    env = {"MPLBACKEND": str(workflow.get("mpl_backend", "Agg"))}
    params_payload = dict(sim_cfg.get("params", {}) or {})
    if target == "1d" and "model_tag" not in params_payload:
        params_payload["model_tag"] = default_1d_model_tag(dim_cfg)

    for script in scripts:
        payload = {
            "ansatz": ansatz,
            "filter_type": filter_type,
            "script": script,
            "params": params_payload,
        }
        print(f"    {target.upper()}: {script}")
        run_subprocess(RUN_SCRIPT_CODE, dim_dir, payload, env, dry_run)

    finalize_results_dir(scratch, output, dry_run)


def run_workflow(config, args):
    workflow = config.get("workflow", {})
    targets = selected_targets(args.target or workflow.get("target", "all"))
    ansatzes = selected_ansatzes(args.ansatz or workflow.get("ansatz", "both"))
    do_train, do_simulate = phase_flags(args.phase, workflow)

    for target in targets:
        dim_cfg = dict(config.get(target, {}))
        dim_cfg["dimension"] = target
        if not dim_cfg.get("enabled", True):
            print(f"==> {target.upper()}: disabled in config")
            continue

        simulated_ansatzes = []
        for ansatz in ansatzes:
            if do_train:
                train_dimension(target, ansatz, dim_cfg, workflow, args.dry_run)
            if do_simulate:
                simulate_dimension(target, ansatz, dim_cfg, workflow, args.dry_run)
                simulated_ansatzes.append(ansatz)

        if do_simulate and target == "1d":
            print_1d_paper_summary_tables(dim_cfg, simulated_ansatzes, args.dry_run)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train and evaluate NN_FPN neural-network and constant filter experiments."
    )
    parser.add_argument(
        "-c",
        "--config",
        default=str(DEFAULT_CONFIG),
        help="YAML config path. Default: configs/reproduce_all.yaml",
    )
    parser.add_argument(
        "--target", choices=["1d", "2d", "all"], help="Override config target"
    )
    parser.add_argument(
        "--ansatz",
        choices=["nn", "const", "constant", "both"],
        help="Override config ansatz selection",
    )
    parser.add_argument(
        "--phase",
        choices=["train", "simulate", "all"],
        help="Run only training, only simulations, or both",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without running them",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    run_workflow(config, args)


if __name__ == "__main__":
    main()
