#!/usr/bin/env python3
"""Run 1D simulations for the best checkpoint from each NN sweep category."""

import argparse
import csv
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
ONE_D_ROOT = ROOT / "1D"


@dataclass(frozen=True)
class BestModel:
    N: int
    model_tag: str
    hidden_width: int
    feature_variant: str
    feature_normalization: str
    material_feature_normalization: str
    feature_log_scale: float
    feature_log_clip: tuple[float, float]
    include_material_scale_features: bool
    include_material_ratios: bool


BEST_MODELS = {
    3: BestModel(
        N=3,
        model_tag="sweep_n3_7412e87a",
        hidden_width=64,
        feature_variant="log_material_only",
        feature_normalization="none",
        material_feature_normalization="none",
        feature_log_scale=0.1,
        feature_log_clip=(0.0, 20.0),
        include_material_scale_features=True,
        include_material_ratios=True,
    ),
    7: BestModel(
        N=7,
        model_tag="sweep_n7_4dd48523",
        hidden_width=512,
        feature_variant="no_norm_log",
        feature_normalization="none",
        material_feature_normalization="sample",
        feature_log_scale=0.1,
        feature_log_clip=(0.0, 10.0),
        include_material_scale_features=False,
        include_material_ratios=True,
    ),
    9: BestModel(
        N=9,
        model_tag="sweep_n9_7f2ab1ea",
        hidden_width=256,
        feature_variant="baseline_norm",
        feature_normalization="none",
        material_feature_normalization="sample",
        feature_log_scale=1.0,
        feature_log_clip=(0.0, 40.0),
        include_material_scale_features=True,
        include_material_ratios=False,
    ),
}

STANDARD_CASES = [
    (0, "Gaussian", (0.5, 1.0)),
    (1, "Vanishing_cross_section", (0.5, 1.0)),
    (2, "Discontinuous_cross_section", (0.5, 1.0)),
]
REEDS_CASE = (6, "Reeds", (5.0, 10.0))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the 1D benchmark simulations using the best NN sweep model for each N."
    )
    parser.add_argument(
        "--output-dir",
        default="results_nn_best_sweeps",
        help="Directory under 1D/ where generated plots are moved after the run.",
    )
    parser.add_argument(
        "--csv",
        default="error_reduction_table_best_sweeps.csv",
        help="CSV filename written under 1D/ with the combined results.",
    )
    parser.add_argument(
        "--num-tests",
        type=int,
        default=1,
        help="Number of model replicas to test for each N. The sweep winners use replica 0.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto, cpu, cuda, cuda:0, etc.",
    )
    parser.add_argument(
        "--skip-reeds",
        action="store_true",
        help="Skip the long Reeds simulations.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not delete an existing 1D/<output-dir> before moving new plots.",
    )
    parser.add_argument(
        "--print-results",
        action="store_true",
        help="Print detailed per-run testing output in addition to the summary rows.",
    )
    return parser.parse_args()


def as_float(value):
    tensor = torch.as_tensor(value).detach().cpu()
    if tensor.numel() != 1:
        raise ValueError(
            f"Expected scalar test result, got shape {tuple(tensor.shape)}"
        )
    return float(tensor.item())


def reset_grid(params, *, xl, xr, num_x, T):
    L = xr - xl
    dx = L / num_x
    dt = dx / 2
    params.update(
        {
            "xl": xl,
            "xr": xr,
            "L": L,
            "num_x": num_x,
            "dx": dx,
            "dt": dt,
            "num_t": int((T + dt) // dt),
            "x_edges": torch.linspace(xl, xr, num_x + 1),
            "x": torch.linspace(xl + dx / 2, xr - dx / 2, num_x),
        }
    )


def configure_model(params, model, nn_feature_count):
    params.update(
        {
            "N": model.N,
            "model_tag": model.model_tag,
            "num_hidden": model.hidden_width,
            "feature_variant": model.feature_variant,
            "feature_normalization": model.feature_normalization,
            "material_feature_normalization": model.material_feature_normalization,
            "feature_log_scale": model.feature_log_scale,
            "feature_log_clip": list(model.feature_log_clip),
            "feature_eps": 1e-8,
            "include_material_scale_features": model.include_material_scale_features,
            "include_material_ratios": model.include_material_ratios,
        }
    )
    params["num_features"] = nn_feature_count(model.N, params)


def validate_checkpoints(models, num_tests, tagged_model_path):
    missing = []
    for model in models:
        for model_idx in range(num_tests):
            path = ONE_D_ROOT / tagged_model_path(
                model.N, model_idx, filter_type=1, model_tag=model.model_tag
            )
            if not path.exists():
                missing.append(path.relative_to(ROOT))
    if missing:
        preview = "\n  - ".join(str(path) for path in missing[:12])
        more = "" if len(missing) <= 12 else f"\n  ... and {len(missing) - 12} more"
        raise FileNotFoundError(f"Missing checkpoints:\n  - {preview}{more}")


def run_case(params, testing, model, model_idx, ic_idx, problem, T):
    if ic_idx == 6:
        reset_grid(params, xl=0.0, xr=8.0, num_x=512, T=T)
    else:
        reset_grid(params, xl=-1.0, xr=1.0, num_x=128, T=T)
    params["IC_idx"] = ic_idx
    params["T"] = T
    value = testing(params, model_idx)
    return {
        "table": "1d_best_sweeps",
        "ansatz": "nn",
        "model_tag": model.model_tag,
        "ic_idx": ic_idx,
        "problem": problem,
        "final_time": T,
        "N": model.N,
        "model_idx": model_idx,
        "flux_error_reduction": as_float(value),
    }


def write_csv(path, rows):
    fieldnames = [
        "table",
        "ansatz",
        "model_tag",
        "ic_idx",
        "problem",
        "final_time",
        "N",
        "model_idx",
        "flux_error_reduction",
    ]
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.num_tests < 1:
        raise SystemExit("--num-tests must be at least 1")

    os.chdir(ONE_D_ROOT)
    sys.path.insert(0, str(ONE_D_ROOT / "src"))

    from funcs_common import nn_feature_count
    from params_common import params, resolve_device, tagged_model_path
    from test_model import testing

    models = [BEST_MODELS[N] for N in sorted(BEST_MODELS)]
    validate_checkpoints(models, args.num_tests, tagged_model_path)

    scratch = ONE_D_ROOT / "results"
    output_dir = ONE_D_ROOT / args.output_dir
    if scratch.exists():
        shutil.rmtree(scratch)
    if output_dir.exists():
        if args.keep_existing:
            raise SystemExit(f"Output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)

    params.update(
        {
            "batch_size": 1,
            "tt_flag": 1,
            "device": resolve_device(args.device),
            "ablation_idx": 0,
            "filter_type": 1,
            "N_exact": 127,
            "show_plot": 0,
            "print_results": bool(args.print_results),
        }
    )

    rows = []
    cases = list(STANDARD_CASES)
    if not args.skip_reeds:
        cases.append(REEDS_CASE)

    print("1D best-sweep NN simulations")
    print(f"{'problem':<30} {'T':>5} {'N':>4} {'model_tag':<22} {'flux reduction':>15}")
    print("-" * 83)
    for ic_idx, problem, times in cases:
        for T in times:
            for model in models:
                configure_model(params, model, nn_feature_count)
                for model_idx in range(args.num_tests):
                    row = run_case(
                        params, testing, model, model_idx, ic_idx, problem, T
                    )
                    rows.append(row)
                    print(
                        f"{problem:<30} {T:>5.1f} {model.N:>4} "
                        f"{model.model_tag:<22} {row['flux_error_reduction']:>15.6f}"
                    )

    csv_path = ONE_D_ROOT / args.csv
    write_csv(csv_path, rows)
    if scratch.exists():
        shutil.move(str(scratch), str(output_dir))

    print(f"\nSaved CSV: {csv_path.relative_to(ROOT)}")
    if output_dir.exists():
        print(f"Saved plots: {output_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
