#!/usr/bin/env python3
"""Train and evaluate 10-replica NN/constant runs from the best 1D sweep settings."""

import argparse
import csv
import io
import json
import os
import shutil
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
ONE_D = ROOT / "1D"
OUT_DIR = ONE_D / "best_sweep_10x"

BEST_CONFIGS = {
    3: ROOT / "configs" / "best_sweeps" / "1d_nn_N3.yaml",
    7: ROOT / "configs" / "best_sweeps" / "1d_nn_N7.yaml",
    9: ROOT / "configs" / "best_sweeps" / "1d_nn_N9.yaml",
}

STANDARD_CSVS = {
    "nn": "error_reduction_table.csv",
    "constant": "error_reduction_table_const.csv",
}
REEDS_CSVS = {
    "nn": "error_reduction_table_reeds.csv",
    "constant": "error_reduction_table_reeds_const.csv",
}


def load_yaml(path):
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def write_yaml(path, data):
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream, sort_keys=False)


def derived_config(N, source_path, num_replicates, disable_wandb):
    cfg = load_yaml(source_path)
    tag = f"best_sweep_n{N}_10x"

    workflow = cfg.setdefault("workflow", {})
    workflow.update(
        {
            "target": "1d",
            "ansatz": "both",
            "phase": "all",
            "train": True,
            "simulate": True,
            "overwrite_results": True,
        }
    )
    if disable_wandb:
        workflow.setdefault("wandb", {})["enabled"] = False
        workflow["wandb"]["mode"] = "disabled"

    dim = cfg.setdefault("1d", {})
    train = dim.setdefault("train", {})
    train["Ns"] = [N]
    train["num_replicates"] = num_replicates
    train_params = train.setdefault("params", {})
    train_params["model_tag"] = tag

    simulate = dim.setdefault("simulate", {})
    simulate["result_dir_template"] = f"results_{{label}}_best_N{N}_10x"
    sim_params = simulate.setdefault("params", {})
    sim_params["model_tag"] = tag
    sim_params["Ns"] = [N]
    sim_params["num_tests"] = num_replicates

    return cfg


def run_command(command, log_path):
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)


def copy_outputs(N):
    copied = []
    for suffix, filenames in (("standard", STANDARD_CSVS), ("reeds", REEDS_CSVS)):
        for ansatz, filename in filenames.items():
            source = ONE_D / filename
            if not source.exists():
                raise FileNotFoundError(f"Expected output CSV was not written: {source}")
            dest = OUT_DIR / f"N{N}_{ansatz}_{suffix}.csv"
            shutil.copy2(source, dest)
            copied.append(dest)
    return copied


def read_rows():
    rows = []
    for path in sorted(OUT_DIR.glob("N*_*.csv")):
        with path.open("r", newline="") as csv_file:
            for row in csv.DictReader(csv_file):
                row = dict(row)
                row["source_csv"] = str(path.relative_to(ROOT))
                rows.append(row)
    return rows


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


def build_table_data(rows):
    data = {}
    table_order = []
    for row in rows:
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
            "num_tests": int(row["num_tests"]),
            "model_tag": row["model_tag"],
        }
    return table_order, data


def print_latex_table(table_key, times, Ns, data):
    _, problem = table_key
    print()
    print(_problem_label(problem))
    print(r"\begin{tabular}{c c c c c c c c}")
    print(
        r"   & \multicolumn{3}{c}{Neural Network ($r_N^{\text{nn}}$)} "
        r"& \phantom{abc} & \multicolumn{3}{c}{Constant ($r_N^{\text{const}}$)} \\")
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
            + r" \\")
    print(r"        \bottomrule")
    print(r"\end{tabular}")


def print_terminal_table(table_key, times, Ns, data):
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
        print(
            f"{time_value:>5.1f} "
            + " ".join(f"{cell:>{cell_width}}" for cell in nn_cells)
            + " | "
            + " ".join(f"{cell:>{cell_width}}" for cell in const_cells)
        )


def print_combined_tables():
    rows = read_rows()
    table_order, data = build_table_data(rows)
    table_shapes = []
    print("Paper-style summary tables")
    for table_key in table_order:
        times = sorted({key[2] for key in data if key[:2] == table_key})
        Ns = [3, 7, 9]
        table_shapes.append((table_key, times, Ns))
        print_latex_table(table_key, times, Ns, data)

    print("\nTerminal-readable summary tables")
    print("* = lower mean error reduction for that T,N comparison")
    for table_key, times, Ns in table_shapes:
        print_terminal_table(table_key, times, Ns, data)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train NN and constant best-sweep 1D settings 10 times and print paper tables."
    )
    parser.add_argument("--num-replicates", type=int, default=10)
    parser.add_argument("--only-simulate", action="store_true", help="Skip training and only rerun simulations.")
    parser.add_argument("--only-tables", action="store_true", help="Only reprint tables from copied CSVs.")
    parser.add_argument("--enable-wandb", action="store_true", help="Keep W&B enabled in generated configs.")
    return parser.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.only_tables:
        print_combined_tables()
        return

    generated = []
    for N, source in BEST_CONFIGS.items():
        cfg = derived_config(N, source, args.num_replicates, disable_wandb=not args.enable_wandb)
        path = OUT_DIR / f"best_sweep_N{N}_10x.yaml"
        write_yaml(path, cfg)
        generated.append((N, path))

    for N, cfg_path in generated:
        log_path = OUT_DIR / f"N{N}_workflow.log"
        phase = "simulate" if args.only_simulate else "all"
        command = [
            sys.executable,
            str(ROOT / "main.py"),
            "-c",
            str(cfg_path),
            "--target",
            "1d",
            "--ansatz",
            "both",
            "--phase",
            phase,
        ]
        print(f"==> N={N}: running {phase}; log={log_path.relative_to(ROOT)}", flush=True)
        try:
            run_command(command, log_path)
        except subprocess.CalledProcessError as exc:
            print(f"N={N} failed. See {log_path.relative_to(ROOT)}", file=sys.stderr)
            raise SystemExit(exc.returncode) from exc
        copied = copy_outputs(N)
        print("    copied " + ", ".join(str(path.relative_to(ROOT)) for path in copied), flush=True)

    table_buffer = io.StringIO()
    with redirect_stdout(table_buffer):
        print_combined_tables()
    table_text = table_buffer.getvalue()
    table_path = OUT_DIR / "paper_tables.txt"
    table_path.write_text(table_text, encoding="utf-8")
    print(table_text)
    print(f"Saved generated configs/logs/CSVs/tables under {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
