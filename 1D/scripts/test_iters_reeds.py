import csv
import os
import sys

import torch

# Add src to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from params_common import model_tag_from_params, params, resolve_device, tagged_model_path
from test_model import testing


IC_idx = 6
params["IC_idx"] = IC_idx

num_tests = int(params.get("num_tests", 5))

params["batch_size"] = 1
params["tt_flag"] = 1
params["device"] = resolve_device(params.get("device"))
params["ablation_idx"] = 0
params["print_results"] = False
params["num_x"] = int(params.get("reeds_num_x", 512))


Ns = [3, 7, 9]

filter_type = params["filter_type"]
if filter_type == 1:
    file_name = "error_reduction_table_reeds.txt"
elif filter_type == 3:
    file_name = "error_reduction_table_reeds_const.txt"
else:
    raise ValueError(f"Unsupported filter_type: {filter_type}")

csv_name = os.path.splitext(file_name)[0] + ".csv"
ansatz = "constant" if filter_type == 3 else "nn"
model_tag = model_tag_from_params(params)


def _checkpoint_path(N, model_idx):
    return tagged_model_path(N, model_idx, filter_type, model_tag)


def _validate_checkpoints():
    missing = []
    for N in Ns:
        for model_idx in range(num_tests):
            path = _checkpoint_path(N, model_idx)
            if not os.path.exists(path):
                missing.append(path)
    if missing:
        preview = "\n  - ".join(missing[:12])
        more = "" if len(missing) <= 12 else f"\n  ... and {len(missing) - 12} more"
        raise FileNotFoundError(
            f"Missing checkpoints for num_tests={num_tests}:\n  - {preview}{more}"
        )


def _error_stats(values):
    tensor = torch.stack([torch.as_tensor(value).detach().cpu() for value in values])
    return tensor.mean().item(), tensor.std(unbiased=False).item()


def _write_csv(rows):
    fieldnames = [
        "table",
        "ansatz",
        "model_tag",
        "ic_idx",
        "problem",
        "final_time",
        "N",
        "mean_flux_error_reduction",
        "std_flux_error_reduction",
        "num_tests",
        "paper_table_cell",
    ]
    with open(csv_name, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_summary_header():
    print(f"1D Reeds results ({ansatz}, model_tag={model_tag}, num_tests={num_tests})")
    print(f"{'problem':<30} {'T':>5} {'N':>4} {'flux reduction':>21}")
    print("-" * 63)


def _print_summary_row(problem, T_val, N, mean_val, std_val):
    cell = f"{mean_val:.4f} +/- {std_val:.4f}"
    print(f"{problem:<30} {T_val:>5.1f} {N:>4} {cell:>21}")


_validate_checkpoints()
_print_summary_header()


csv_rows = []
with open(file_name, "w") as f:
    f.write(f"IC_idx = {IC_idx}\n")
    f.write("t\t" + "\t".join([f"N={N}" for N in Ns]) + "\n")

    times = [5.0, 10.0]

    for T_val in times:
        params["T"] = T_val
        params["dx"] = 8.0 / params["num_x"]
        params["dt"] = params["dx"] / 2
        params["num_t"] = int((T_val + params["dt"]) // params["dt"])
        row = [f"{T_val:.1f}"]

        for N in Ns:
            params["N"] = N
            error_reduction = []

            for j in range(num_tests):
                error_red = testing(params, j)
                error_reduction.append(error_red)

            mean_val, std_val = _error_stats(error_reduction)
            _print_summary_row("Reeds", T_val, N, mean_val, std_val)
            cell = f"{mean_val:.4f} ± {std_val:.4f}"
            csv_rows.append(
                {
                    "table": "1d_reeds",
                    "ansatz": ansatz,
                    "model_tag": model_tag,
                    "ic_idx": IC_idx,
                    "problem": "Reeds",
                    "final_time": T_val,
                    "N": N,
                    "mean_flux_error_reduction": mean_val,
                    "std_flux_error_reduction": std_val,
                    "num_tests": num_tests,
                    "paper_table_cell": cell,
                }
            )

            row.append(cell)

        f.write("\t".join(row) + "\n")

    f.write("\n")

_write_csv(csv_rows)
print(f"Saved tables: {file_name}, {csv_name}")
