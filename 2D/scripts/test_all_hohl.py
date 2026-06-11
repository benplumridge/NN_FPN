import csv
import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from params_common import params, resolve_device
from test_model import testing


### IC INDEX
# 0 - Gaussian
# 1 - Step
# 2 - Discontinuous source
# 3 - Bump
# 4 - Hat
# 5 - Holhraum
# 6 - lattice
# params['IC_idx'] = 0

params["batch_size"] = 1
params["tt_flag"] = 1
params["device"] = resolve_device(params.get("device"))

N_exact = 37
filter_order = 4

params["IC_idx"] = 0
num_x = 130
num_y = 130
dx = 0.02
dy = 0.02

xl = 0
xr = 1.3
yl = 0
yr = 1.3
x_edges = torch.linspace(xl, xr, num_x + 1, dtype=torch.float32)
y_edges = torch.linspace(yl, yr, num_y + 1, dtype=torch.float32)
x = torch.linspace(xl + dx / 2, xr - dx / 2, num_x, dtype=torch.float32)
y = torch.linspace(yl + dy / 2, yr - dy / 2, num_y, dtype=torch.float32)

params["num_x"] = num_x
params["num_y"] = num_y
params["dx"] = dx
params["dy"] = dy
params["x"] = x
params["y"] = y
params["x_edges"] = y_edges
params["y_edges"] = y_edges

params["dt"] = params["dx"] / 2

params["IC_idx"] = 5


Ns = [3, 5, 7, 9]
Ts = [1.5, 3.0]
IC_NAMES = {5: "hohlraum"}

results = {}

for T in Ts:
    results[(params["IC_idx"], T)] = {}

    for N in Ns:
        num_features = 2 * (N + 1) + 2
        num_hidden = num_features // 2
        num_basis = (N + 1) * (N + 2) // 2

        params["num_basis"] = num_basis
        params["num_features"] = num_features
        params["num_hidden"] = num_hidden
        params["N"] = N
        params["T"] = T

        FPN_error, FPN_error_reduction = testing(params)

        results[(params["IC_idx"], T)][N] = (
            FPN_error,
            FPN_error_reduction,
        )

filter_type = params["filter_type"]
if filter_type == 0:
    file_name = "error_reduction_hohl_NN"
elif filter_type in {1, 2}:
    file_name = "error_reduction_hohl_const"
else:
    raise ValueError(f"Unsupported filter_type: {filter_type}")

csv_name = file_name + ".csv"
ansatz = "nn" if filter_type == 0 else "constant"


def _as_float(value):
    return float(torch.as_tensor(value).detach().cpu())


def _write_csv(rows):
    fieldnames = [
        "table",
        "ansatz",
        "ic_idx",
        "problem",
        "final_time",
        "N",
        "flux_error",
        "flux_error_reduction",
    ]
    with open(csv_name, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


csv_rows = []
with open(file_name, "w") as f:
    for (IC_idx, T), data in results.items():
        f.write("=" * 90 + "\n")
        f.write(f"Test Problem IC_idx = {IC_idx}, Final Time T = {T}\n")
        f.write("=" * 90 + "\n\n")

        header = ""
        for N in Ns:
            header += f"{f'N = {N} Error':>20}{f'N = {N} Reduction':>25}"

        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        row = ""
        for N in Ns:
            error, reduction = data[N]
            error_value = _as_float(error)
            reduction_value = _as_float(reduction)
            csv_rows.append(
                {
                    "table": "2d_hohlraum",
                    "ansatz": ansatz,
                    "ic_idx": IC_idx,
                    "problem": IC_NAMES.get(IC_idx, str(IC_idx)),
                    "final_time": T,
                    "N": N,
                    "flux_error": error_value,
                    "flux_error_reduction": reduction_value,
                }
            )
            row += f"{error_value:>20.4f}{reduction_value:>25.4f}"

        f.write(row + "\n\n\n")

_write_csv(csv_rows)
print(f"Tables saved to {file_name} and {csv_name}")
