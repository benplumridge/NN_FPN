import sys
import os
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

if filter_type == 0:
    file_name = "error_reduction_hohl_NN"
elif filter_type in {1, 2}:
    file_name = "error_reduction_hohl_const"

with open(file_name, "w") as f:
    for (IC_idx, T), data in results.items():
        f.write("=" * 90 + "\n")
        f.write(f"Test Problem IC_idx = {IC_idx}, Final Time T = {T}\n")
        f.write("=" * 90 + "\n\n")

        # Header
        header = ""

        for N in Ns:
            header += f"{f'N = {N} Error':>20}{f'N = {N} Reduction':>25}"

        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        # Row
        row = ""

        for N in Ns:
            error, reduction = data[N]

            row += f"{error:>20.4f}{reduction:>25.4f}"

        f.write(row + "\n\n\n")
