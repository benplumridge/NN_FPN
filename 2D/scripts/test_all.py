import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from params_common import params
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
params["device"] = "cpu"

N_exact = 37
filter_order = 4

params["IC_idx"] = 0
num_x = 100
num_y = 100
dx = 0.02
dy = 0.02

xl = -1
xr = 1
yl = -1
yr = 1
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

Ns = [3, 5, 7, 9]
T_gauss = [0.75]
T_latt  = [1.6, 3.2]
Ts = [T_gauss, T_latt]
# Dictionary to store results
# results[(IC_idx, T)][N] = (error, reduction)
results = {}

params["IC_idx"] = 0

for T in T_gauss:

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


num_x = 280
num_y = 280
dx = 0.025
dy = 0.025

params["IC_idx"] = 6
params["num_x"] = num_x
params["num_y"] = num_y
params["dx"] = dx
params["dy"] = dy

xl = 0
xr = 7
yl = 0
yr = 7

x_edges = torch.linspace(xl, xr, num_x + 1, dtype=torch.float32)
y_edges = torch.linspace(yl, yr, num_y + 1, dtype=torch.float32)

x = torch.linspace(xl + dx / 2, xr - dx / 2, num_x, dtype=torch.float32)
y = torch.linspace(yl + dy / 2, yr - dy / 2, num_y, dtype=torch.float32)

params["x"] = x
params["y"] = y
params["x_edges"] = x_edges
params["y_edges"] = y_edges

for T in T_latt:

    results[(params["IC_idx"], T)] = {}

    for N in Ns:

        num_basis = (N + 1) * (N + 2) // 2
        num_features = 2 * (N + 1) + 2
        num_hidden = num_features // 2

        params["num_basis"] = num_basis
        params["num_features"] = num_features
        params["num_hidden"] = num_hidden
        params["N"] = N

        dt = dx / 2
        params["dt"] = dt
        params["T"] = T
        params["num_t"] = int((T + dt) // dt)

        FPN_error, FPN_error_reduction = testing(params)

        results[(params["IC_idx"], T)][N] = (
            FPN_error,
            FPN_error_reduction,
        )


with open("error_reduction_table.txt", "w") as f:

    for (IC_idx, T), data in results.items():

        f.write("=" * 90 + "\n")
        f.write(f"Test Problem IC_idx = {IC_idx}, Final Time T = {T}\n")
        f.write("=" * 90 + "\n\n")

        # Header
        header = ""

        for N in Ns:
            header += (
                f"{f'N = {N} Error':>20}"
                f"{f'N = {N} Reduction':>25}"
            )

        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        # Row
        row = ""

        for N in Ns:

            error, reduction = data[N]

            row += (
                f"{error:>20.4f}"
                f"{reduction:>25.4f}"
            )

        f.write(row + "\n\n\n")