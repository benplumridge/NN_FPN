# IC INDEX
# 0 - Gaussian
# 1 - Vanishing Cross Section
# 2 - Discontinuous Cross Section
# 3 - Heavi-side
# 4 - Bump
# 5 - Discontinuous Source
# 6 - Reeds
# IC_idx = 0

import sys
import os
import torch

# Add src to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from params_common import model_tag_from_params, params, resolve_device
from funcs_common import filter_func, nn_feature_count
from test_model import testing

params["batch_size"] = 1
params["tt_flag"] = 1
params["device"] = resolve_device(params.get("device"))
params["ablation_idx"] = 0
params["print_results"] = False
# params['IC_idx'] = IC_idx

IC_NAMES = {
    0: "Gaussian",
    1: "Vanishing_cross_section",
    2: "Discontinuous_cross_section",
    6: "Reeds",
}


def _as_float(value):
    return float(torch.as_tensor(value).detach().cpu())


def _print_summary_header():
    ansatz = "constant" if params["filter_type"] == 3 else "nn"
    model_tag = model_tag_from_params(params)
    print(f"1D plot sweep ({ansatz}, model_tag={model_tag})")
    print(f"{'problem':<30} {'T':>5} {'N':>4} {'flux reduction':>15}")
    print("-" * 57)


def _print_summary_row(problem, T_val, N, reduction):
    print(f"{problem:<30} {float(T_val):>5.1f} {N:>4} {_as_float(reduction):>15.4f}")


N_exact = 127
filter_order = 4
num_x = params["num_x"]
dx = params["dx"]
Ns = [int(N) for N in params.get("Ns", [3, 7, 9])]

_print_summary_header()

for IC_idx in [0, 1, 2]:
    params["IC_idx"] = IC_idx
    for N in Ns:
        filt_input = torch.arange(0, N + 1, 1) / (N + 1)
        filter = -torch.log(filter_func(filt_input, filter_order))
        params["num_features"] = nn_feature_count(N, params)
        params.setdefault("num_hidden", N + 2)
        params["filter"] = filter
        params["N"] = N
        for T in [0.5, 1]:
            params["T"] = T
            dt = dx / 2
            num_t = int((T + dt) // dt)
            params["dt"] = dt
            params["num_t"] = num_t
            error_red = testing(params)
            _print_summary_row(IC_NAMES[IC_idx], T, N, error_red)

params["IC_idx"] = 6
for N in Ns:
    filt_input = torch.arange(0, N + 1, 1) / (N + 1)
    filter = -torch.log(filter_func(filt_input, filter_order))
    params["num_features"] = nn_feature_count(N, params)
    params.setdefault("num_hidden", N + 2)
    params["filter"] = filter
    params["N"] = N
    for T in [5, 10]:
        params["T"] = T
        num_x = 512
        xr = 8
        xl = 0
        L = xr - xl
        dx = L / num_x
        params["dx"] = dx
        params["num_x"] = num_x
        dt = dx / 2
        num_t = int((T + dt) // dt)
        params["dt"] = dt
        params["num_t"] = num_t

        error_red = testing(params)
        _print_summary_row(IC_NAMES[params["IC_idx"]], T, N, error_red)
