
import torch

# IC INDEX
# 0 - Gaussian
# 1 - Vanishing Cross Section
# 2 - Discontinuous Cross Section
# 3 - Heavi-side
# 4 - Bump
# 5 - Discontinuous Source
# 6 - Reeds
IC_idx = 0

import sys
import os
import numpy as np


# Add src to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from params_common import params
from test_model import testing

params["batch_size"] = 1
params["tt_flag"] = 1
params["device"] = "cpu"
params["IC_idx"] = IC_idx
params["const_net"] = False
params["model_idx"] = 0
params["ablation_idx"] = 0


error_FPN = []

num_its = 5

if IC_idx == 6:
    xl =  params["xl"]
    xr =  params["xr"]
    L  =  params["L"]
    T  =  10
    num_x = 128*4
    dx = L / num_x
    dt = dx / 2
    num_t = int((T + dt) // dt)
    x_edges = torch.linspace(xl, xr, num_x + 1)
    x = torch.linspace(xl + dx / 2, xr - dx / 2, num_x)
    params.update({
    "dx": dx,
    "dt": dt,
    "num_t": num_t,
    "x_edges": x_edges,
    "x": x,
    "num_x": num_x,
    "T"    : T
})


with open("testing_times.txt", "w") as run_times:
    run_times.write("# exact_time  PN_time  FPN_time\n")
    for j in range(num_its):
        err_PN, err_FPN_out = testing(params, j, run_times)
        error_FPN.append(err_FPN_out)

mean_val = np.mean(error_FPN)
std_val  = np.std(error_FPN)

print(f"mean FPN error = ", mean_val, "std dev =", std_val)

data = np.loadtxt("testing_times.txt", comments="#")

means = np.mean(data, axis=0)
stds  = np.std(data, axis=0, ddof=1)

print(f"Mean times    (Exact, PN, FPN): {means}")
print(f"Std dev times (Exact, PN, FPN): {stds}")