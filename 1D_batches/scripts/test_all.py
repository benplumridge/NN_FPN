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
from params_common import params
from test_model import testing

params["batch_size"] = 1
params["tt_flag"] = 1
params["device"] = "cpu"

N_exact = 127
num_x = params["num_x"]
dx = params["dx"]

for IC_idx in [0, 1, 2]:
    params["IC_idx"] = IC_idx
    for N in [3, 7, 9]:
        num_features = 2 * N + 4
        num_hidden = N + 2
        params["num_features"] = num_features
        params["num_hidden"] = num_hidden
        params["N"] = N
        for T in [0.5, 1]:
            params["T"] = T
            dt = dx / 2
            num_t = int((T + dt) // dt)
            params["dt"] = dt
            params["num_t"] = num_t
            testing(params)

params["IC_idx"] = 6
#for N in [3]:
for N in [3, 7, 9]:
    num_features = 2 * N + 4
    num_hidden = N + 2
    params["num_features"] = num_features
    params["num_hidden"] = num_hidden
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

        testing(params)
