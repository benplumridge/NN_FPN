# IC INDEX
# 0 - Vanishing Cross Section
# 1 - Discontinuous Cross Section
# 2 - Gaussian
# 3 - Heavi-side
# 4 - Bump
# 5 - Discontinuous Source
# 6 - Reeds
IC_idx = 1

import sys
import os
import torch

# Add src to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from params_common import params, params, filter_func, filter_order, N_exact
from test_model import testing

params["batch_size"] = 1
params["tt_flag"] = 1
params["device"] = "cpu"

for IC_idx in [2, 0, 1, 6]:
    for N in [3, 7, 9]:
        for t in [0.5, 1]:
            params["N"] = N
            params["num_features"] = 2 * N + 4
            filter = torch.zeros(N_exact + 1)
            filt_input = torch.zeros(N + 1)
            filt_input[0 : N + 1] = torch.arange(0, N + 1, 1) / (N + 1)
            filter[0 : N + 1] = -torch.log(filter_func(filt_input, filter_order))
            params["filter"] = filter

            params["T"] = t
            params["num_t"] = int((params["T"] + params["dt"]) // params["dt"])
            params["IC_idx"] = IC_idx
            print("test case", "N =", N, "IC_idx =", IC_idx, "T =", t)
            testing(params)
            print("-----")
