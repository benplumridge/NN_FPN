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

import argparse

parser = argparse.ArgumentParser(description="Train the model for N")
parser.add_argument("N", type=int, help="N")
parser.add_argument("ablation_idx", type=int, help="ablation study index")
parser.add_argument("model_idx", type=int, help="Model index")
parser.add_argument("--const_net", action="store_true", help="train constant network")
args = parser.parse_args()


params["N"] = int(args.N)
params["const_net"] = args.const_net
params["ablation_idx"] = int(args.ablation_idx)
params["model_idx"] = int(args.model_idx)
params["batch_size"] = 1
params["tt_flag"] = 1
params["device"] = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
# params['IC_idx'] = IC_idx

N_exact = params["N_exact"]
num_x = params["num_x"]
dx = params["dx"]

for IC_idx in [0, 1, 2]:
    params["IC_idx"] = IC_idx
    for T in [0.5, 1]:
        params["T"] = T
        dt = dx / 2
        num_t = int((T + dt) // dt)
        params["dt"] = dt
        params["num_t"] = num_t
        testing(params)

params["IC_idx"] = 6
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
