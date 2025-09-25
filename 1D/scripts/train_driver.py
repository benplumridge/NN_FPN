import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from numpy import asarray
from numpy import savetxt

# Add src to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from params_common import params
from train_model import training

import argparse


def filter_func(z, p):
    return torch.exp(-(z**p))


parser = argparse.ArgumentParser(description="Train the model for N")
parser.add_argument("N", type=int, help="N")
parser.add_argument("--const_net", action="store_true", help="train constant network")
args = parser.parse_args()

N = int(args.N)
params["N"] = N
params["const_net"] = args.const_net
if N == 3:
    params["learning_rate"] = 1e-1
    params["weight_decay"] = 1e-5
elif N == 7 or N == 9:
    params["learning_rate"] = 1e-1
    params["weight_decay"] = 1e-5


params["num_IC"] = 4
params["batch_size"] = (
    64  ## make batch size a multiple of the number of Initial Conditions
)
params["num_epochs"] = 200
params["momentum_factor"] = 0.2
params["sigs_max"] = 1
# params["GD_optimizer"] = "SGD"
params["GD_optimizer"] = "Adam"
params["tt_flag"] = 0
params["IC_idx"] = 0
params["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params["num_features"] = 2 * N + 4
params["num_hidden"] = N + 2
print("N = ", N)
filt_input = torch.arange(0, N + 1, 1) / (N + 1)

params["filter"] = -torch.log(
    filter_func(filt_input.to(params["device"]), params["filter_order"])
)

NN_model = training(params)
N = params["N"]
model_savename = (
    f"trained_models/model_N{N}_const.pth"
    if params["const_net"]
    else f"trained_models/model_N{N}.pth"
)
torch.save(NN_model, model_savename)
