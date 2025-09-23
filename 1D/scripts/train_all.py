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


def filter_func(z, p):
    return torch.exp(-(z**p))


params["num_IC"] = 4
params["batch_size"] = (
    20  ## make batch size a multiple of the number of Initial Conditions
)
params["num_epochs"] = 100
params["learning_rate"] = 1e-2  # N = 3 lr = 1e-2,  N = 7 and 9 lr = 1e-3
# params["learning_rate"] = 1e2
params["momentum_factor"] = 0.9
params["sigs_max"] = 1
GD_idx = 1

if GD_idx == 0:
    params["GD_optimizer"] = "SGD"
elif GD_idx == 1:
    params["GD_optimizer"] = "Adam"

params["tt_flag"] = 0
params["IC_idx"] = 0
params["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# for n in [7,9]:
for n in [3]:  # ,7,9]:
    N = n
    params["N"] = N
    if N == 3:
        params["learning_rate"] = 1e-2
    elif N == 7 or N == 9:
        params["learning_rate"] = 1e-3
    params["num_features"] = 2 * N + 4
    params["num_hidden"] = N + 2
    filt_input = torch.arange(0, N + 1, 1) / (N + 1)
    params["filter"] = -torch.log(filter_func(filt_input, params["filter_order"]))
    NN_model = training(params)
    torch.save(NN_model, f"trained_models/model_N{N}.pth")
