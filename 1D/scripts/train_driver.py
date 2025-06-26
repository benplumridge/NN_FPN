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
from params_common import params, filter_func, filter_order, N_exact
from train_model import training


params["num_IC"] = 4
params["batch_size"] = (
    32  ## make batch size a multiple of the number of Initial Conditions
)
params["num_epochs"] = 50
params["num_hidden"] = 200
params["learning_rate"] = 0.01
params["momentum_factor"] = 0.9
params["sigs_max"] = 1
# params["GD_optimizer"] = "SGD"
params["GD_optimizer"] = "Adam"
params["tt_flag"] = 0
params["IC_idx"] = 0
params["device"] = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

for N in [3, 7, 9]:  # Example values for N
    params["N"] = N
    params["num_features"] = 2 * N + 4
    filter = torch.zeros(N_exact + 1)
    filt_input = torch.zeros(N + 1)
    filt_input[0 : N + 1] = torch.arange(0, N + 1, 1) / (N + 1)
    filter[0 : N + 1] = -torch.log(filter_func(filt_input, filter_order))
    params["filter"] = filter
    print("Train model", "N =", N)
    NN_model = training(params)

    torch.save(NN_model, f"trained_models/model_N{N}.pth")
