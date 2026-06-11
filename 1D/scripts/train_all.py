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
from params_common import params, resolve_device
from train_model import training


num_trains = 10
params["num_IC"] = 4
params["batch_size"] = (
    64  ## make batch size a multiple of the number of Initial Conditions
)
params["num_epochs"] = 200
params["learning_rate"] = 1e-1
params["momentum_factor"] = 0.9
params["sigs_max"] = 1
params["obj_idx"] = 0
GD_idx = 1

if GD_idx == 0:
    params["GD_optimizer"] = "SGD"
elif GD_idx == 1:
    params["GD_optimizer"] = "Adam"

params["tt_flag"] = 0
params["IC_idx"] = 0
params["device"] = resolve_device(params.get("device"))
params["ablation_idx"] = 0
filter_type = params["filter_type"]

for j in range(num_trains):
    for n in [3, 7, 9]:
        N = n
        params["N"] = N
        if filter_type in (1, 2):
            params["num_features"] = 2 * N + 4
        NN_model = training(params)
        if filter_type in (1, 2):
            torch.save(NN_model, f"trained_models/model_N{N}_{j}.pth")
        if filter_type == 3:
            torch.save(NN_model, f"trained_models_const/model_N{N}_{j}.pth")
