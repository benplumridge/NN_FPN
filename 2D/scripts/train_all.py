import sys
import os
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from params_common import params
from train_model import training


params["num_IC"] = 5
params["batch_size"] = (
    5  ## make batch size a multiple of the number of Initial Conditions
)
params["num_epochs"] = 200
params["learning_rate"] = 1e-2
params["momentum_factor"] = 0.9
params["sigs_max"] = 1
params["tt_flag"] = 0
params["IC_idx"] = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
params["device"] = device

# Optimizer flag
# 0  - SGD
# 1  - Adam
GD_opt_flag = 1

if GD_opt_flag == 0:
    params["GD_optimizer"] = "SGD"
elif GD_opt_flag == 1:
    params["GD_optimizer"] = "Adam"



for N in [3,5,7,9]:
    num_features = 2 * (N + 1) + 2
    num_basis = (N + 1) * (N + 2) // 2
    params['num_features'] = num_features
    params['num_basis'] = num_basis
    params["N"] = N
    params["filter"] = filter
    NN_model = training(params)
    filename = f"trained_models/model_N{N}.pth"
    torch.save(NN_model, filename)

