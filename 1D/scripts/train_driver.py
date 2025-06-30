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


params["num_IC"] = 4
params["batch_size"] = (
    4  ## make batch size a multiple of the number of Initial Conditions
)
params["num_epochs"] = 200
params["num_hidden"] = 100
params["learning_rate"] = 0.01
params["momentum_factor"] = 0.9
params["sigs_max"] = 1
# params["GD_optimizer"] = "SGD"
params["GD_optimizer"] = "Adam"
params["tt_flag"] = 0
params["IC_idx"] = 0
params["device"] = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

NN_model = training(params)
N = params["N"]
torch.save(NN_model, f"trained_models/model_N{N}.pth")
