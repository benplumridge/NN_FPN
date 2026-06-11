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
from params_common import model_tag_from_params, params, resolve_device, tagged_model_path
from train_model import training


params["num_IC"] = 4
params["batch_size"] = (
    64  ## make batch size a multiple of the number of Initial Conditions
)
params["num_epochs"] = 200
params["learning_rate"] = 1e-1  # lr= 10  for N = 3,  l3 = 100 for N = 7,9
params["momentum_factor"] = 0.9
params["sigs_max"] = 1
params["obj_idx"] = 0
# params["GD_optimizer"] = "SGD"
params["GD_optimizer"] = "Adam"
params["tt_flag"] = 0
params["IC_idx"] = 0
params["device"] = resolve_device(params.get("device"))
params["ablation_idx"] = 0
filter_type = params["filter_type"]
params["model_tag"] = model_tag_from_params(params)

NN_model = training(params)


N = params["N"]
if filter_type in (1, 2):
    torch.save(NN_model, tagged_model_path(N, 0, filter_type, params["model_tag"]))
elif filter_type == 3:
    torch.save(NN_model, tagged_model_path(N, 0, filter_type, params["model_tag"]))
