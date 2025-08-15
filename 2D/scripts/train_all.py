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



def filter_func(z, p):
    return torch.exp(-(z**p))


def filter_coefficients(filter_order, N, num_basis):
    filter = torch.zeros(N + 1)
    filter[1 : N + 1] = -torch.log(
        filter_func(torch.arange(1, N + 1) / (N + 1), filter_order)
    )

    filter_expand = torch.zeros(num_basis)
    idx = 0
    for l in range(1, N + 2):
        filter_expand[idx : idx + l] = filter[l - 1]
        idx += l
    return filter_expand

for N in [3,5,7,9]:
    num_features = 2 * (N + 1) + 2
    num_basis = (N + 1) * (N + 2) // 2
    filter = filter_coefficients(params["filter_order"], N, num_basis)
    params['num_features'] = num_features
    params['num_basis'] = num_basis
    params["N"] = N
    params["filter"] = filter
    NN_model = training(params)
    filename = f"trained_models/model_N{N}.pth"
    torch.save(NN_model, filename)

