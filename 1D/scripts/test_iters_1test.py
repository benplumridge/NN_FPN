import sys
import os
import torch

# Add src to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from params_common import params
from test_model import testing


# IC INDEX
# 0 - Gaussian
# 1 - Vanishing Cross Section
# 2 - Discontinuous Cross Section
# 3 - Heavi-side
# 4 - Bump
# 5 - Discontinuous Source
# 6 - Reeds
IC_idx = 2

num_tests = 1

params["batch_size"] = 1
params["tt_flag"] = 1
params["device"] = "cpu"
params["ablation_idx"] = 0 
params["IC_idx"] = IC_idx



error_reduction = []

for j in range(num_tests):
    error_red = testing(params, j)
    error_reduction.append(error_red)

error_reduction = torch.tensor(error_reduction)

print(
    f"Error_reduction = {error_reduction.mean().item():.4f} \pm {error_reduction.std().item():.4f}"
    )

