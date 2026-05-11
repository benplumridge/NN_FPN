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
IC_idx = 6
params["IC_idx"] = IC_idx
num_tests = 10

params["batch_size"] = 1
params["tt_flag"] = 1
params["device"] = "cpu"



with open("error_reduction_table.txt", "w") as f:
    f.write(f"IC_idx = {IC_idx}\n")

    row_str = ""  # this will store one row

    for abl_idx in range(10):
        params["ablation_idx"] = abl_idx
        error_reduction = []

        for j in range(num_tests):
            error_red = testing(params, j)
            error_reduction.append(error_red)

        error_reduction = torch.tensor(error_reduction)
        mean_val = error_reduction.mean().item()
        std_val = error_reduction.std().item()

        # append formatted entry to row
        row_str += f"{mean_val:.4f} \\pm {std_val:.4f}\t"

    # write the full row once
    f.write(row_str + "\n")

print("Tables saved to error_reduction_table.txt")