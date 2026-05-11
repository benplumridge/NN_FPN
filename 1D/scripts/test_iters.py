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
# IC_idx = 2

num_tests = 5

params["batch_size"] = 1
params["tt_flag"] = 1
params["device"] = "cpu"
params["ablation_idx"] = 0 


IC_indices = [0, 1, 2]
Ns = [3, 7, 9]
#Ns = [3]

with open("error_reduction_table.txt", "w") as f:

    for IC_idx in IC_indices:
        params["IC_idx"] = IC_idx
        f.write(f"IC_idx = {IC_idx}\n")
        f.write("t\t" + "\t".join([f"N={N}" for N in Ns]) + "\n")  

        times = [0.5, 1.0]

        for T_val in times:
            params["T"] = T_val 
            params["num_t"] = int((T_val + params["dt"]) // params["dt"])
            row = [f"{T_val:.1f}"]

            for N in Ns:
                params["N"] = N
                error_reduction = []

                for j in range(num_tests):
                    error_red = testing(params, j)
                    error_reduction.append(error_red)

                error_reduction = torch.tensor(error_reduction)
                mean_val = error_reduction.mean().item()
                std_val = error_reduction.std().item()

                row.append(f"{mean_val:.4f} ± {std_val:.4f}")

            f.write("\t".join(row) + "\n")
        
        f.write("\n")  

print("Tables saved to error_reduction_table.txt")
