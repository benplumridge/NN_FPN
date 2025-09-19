import sys
import os
import torch
import numpy as np

# Add src to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from params_common import params
from train_model import training
from test_model import testing

# -------------------------
# User settings
# -------------------------
num_iterations = 10
N_values = [7]          
IC_idx_short = [0, 1, 2]    # ICs for short T
IC_idx_long = [6]           # ICs for long T
T_short = [0.5, 1]
T_long = [5, 10]

# Base training parameters
params["num_IC"] = 4
params["batch_size"] = 20
params["num_epochs"] = 500
#params["learning_rate"] = 1e-3
params["momentum_factor"] = 0.9
params["sigs_max"] = 1
GD_idx = 1
params["GD_optimizer"] = "SGD" if GD_idx == 0 else "Adam"
params["device"] = "cpu"  # or "cuda" if GPU available
params["IC_idx"] = 0
# -------------------------
# Storage for errors
# -------------------------
errors_list = []
models      = []
# -------------------------
# Main loop
# -------------------------
for N in N_values:
    for it in range(num_iterations):
        print(f"\n=== Iteration {it+1}/{num_iterations} ===")    
    
        params["tt_flag"] = 0
        # Update network parameters
        params['N'] = N
        params['num_features'] = 2 * N + 4
        params['num_hidden'] = N + 2

        if N == 3:
            params["learning_rate"] = 1e-2
        elif N in [7,9]:
            params["learning_rate"] = 1e-3
        # Train model
        print(f"Training model N={N} ...")
        model = training(params)
        models.append(model.state_dict()) 
        params["tt_flag"] = 1
    torch.save(models, f"trained_models/models_N{N}.pth")

        # ----- Short T testing (default spatial setup) -----
        #for IC_idx in IC_idx_short:

        #     params["IC_idx"] = IC_idx
        #     for T in T_short:
        #         params["T"] = T
        #         dt = params["dx"] / 2
        #         params["dt"] = dt
        #         params["num_t"] = int((T + dt) // dt)
                
        #         err = testing(params, model)
        #         errors_list.append(f"{it},{N},{IC_idx},{T},{err}")
        #         print(f"Iteration {it}, N={N}, IC={IC_idx}, T={T}, Error={err}")
        
        # # ----- Long T testing (override spatial setup) -----
        # for IC_idx in IC_idx_long:
        #     params["IC_idx"] = IC_idx
        #     # Set special spatial mesh for long T test
        #     num_x_long = 512
        #     xl, xr = 0, 8
        #     L_long = xr - xl
        #     dx_long = L_long / num_x_long
        #     params["num_x"] = num_x_long
        #     params["dx"] = dx_long
            
        #     for T in T_long:
        #         params["T"] = T
        #         dt = params["dx"] / 2
        #         params["dt"] = dt
        #         params["num_t"] = int((T + dt) // dt)
                
        #         err = testing(params, model)
        #         errors_list.append(f"{it},{N},{IC_idx},{T},{err}")
        #         print(f"Iteration {it}, N={N}, IC={IC_idx}, T={T}, Error={err}")
    
        #params["learning_rate"] = 1e-3
# -------------------------
# Save all errors to a text file
# -------------------------
    

# errors_file = "results/errors_all.txt"
# os.makedirs(os.path.dirname(errors_file), exist_ok=True)
# with open(errors_file, "w") as f:
#     f.write("iteration,N,IC_idx,T,error\n")
#     f.write("\n".join(errors_list))

# print(f"\nAll errors saved to {errors_file}")

# print("Mean relative error = ", np.mean())