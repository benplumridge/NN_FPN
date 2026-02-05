# IC INDEX
# 0 - Gaussian
# 1 - Vanishing Cross Section
# 2 - Discontinuous Cross Section
# 3 - Heavi-side
# 4 - Bump
# 5 - Discontinuous Source
# 6 - Reeds
IC_idx = 0

import sys
import os
import numpy as np

# Add src to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from params_common import params
from test_model import testing

params["batch_size"] = 1
params["tt_flag"] = 1
params["device"] = "cpu"
params["IC_idx"] = IC_idx
params["const_net"] = False
#params["model_idx"] = 1
params["ablation_idx"] = 0

num_its = 5
with open("testing_times.txt", "w") as run_times:
    run_times.write("# exact_time  PN_time  FPN_time\n")
    for j in range(num_its):
        testing(params, j, run_times)


data = np.loadtxt("testing_times.txt", comments="#")

means = np.mean(data, axis=0)
stds  = np.std(data, axis=0, ddof=1)

print(f"Mean times    (Exact, PN, FPN): {means}")
print(f"Std dev times (Exact, PN, FPN): {stds}")
