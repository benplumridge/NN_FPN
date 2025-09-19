# IC INDEX
# 0 - Vanishing Cross Section
# 1 - Discontinuous Cross Section
# 2 - Gaussian
# 3 - Heavi-side
# 4 - Bump
# 5 - Discontinuous Source
# 6 - Reeds
IC_idx = 1

# N = 3:  sigf = 24.1
# N = 7:  sigf = 13.2
# N = 9:  sigf = 9.2
sigf = 9.2

import sys
import os

# Add src to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from params_common import params
from test_constant_model import testing

params["batch_size"] = 1
params["tt_flag"] = 1
params["device"] = "cpu"
params["IC_idx"] = IC_idx
params["sigf"] = sigf

testing(params)
