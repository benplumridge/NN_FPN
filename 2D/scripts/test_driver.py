import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from params_common import params, resolve_device
from test_model import testing


### IC INDEX
# 0 - Gaussian
# 1 - Step
# 2 - Discontinuous source
# 3 - Bump
# 4 - Hat
# 5 - Holhraum
# 6 - lattice
# 7 - Guassian (used in training)
params["IC_idx"] = 7

params["batch_size"] = 1
params["tt_flag"] = 1
params["device"] = resolve_device(params.get("device"))

FPN_error, FPN_error_reduction = testing(params)
