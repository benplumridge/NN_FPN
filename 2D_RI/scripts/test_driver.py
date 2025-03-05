import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from params_common import params
from test_model import testing 


### IC INDEX
# 0 - Gaussian
# 1 - Step 
# 2 - Discontinuous source
# 3 - Bump
# 4 - Hat
# 5 - Holhraum
params['IC_idx']     = 5

params['batch_size'] = 1
params['tt_flag']    = 1
params['device']     = 'cpu'

testing(params)



