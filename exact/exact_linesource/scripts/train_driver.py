import sys
import os
import numpy as np
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from params_common import params
from train_model import training


params['num_IC']          = 5
params['batch_size']      = 5 ## make batch size a multiple of the number of Initial Conditions
params['num_epochs']      = 50
params['learning_rate']   = 1e0
params['momentum_factor'] = 0.9
params['sigs_max']        = 1
params['tt_flag']    = 0
params['IC_idx']     = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
params['device'] = device

#Optimizer flag
#0  - SGD
#1  - Adam
GD_opt_flag  = 0

if GD_opt_flag ==0:     
    params['GD_optimizer']    = 'SGD'
elif GD_opt_flag ==1:
    params['GD_optimizer']    = 'Adam'

NN_model = training(params)
N = params['N']
filename = f"model_N{N}.pth"
torch.save(NN_model, filename)   