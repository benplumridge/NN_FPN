import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from IC import gaussian, step, disc_source, bump, hat, holhraum
from funcs_common import obj_func, timestepping, compute_cell_average, rotation_test

def testing(params,save_flag):
    
    num_x   = params['num_x']
    num_y   = params['num_y']
    N_exact = params['N_exact']
    N       = params['N']
    num_basis = params['num_basis']
    num_basis_exact = params['num_basis_exact']
    IC_idx  = params['IC_idx']
    x       = params['x']
    y       = params['y']
    xl      = params['xl']
    xr      = params['xr']
    x_edges       = params['x_edges']
    y_edges       = params['y_edges']
    batch_size = params['batch_size']
    plot_idx   = params['plot_idx']
    device     = params['device']

    model_filename =  load_model(N)
    NN_model = torch.load(model_filename,map_location=torch.device(device))
    NN_model.to(device)
    NN_model.eval()

    with torch.no_grad():
        if IC_idx == 0:
            psi0_out, source_out, sigs_out, sigt_out = gaussian(num_x,num_y,x_edges,y_edges)
        elif IC_idx == 1:
            psi0_out, source_out , sigs_out, sigt_out = step(num_x,num_y,x_edges,y_edges)
        elif IC_idx == 2:
            psi0_out, source_out , sigs_out, sigt_out = disc_source(num_x,num_y,x_edges,y_edges)
        elif IC_idx == 3:  
            psi0_out, source_out , sigs_out, sigt_out = bump(num_x,num_y,x_edges,y_edges)
        elif IC_idx ==4:
            psi0_out, source_out , sigs_out, sigt_out = hat(num_x,num_y,x_edges,y_edges)
        elif IC_idx ==5:
            psi0_out, source_out , sigs_out, sigt_out, params = holhraum(params)
            #NOTE: Holhraum overwrites parameters defined in parameter file
            num_x   = params['num_x']
            num_y   = params['num_y']

        psi0_edges   = torch.zeros(batch_size,num_y + 1,num_x + 1, dtype=torch.float32)
        sigs_edges   = torch.zeros(batch_size,num_y + 1,num_x + 1, dtype=torch.float32)
        sigt_edges   = torch.zeros(batch_size,num_y + 1,num_x + 1, dtype=torch.float32)
        source_edges = torch.zeros(batch_size,num_y + 1,num_x + 1, dtype=torch.float32)
        
        psi0_edges[0,:,:]   = psi0_out
        sigs_edges[0,:,:]   = sigs_out 
        sigt_edges[0,:,:]   = sigt_out 
        source_edges[0,:,:] = source_out

        psi0   = compute_cell_average(psi0_edges,num_x,num_y,batch_size)
        sigs   = compute_cell_average(sigs_edges,num_x,num_y,batch_size)
        sigt   = compute_cell_average(sigt_edges,num_x,num_y,batch_size)  
        source = compute_cell_average(source_edges,num_x,num_y,batch_size)

        exact  = timestepping(psi0, 0, 0, params, sigs, sigt, N_exact,num_basis_exact, source)[0]

    exact =  exact[0,:,:].detach().numpy() 

    if save_flag == 0:
        np.savetxt("holhraum_63.txt", exact)
    if save_flag ==1:
        np.savetxt("linesource_63.txt", exact)



def load_model(N):
    valid_N = {3, 5, 7, 9}
    if N not in valid_N:
        raise ValueError(f"Invalid value for N: {N}. Expected one of {valid_N}.")

    filename = f"trained_models/model_N{N}.pth"

    return filename
