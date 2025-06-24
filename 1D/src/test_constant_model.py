import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from funcs_common import SimpleNN, obj_func, timestepping, compute_cell_average
from IC import gaussian, heaviside, bump,  disc_source, vanishing_cs, disc_cs, reeds


def testing(params):
    
    num_x   = params['num_x']
    num_t   = params['num_t']
    N_exact = params['N_exact']
    N       = params['N']
    batch_size = params['batch_size']
    x       = params['x']
    xl      = params['xl']
    xr      = params['xr']
    x_edges = params['x_edges']
    dx      = params['dx']
    device  = params['device']
    IC_idx  = params['IC_idx']

    NN_model = 0

    with torch.no_grad():
        exact  = torch.zeros([num_t+1,num_x,N_exact+1])

        if IC_idx == 0:
            psi0_out,sigs_out,sigt_out,source_out  = vanishing_cs(num_x,x_edges)         
        elif IC_idx ==1:
            psi0_out,sigs_out,sigt_out,source_out = disc_cs(num_x,x_edges)
        elif IC_idx == 2:
            psi0_out,sigs_out,sigt_out,source_out  = gaussian(num_x,x_edges)
        elif IC_idx == 3:
            psi0_out,sigs_out,sigt_out,source_out  = heaviside(num_x,x_edges)
        elif IC_idx == 4:
            psi0_out,sigs_out,sigt_out,source_out  = bump(num_x,x_edges)
        elif IC_idx == 5:
            psi0_out,sigs_out,sigt_out,source_out  = disc_source(num_x,x_edges)
        elif IC_idx == 6:
            psi0_out,sigs_out,sigt_out,source_out,params  = reeds(params)
            x       = params['x']
            xl      = params['xl']
            xr      = params['xr']
            x_edges = params['x_edges']
            dx      = params['dx']
            num_x   = params['num_x']
            num_t   = params['num_t']
            
        psi0_edges        = torch.zeros(batch_size,num_x + 1)
        sigs_edges        = torch.zeros(batch_size,num_x + 1)
        sigt_edges        = torch.zeros(batch_size,num_x + 1)
        source_edges      = torch.zeros(batch_size,num_x + 1)
        psi0_edges[0,:]   = psi0_out
        sigs_edges[0,:]   = sigs_out
        sigt_edges[0,:]   = sigt_out
        source_edges[0,:] = source_out

        psi0   = compute_cell_average(psi0_edges, batch_size,num_x)
        sigs   = compute_cell_average(sigs_edges, batch_size,num_x,)
        sigt   = compute_cell_average(sigt_edges,batch_size,num_x)  
        source = compute_cell_average(source_edges,batch_size,num_x)

        exact  = timestepping(psi0, 0, 0, params, sigs, sigt, N_exact, source,batch_size,device)[0]

        FPN      = torch.zeros([num_t+1,num_x,N+1])

        PN    = timestepping(psi0, 0, 0, params, sigs, sigt, N, source,batch_size,device)[0]
        sq_error0     = obj_func(PN-exact,dx)

        FPN, sigf     = timestepping(psi0, 2, NN_model, params, sigs, sigt, N, source,batch_size,device)

        sq_errorf  = obj_func(FPN -exact,dx)

        error_reduction = sq_errorf/sq_error0
        print('error_reduction = ', error_reduction)

    sigf =  sigf[0,:].detach().numpy()
    exact = exact[0,:].detach().numpy()
    PN = PN[0,:].detach().numpy()
    FPN    = FPN[0,:].detach().numpy()
   
    plt.rcParams.update({'font.size': 16}) 
    fig, ax1 = plt.subplots()
    
    # Plot on the first y-axis (left side)
    line1, = ax1.plot(x, exact/np.sqrt(2), label='Exact', color='r')
    line2, = ax1.plot(x, PN/np.sqrt(2), linestyle='--', color='b', label='y_PN')
    line3, = ax1.plot(x, FPN/np.sqrt(2), linestyle='-.', color='g', label='NN Filter')

    # Set labels and limits 
    ax1.set_xlim([xl, xr])
    ax1.set_ylabel('Scalar Flux')
    ax1.set_xlabel('x')

    # Create a second y-axis that shares the same x-axis
    ax2 = ax1.twinx()

    line4, = ax2.plot(x, sigf, linestyle=':', color='m', label=r'$\sigma_f$')
    ax2.set_ylabel(r'$\sigma_f$')
    lines = [line1, line2, line3, line4]  # Combine line objects
    labels = [line.get_label() for line in lines]  # Get labels for the lines
    ax1.legend(lines, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.15))
    plt.show()

    return 0