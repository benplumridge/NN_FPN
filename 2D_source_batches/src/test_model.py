import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from IC import gaussian, step, disc_source, bump, hat, holhraum
from funcs_common import obj_func, timestepping, compute_cell_average, rotation_test

def testing(params):
    
    num_x   = params['num_x']
    num_y   = params['num_y']
    N_exact = params['N_exact']
    N       = params['N']
    num_basis = params['num_basis']
    num_basis_exact = params['num_basis_exact']
    IC_idx  = params['IC_idx']
    x       = params['x']
    y       = params['y']
    x_edges       = params['x_edges']
    y_edges       = params['y_edges']
    batch_size = params['batch_size']
    plot_idx   = params['plot_idx']

    model_filename =  load_model(N)
    NN_model = torch.load(model_filename)
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
            x       = params['x']
            y       = params['y']

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
        PN     = timestepping(psi0, 0, 0, params, sigs, sigt, N,num_basis, source)[0]

        sq_error0  = obj_func(PN-exact)

        FPN,sigf = timestepping(psi0, 1, NN_model, params, sigs, sigt, N,num_basis, source)
        sq_errorf    = obj_func(FPN -exact)

    error_reduction = torch.sqrt(sq_errorf/sq_error0)
    print('error_reduction = ', error_reduction)

    sigf  =  sigf.detach().numpy()
    exact = exact[0,:,:].detach().numpy() 
    PN    = PN[0,:,:].detach().numpy()
    FPN   = FPN[0,:,:].detach().numpy()

    #rot_error = rotation_test(FPN)
    #print('Rotation error =', rot_error)

    fig, axs = plt.subplots(1, 3, figsize=(18, 4.5), constrained_layout=True)
    #levels = np.linspace(np.min(PN), np.max(PN), num=100)  
    levels = 100
    plt.rcParams.update({'font.size': 16})
    
    cmap = mpl.cm.jet

    # Plot exact solution in the first subplot
    contour1 = axs[0].contourf(y, x, exact, levels, cmap=cmap)
    axs[0].set_title(f"$P_{{{N_exact}}}$")

    # Plot PN solution in the second subplot
    contour2 = axs[1].contourf(y, x, PN, levels, cmap=cmap)
    axs[1].set_title(f"$P_{{{N}}}$")

    # Plot FPN solution in the third subplot
    contour3 = axs[2].contourf(y, x, FPN, levels, cmap=cmap)
    axs[2].set_title(f"$FP_{{{N}}}$")

    # Add a single shared colorbar for the last plot
    fig.colorbar(contour3, ax=axs, orientation='vertical', shrink=0.8)
    
    for ax in axs.flat:
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=True)

    plt.rcParams.update({'font.size': 16}) 
    fig, ax1 = plt.subplots()

    line1, = ax1.plot(x, exact[plot_idx ,:], color='r', label= f'$P_{{{N_exact}}}$')
    line2, = ax1.plot(x, PN[plot_idx ,:], linestyle='--', color='b', label=f'$P_{{{N}}}$')
    line3, = ax1.plot(x, FPN[plot_idx ,:], linestyle='-.', color='g', label=f'$FP_{{{N}}}$')  

    ax2 = ax1.twinx()

    line4, = ax2.plot(x, sigf[plot_idx,:], linestyle=':', color='m', label=r'$\sigma_f$')

    # Combine all lines and labels
    lines = [line1, line2, line3, line4] 
    labels = [line.get_label() for line in lines]

    # Add legend explicitly
    ax1.legend(lines, labels, bbox_to_anchor=(1.08, 1.15), ncol=4, frameon=False)

    plt.show()

def load_model(N):
    valid_N = {3, 5, 7, 9}
    if N not in valid_N:
        raise ValueError(f"Invalid value for N: {N}. Expected one of {valid_N}.")

    filename = f"trained_models/model_N{N}.pth"

    return filename
