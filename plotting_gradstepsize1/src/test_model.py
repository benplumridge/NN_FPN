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
    xl      = params['xl']
    xr      = params['xr']
    x_edges       = params['x_edges']
    y_edges       = params['y_edges']
    batch_size = params['batch_size']
    plot_idx   = params['plot_idx']
    device     = params['device']
    T          = params['T']

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
            x       = params['x']
            xl      = params['xl']
            xr      = params['xr']
            y       = params['y']
            T       = params['T']

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
        # if IC_idx == 0:
        #     exact_np = np.loadtxt("linesource_63.txt")
        # elif IC_idx ==5:
        #     if T == 2:
        #         exact_np = np.loadtxt("holhraum_63_T2.txt")
        #     elif T==4:
        #         exact_np = np.loadtxt("holhraum_63_T4.txt")
        # exact    = torch.zeros(batch_size,num_y,num_x)
        # exact[0,:,:]  = torch.from_numpy(exact_np)

        PN     = timestepping(psi0, 0, 0, params, sigs, sigt, N,num_basis, source)[0]

        sq_error0  = torch.sqrt(obj_func(PN-exact)/obj_func(exact))

        FPN,sigf = timestepping(psi0, 1, NN_model, params, sigs, sigt, N,num_basis, source)
        sq_errorf    = torch.sqrt(obj_func(FPN -exact)/obj_func(exact))

    error_reduction = sq_errorf/sq_error0
    print('PN error = ', sq_error0,   'FPN error = ', sq_errorf ,  'error_reduction = ', error_reduction)

    sigf  =  sigf.detach().numpy()
    exact =  exact[0,:,:].detach().numpy() 
    PN    =  PN[0,:,:].detach().numpy()
    FPN   =  FPN[0,:,:].detach().numpy()

    np.savetxt("holhraum37_T2.txt", exact)
    #np.savetxt("holhraum37_T4.txt", exact)
    #np.savetxt("linesource_37.txt", exact)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    #levels = np.linspace(np.min(PN), np.max(PN), num=100)  
    levels = 100

    plt.rcParams.update({'font.size': 16})
    
    cmap = mpl.cm.magma
    
    # Plot exact solution in the first subplot
    # contour1 = axs[0].contourf(y, x, exact, levels, cmap=cmap)
    # axs[0].set_title(f"$P_{{{N_exact}}}$")

    # Plot PN solution in the second subplot
    contour2 = axs[0].contourf(y, x, PN, levels, cmap=cmap)
    axs[0].set_title(f"$P_{{{N}}}$")

    # Plot FPN solution in the third subplot
    contour3 = axs[1].contourf(y, x, FPN, levels, cmap=cmap)
    axs[1].set_title(f"$FP_{{{N}}}$")

    # Add a single shared colorbar for the last plot
    fig.colorbar(contour2, ax=axs, orientation='vertical', shrink=0.8)
    
    for ax in axs.flat:
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)


    fig, axs = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    #levels = np.linspace(np.min(PN), np.max(PN), num=100)  
    levels = 100

    plt.rcParams.update({'font.size': 16})
    
    #cmap = mpl.cm.gist_ncar
    #cmap = mpl.cm.nipy_spectral
    cmap = mpl.cm.magma
    
    # Plot exact solution in the first subplot
    contour1 = axs.contourf(y, x, exact, levels, cmap=cmap)
    axs.set_title(f"$P_{{{N_exact}}}$")
    axs.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    axs.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)


    # Add a single shared colorbar for the last plot
    fig.colorbar(contour1, ax=axs, orientation='vertical', shrink=0.8)

    floor = 1e-10

    exact = np.maximum(floor,exact)
    PN    = np.maximum(floor,PN)
    FPN   = np.maximum(floor,FPN)

    logexact = np.log10(exact) 
    logPN    = np.log10(PN)
    logFPN   = np.log10(FPN)
    levels = np.linspace(-10, 0, num=100)

    # rot_error = rotation_test(FPN)
    # print('Rotation error =', rot_error)


    # PN_sym = PN - np.flip(PN, axis = 0)
    # print('Max symmetry error =', np.max(PN_sym))

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    #levels = np.linspace(np.min(PN), np.max(PN), num=100)  
    #levels = 100

    plt.rcParams.update({'font.size': 16})
    
    #cmap = mpl.cm.gist_ncar
    #cmap = mpl.cm.nipy_spectral
    cmap = mpl.cm.jet
    
    # Plot exact solution in the first subplot
    # contour1 = axs[0].contourf(y, x, exact, levels, cmap=cmap)
    # axs[0].set_title(f"$P_{{{N_exact}}}$")

    # Plot PN solution in the second subplot
    contour2 = axs[0].contourf(y, x, logPN, levels, cmap=cmap)
    axs[0].set_title(f"$P_{{{N}}}$")

    # Plot FPN solution in the third subplot
    contour3 = axs[1].contourf(y, x, logFPN, levels, cmap=cmap)
    axs[1].set_title(f"$FP_{{{N}}}$")

    # Add a single shared colorbar for the last plot
    fig.colorbar(contour2, ax=axs, orientation='vertical', shrink=0.8)
    
    for ax in axs.flat:
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)


    fig, axs = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    #levels = np.linspace(np.min(PN), np.max(PN), num=100)  
    levels = 100

    plt.rcParams.update({'font.size': 16})
    
    #cmap = mpl.cm.gist_ncar
    #cmap = mpl.cm.nipy_spectral
    cmap = mpl.cm.jet
    
    # Plot exact solution in the first subplot
    contour1 = axs.contourf(y, x, logexact, levels, cmap=cmap)
    axs.set_title(f"$P_{{{N_exact}}}$")
    axs.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    axs.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)


    # Add a single shared colorbar for the last plot
    fig.colorbar(contour1, ax=axs, orientation='vertical', shrink=0.8)

    plt.rcParams.update({'font.size': 16}) 
    fig, ax1 = plt.subplots()
    plt.xlim(xl,xr)
    line1, = ax1.plot(x, exact[plot_idx ,:], color='r', label= f'$P_{{{N_exact}}}$')
    line2, = ax1.plot(x, PN[plot_idx ,:], linestyle='--', color='b', label=f'$P_{{{N}}}$')
    line3, = ax1.plot(x, FPN[plot_idx ,:], linestyle='-.', color='g', label=f'$FP_{{{N}}}$')  

    ax2 = ax1.twinx()

    line4, = ax2.plot(x, sigf[plot_idx,:], linestyle=':', color='m', label=r'$\sigma_f$')

    # Combine all lines and labels
    lines = [line1, line2, line3, line4] 
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, bbox_to_anchor=(1.08, 1.15), ncol=4, frameon=False)

    figsig, axsig = plt.subplots(constrained_layout=True)
    # Add legend explicitly
    
    fig, ax1 = plt.subplots()
    plt.xlim(xl,xr)
    exact45 = exact[np.arange(exact.shape[0]), np.arange(exact.shape[0])]
    PN45    = PN[np.arange(PN.shape[0]), np.arange(PN.shape[0])]
    FPN45   = FPN[np.arange(FPN.shape[0]), np.arange(FPN.shape[0])]
    line1, = ax1.plot(x, exact45, color='r', label= f'$P_{{{N_exact}}}$')
    line2, = ax1.plot(x, PN45, linestyle='--', color='b', label=f'$P_{{{N}}}$')
    line3, = ax1.plot(x, FPN45, linestyle='-.', color='g', label=f'$FP_{{{N}}}$')  

    ax2 = ax1.twinx()

    line4, = ax2.plot(x, sigf[plot_idx,:], linestyle=':', color='m', label=r'$\sigma_f$')

    # Combine all lines and labels
    lines = [line1, line2, line3, line4] 
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, bbox_to_anchor=(1.08, 1.15), ncol=4, frameon=False)


    # Plot exact solution in the first subplot
    contour_sig = axsig.contourf(y, x, sigf, levels, cmap=cmap )
    axsig.set_title(fr"$\sigma_f$, $t = {T}$")
    axsig.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    axsig.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

    # Add a single shared colorbar for the last plot
    figsig.colorbar(contour_sig, ax=axsig, orientation='vertical', shrink=0.8)

    plt.show()

def load_model(N):
    valid_N = {3, 5, 7, 9}
    if N not in valid_N:
        raise ValueError(f"Invalid value for N: {N}. Expected one of {valid_N}.")

    filename = f"trained_models/model_N{N}.pth"

    return filename
