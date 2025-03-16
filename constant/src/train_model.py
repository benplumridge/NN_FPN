import torch.optim as optim
import torch 
import numpy as np
from funcs_common import SimpleNN, obj_func,  timestepping, compute_cell_average
from IC import gaussian, step, disc_source, bump, hat
from training_sources import frame_source, two_rect_source, gaussian_source, pulse_source

def training(params):
    
    num_x   = params['num_x']
    num_y   = params['num_y']
    N_exact = params['N_exact']
    N       = params['N']
    num_basis       = params['num_basis']
    num_basis_exact  = params['num_basis_exact']
    num_epochs  = params['num_epochs']
    batch_size  = params['batch_size']
    learning_rate = params['learning_rate']
    momentum_factor = params['momentum_factor']
    num_features = params['num_features']
    num_hidden = params['num_hidden']
    x_edges = params['x_edges']
    y_edges = params['y_edges']
    sigs_max = params['sigs_max']
    GD_optimizer = params['GD_optimizer']
    num_IC       = params['num_IC']
    device       = params['device']
    
    NN_model = SimpleNN()

    #NN_model = torch.load("trained_models/model_N5.pth")

    if GD_optimizer == 'SGD':
        opt = optim.SGD(NN_model.parameters(), lr=learning_rate, momentum=momentum_factor)
    elif GD_optimizer == 'Adam':
        opt = optim.Adam(NN_model.parameters(), lr=learning_rate)
    NN_model = NN_model.to(device)

    psi0_nodes   = torch.zeros([num_IC,num_y+1,num_x+1])
    source_nodes = torch.zeros([num_IC,num_y+1,num_x+1])

    psi0_nodes[0,:,:] = gaussian(num_x,num_y,x_edges,y_edges)[0]
    psi0_nodes[1,:,:] = step(num_x,num_y,x_edges,y_edges)[0]   
    psi0_nodes[2,:,:] = disc_source(num_x,num_y,x_edges,y_edges)[0]   
    psi0_nodes[3,:,:] = bump(num_x,num_y,x_edges,y_edges)[0]  
    psi0_nodes[4,:,:] = hat(num_x,num_y,x_edges,y_edges)[0] 

    #assemble sources
    source_nodes[0,:,:] = 0
    source_nodes[1,:,:] = frame_source(num_x,num_y,x_edges,y_edges)
    source_nodes[2,:,:] = two_rect_source(num_x,num_y,x_edges,y_edges)
    source_nodes[3,:,:] = gaussian_source(num_x,num_y,x_edges,y_edges)
    source_nodes[4,:,:] = pulse_source(num_x,num_y,x_edges,y_edges)
    
    psi0   = compute_cell_average(psi0_nodes,num_x,num_y,num_IC)
    source = compute_cell_average(source_nodes,num_x,num_y,num_IC)

    psi0   = psi0.to(device)
    source = source.to(device)

    for l in range(num_epochs):
        opt.zero_grad()
        sigs  = torch.rand(batch_size)*sigs_max
        siga  = (1-sigs)*torch.rand(batch_size)
        sigt  = sigs + siga 
        sigs = sigs.to(device)
        sigt = sigt.to(device)

        training_indices = torch.randint(0, num_IC, (batch_size, 2)) 
        psi0_training = psi0[training_indices[:,0],:,:]
        source_training = source[training_indices[:,1],:,:]

        # psi0_training    = torch.zeros([batch_size,num_y,num_x])
        # source_training  = torch.zeros([batch_size,num_y,num_x])
        # for j in range(batch_size):
        #     psi0_training[j,:,:]   = psi0[training_indices[j,0],:,:]
        #     source_training[j,:,:] = source[training_indices[j,1],:,:]
        
        exact = timestepping(psi0_training, 0, 0, params,sigs, sigt, N_exact,num_basis_exact, source_training)[0]

        FPN, sigf  = timestepping(psi0_training, 1, NN_model, params, sigs,sigt,N,num_basis, source_training)
        print('sigf = ', sigf)
        FPN  = FPN.to('cpu')
        exact = exact.to('cpu')    
        loss = obj_func(FPN-exact)
        loss.backward()
        opt.step()
        print('epoch', l)
        if torch.isnan(loss):
            print("NaN loss detected. Stopping training.")
            break
        
    return NN_model,sigf
