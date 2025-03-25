import torch.optim as optim
import torch 
import numpy as np
from funcs_common import SimpleNN, obj_func,  timestepping, compute_cell_average
from IC import holhraum
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
    
    NN_model = SimpleNN(num_features,num_hidden)

    #NN_model = torch.load("trained_models/model_N5.pth")

    if GD_optimizer == 'SGD':
        opt = optim.SGD(NN_model.parameters(), lr=learning_rate, momentum=momentum_factor)
    elif GD_optimizer == 'Adam':
        opt = optim.Adam(NN_model.parameters(), lr=learning_rate)
    NN_model = NN_model.to(device)

    psi0_nodes   = torch.zeros([num_IC,num_y+1,num_x+1])
    source_nodes = torch.zeros([num_IC,num_y+1,num_x+1])

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

    sigs = sigs.to(device)
    sigt = sigt.to(device)
    psi0   = compute_cell_average(psi0_nodes,num_x,num_y,num_IC)
    source = compute_cell_average(source_nodes,num_x,num_y,num_IC)

    psi0   = psi0.to(device)
    source = source.to(device)

    exact = timestepping(psi0, 0, 0, params,sigs, sigt, N_exact,num_basis_exact, source)[0]

    for l in range(num_epochs):
        opt.zero_grad()
        FPN  = timestepping(psi0, 1, NN_model, params, sigs,sigt,N,num_basis, source)[0]
        FPN  = FPN.to('cpu')
        exact = exact.to('cpu')    
        loss = obj_func(FPN-exact)
        loss.backward()
        opt.step()
        print('epoch', l)
        if torch.isnan(loss):
            print("NaN loss detected. Stopping training.")
            break
    return NN_model
