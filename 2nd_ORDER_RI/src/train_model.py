import torch.optim as optim
import torch 
import numpy as np
from funcs_common import SimpleNN, obj_func,  timestepping, compute_cell_average
from IC import gaussian, step, disc_source, bump, hat

def training(device,params):
    
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
    
    NN_model = SimpleNN(num_features,num_hidden)

    #NN_model = torch.load("trained_models/model_N5.pth")

    if GD_optimizer == 'SGD':
        opt = optim.SGD(NN_model.parameters(), lr=learning_rate, momentum=momentum_factor)
    elif GD_optimizer == 'Adam':
        opt = optim.Adam(NN_model.parameters(), lr=learning_rate)
    NN_model = NN_model.to(device)

    psi0_nodes   = torch.zeros([batch_size,num_y+1,num_x+1], device=device)
    source_nodes = torch.zeros([batch_size,num_y+1,num_x+1], device=device)


    reg1 = np.arange(0, batch_size//num_IC)
    gauss_out =  gaussian(num_x,num_y,x_edges,y_edges)
    psi0_nodes[reg1,:,:]   = gauss_out[0]
    source_nodes[reg1,:,:] = gauss_out[1]

    if num_IC == 2 or num_IC == 3:
        reg2 = np.arange(batch_size//num_IC, 2*batch_size//num_IC)
        step_out = step(num_x,num_y,x_edges,y_edges) 
        psi0_nodes[reg2,:,:]    = step_out[0]
        source_nodes[reg2,:,:]  = step_out[1]  

    if num_IC == 3 or num_IC == 4:
        reg3 = np.arange(2*batch_size//num_IC, batch_size)
        disc_source_out = disc_source(num_x,num_y,x_edges,y_edges) 
        psi0_nodes[reg3,:,:] = disc_source_out[0]
        source_nodes[reg3,:,:]  = disc_source_out[1]   
    
    if num_IC ==4 or num_IC == 5:
        reg4 = np.arange(3*batch_size//num_IC, batch_size)
        bump_out = bump(num_x,num_y,x_edges,y_edges) 
        psi0_nodes[reg4,:,:]   = bump_out[0]
        source_nodes[reg4,:,:] = bump_out[1] 

    if num_IC == 5:
        reg5 = np.arange(4*batch_size//num_IC, batch_size)
        hat_out = hat(num_x,num_y,x_edges,y_edges) 
        psi0_nodes[reg5,:,:]   = hat_out[0]
        source_nodes[reg5,:,:] = hat_out[1] 
    
    psi0   = compute_cell_average(psi0_nodes,num_x,num_y,batch_size)
    source = compute_cell_average(source_nodes,num_x,num_y,batch_size)

    for l in range(num_epochs):
        opt.zero_grad()
        sigs  = torch.rand(batch_size)*sigs_max
        siga  = (1-sigs)*torch.rand(batch_size)
        sigt  = sigs + siga 
        exact = timestepping(psi0, 0, 0, params,sigs, sigt, N_exact,num_basis_exact, source)[0]

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
