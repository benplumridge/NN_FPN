import torch
import numpy as np
import torch.nn as nn


class SimpleNN(nn.Module):
    def __init__(self, num_features, num_hidden):
        super(SimpleNN, self).__init__()
        self.hidden1 = nn.Linear(num_features, num_hidden)  # (inputs,hidden)
        #self.bn1 = nn.BatchNorm1d(num_features)
        self.bn5 = nn.BatchNorm1d(num_hidden)
        self.output = nn.Linear(num_hidden, 1)  # (hidden,output)

    def forward(self, x):
        # print("Input shape:", x.shape)  # Debugging line
        original_shape = x.shape
        x = torch.flatten(x, start_dim=0, end_dim=1)
        # print("Flattened input shape:", x.shape)  # Debugging line
        #x = self.bn1(x)
        x = torch.relu(self.hidden1(x))  # Activation hidden layer
        x = self.bn5(x)
        x = torch.relu(self.output(x))  # Activation output layer
        output_shape = [original_shape[0], original_shape[1], 1]
        return x.reshape(output_shape)

def timestepping(y0, filt_switch, NN_model, params,  sigs, sigt, N, source, batch_size,device):
    dt      = params['dt']
    dx      = params['dx']
    tt_flag = params['tt_flag']
    IC_idx  = params['IC_idx']

    num_x = params['num_x']
    num_t = params['num_t']
    filter_func = params['filter']

    # CONSTRUCT A vector: does not need updating
    a = torch.zeros(N)

    for n in range(1, N+1):
        a[n-1] = n / np.sqrt((2*n-1)*(2*n+1))
    A = torch.diag(a, 1) + torch.diag(a, -1)

    eigA, V = torch.linalg.eig(A)
    eigA    = torch.real(eigA)
    V       = torch.real(V)
    absA    = torch.matmul(torch.matmul(V,torch.diag(torch.abs(eigA))),V.T)  

    B = - absA/dx
    C = 0.5*(absA - A)/dx
    D = 0.5*(absA + A)/dx

    B = B.to(device)
    C = C.to(device)
    D = D.to(device)
    source = source.to(device)
    filter_func = filter_func.to(device)
    A = A .to(device)
    absA = absA.to(device)
    sigt = sigt.to(device)
    sigs = sigs.to(device)

    y_prev = torch.zeros([batch_size,num_x,N+1],device=device)  
    y_prev[:,:,0] = y0
    y     = y_prev
    d = torch.zeros(num_x-1)
    d[0:num_x-1]  = 0.5/dx
    Der     = torch.diag(d,1) - torch.diag(d,-1)
    Der[0,num_x-1] = -0.5/dx
    Der[num_x-1,0] =  0.5/dx
   
    Der     = Der.to(device)

    y_avg     = torch.mean(y0,dim=1)  
    y_avg     = y_avg[:,None,None]  
    source_in = source[:,:,None]

    if tt_flag == 0:
        sigt_in   = sigt[:,None,None]
        sigs_in   = sigs[:,None,None]
    if tt_flag == 1:
        sigt_in   = sigt[:,:,None]
        sigs_in   = sigs[:,:,None]

    for k in range(1, num_t + 1):
     
        y1_update = PN_update(params, y_prev, A, absA, B, C, D,  N, source, 
              batch_size,device,filt_switch,y_avg,Der,NN_model,source_in,sigt_in,sigs_in,k)[0]
        y1 = y_prev + dt*y1_update

        #boundary conditions for Reeds problem: reflecting at x = 0 and vacauum at x = 8
        if IC_idx == 6:
            y1 = reeds_BC(y1,num_x,N)

        y2_update, sigf = PN_update(params, y1, A, absA, B, C, D,  N, source, 
              batch_size,device,filt_switch,y_avg,Der,NN_model,source_in,sigt_in,sigs_in,k)
        y = y + 0.5*dt*(y1_update + y2_update)

        #boundary conditions for Reeds problem: reflecting at x = 0 and vacauum at x = 8
        if IC_idx == 6:
            y = reeds_BC(y,num_x,N)
        y_prev = y

    return y, sigf

def PN_update(params, y_prev, A, absA, B, C, D,  N, source, 
              batch_size,device,filt_switch,y_avg,Der,NN_model,source_in,sigt,sigs,idx_t):

    L  = params['L']
    IC_idx = params['IC_idx']

    num_x = params['num_x']
    filter_type = params['filter_type']

    filter_func = params['filter']

    sigf = torch.zeros([batch_size,num_x],device=device)
    if filt_switch == 1:
        if filter_type ==0:
            y_reshaped = y_prev.permute(0,2,1)  
            Dy_reshaped = torch.matmul(y_reshaped,Der) 
            Dy = Dy_reshaped.permute(0,2,1)              
            A_Dy   = torch.matmul(Dy,A.T)           
            yflux  = y_prev[:,:,0]
            yflux = yflux[:,:,None]
            input  = preprocess_features(sigt*y_prev, A_Dy, source_in, sigs*yflux)
            network_output  = NN_model(input)
            sigf    = network_output[:, :, 0] 

        if filter_type == 1:
            sigf0 = NN_model 
            sigf = sigf0*torch.ones(batch_size,num_x)
        
        if IC_idx == 6:
            # if idx_t < 500:
            #     sigf = torch.zeros(batch_size,num_x)
            sigf= torch.clamp(sigf, max=400)
            sigf[:,0] = sigf[:,1]
            sigf[:,num_x-1] = sigf[:,num_x-2]
        

    if filt_switch == 2:
        sigf_const = params['sigf']
        sigf = sigf_const*torch.ones(batch_size,num_x)

    flux_limiter = torch.zeros([batch_size,num_x+2,N+1],device=device)
    y_update = torch.zeros([batch_size,num_x,N+1],device=device)
    y_expand = torch.zeros([batch_size,num_x+2,N+1],device=device)

    y_expand[:,1:num_x+1,:] = y_prev
    if IC_idx != 6:
        y_expand[:,0,:] = y_prev[:,num_x-1,:]
        y_expand[:,num_x+1,:] = y_prev[:,0,:]

    flux_limiter[:,1:num_x+1,:] = minmod(y_expand[:,2:num_x+2,:] - y_expand[:,1:num_x+1,:], 
                y_expand[:,1:num_x+1,:] - y_expand[:,0:num_x,:])
    
    flux_limiter[:,0,:] = flux_limiter[:,num_x+1,:]
    flux_limiter[:,num_x+1,:] = flux_limiter[:,1,:]

    B_y = torch.matmul(y_expand[:, 1:num_x+1, :], B.T) 
    C_y = torch.matmul(y_expand[:, 2:num_x+2, :], C.T)  
    D_y = torch.matmul(y_expand[:, :num_x, :], D.T)     

    sigf_y = sigf[:,:,None]*y_expand[:,1:num_x+1,:]*filter_func[:N+1]
    sigt_y = sigt*y_expand[:,1:num_x+1,:]                         

    flux_diff1 = flux_limiter[:,:num_x,:] - 2*flux_limiter[:,1:num_x+1,:] + flux_limiter[:,2:num_x+2,:]
    flux_diff2 = flux_limiter[:,2:num_x+2,:] - flux_limiter[:,:num_x,:]
    flux1 = 0.25*torch.matmul(flux_diff1, A.T)    
    flux2 = 0.25*torch.matmul(flux_diff2, absA.T)   
    y_update = B_y + C_y + D_y - sigf_y - sigt_y - flux1 + flux2

    y_update[:, :, 0] = y_update[:, :, 0] + sigs[:,:,0]*y_expand[:, 1:num_x+1, 0] + source

    return y_update, sigf

def preprocess_features(y, A_Dy, source,scattering):
    y_NN = y.clone()
    y_NN[...,1::2]  = torch.abs(y[...,1::2])  
    A_Dy_NN = A_Dy.clone()
    A_Dy_NN[...,1::2] = torch.abs(A_Dy[...,1::2]) 

    norm_y      = normalization(y_NN)
    norm_Dy     = normalization(A_Dy_NN)
    norm_scat   = normalization(scattering)
    norm_source = normalization(source)
    inputs = torch.cat(( 
                        norm_Dy,
                        norm_y, 
                        norm_scat,
                        norm_source                    
                        ), 
                        dim=-1)  

    return  inputs

def minmod(a, b):
    mm = torch.zeros_like(a)
    mm = torch.where((torch.abs(a) <= torch.abs(b)) & (a * b > 0), a, mm)
    mm = torch.where((torch.abs(b) < torch.abs(a)) & (a * b > 0), b, mm)
    return mm

def obj_func(z):
    obj_value = 0.5*torch.sum(torch.mean(z**2, dim=[1]))
    print(obj_value)
    return obj_value

def compute_cell_average(f,batch_size,num_x):
    f_average = torch.zeros(batch_size,num_x)
    for m in range(0,num_x):
        f_average[:,m] = 0.5*(f[:,m] + f[:,m+1])
    
    return f_average

def reeds_BC(z,num_x,N):
    for n in range(0,N,2):
        z[:,0,n] = z[:,1,n]
    for n in range(1,N+1,2):
        z[:,0,n] = -z[:,1,n]
    return z

def normalization(f):
    z = (f - torch.mean(f, dim=1, keepdim=True))/(torch.std(f, dim=1, keepdim=True) + 1e-8)
    return z