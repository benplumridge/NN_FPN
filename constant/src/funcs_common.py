import torch
import torch.nn as nn
import numpy as np

class SimpleNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.constant = torch.nn.Parameter(torch.randn(1))  # Initialize randomly

    def forward(self):
        return self.constant  # No input, just return the parameter

    
def obj_func(z):
    obj_value = torch.sum(torch.mean(z**2,dim = [1,2]))
    print(torch.sqrt(obj_value))
    return obj_value

def minmod(a, b):
    return 0.5*(torch.sign(a) + torch.sign(b)) * torch.min(torch.abs(a), torch.abs(b))

def compute_PN_matrices(N):  
    n_sys = (N+1)*(N+2)//2

    # Initialize Mx, My as sparse matrices
    Ax = torch.zeros((n_sys, n_sys), dtype=torch.float32)

    Ay = torch.zeros((n_sys, n_sys), dtype=torch.float32)

    sqrt2   = torch.sqrt(torch.tensor(2, dtype=torch.float32))

    # Loop through values of m
    for m in range(1, N + 1):
        i = torch.arange(1, m + 1)
        p = (m * (m - 1)) // 2 + i
        v = d_param(m, -m + 2 * (torch.ceil(i / 2) - 1)) 
        Ax[p - 1, p + m - 1] = v
        Ay[p - 1, p + m - 1 - (-1) ** i] = -(-1) ** i * v

        i = torch.arange(1, m)  # m - 1
        p = (m * (m - 1)) // 2 + i
        v = f_param(m, -m + 2 + 2 * (torch.ceil(i / 2) - 1))  
        Ax[p - 1, p + m + 1] = -v
        Ay[p - 1 - (-1) ** i, p + m + 1] = (-1) ** i * v

    # Apply sqrt(2) scaling to appropriate indices
    m = torch.arange(1, N + 1, 2)
    i = (m * (m + 1)) // 2
    Ax[i - 1, :] *= sqrt2
    Ay[i - 1, :] *= sqrt2

    m = torch.arange(2, N + 1, 2)
    i = ((m + 1) * (m + 2)) // 2
    Ax[:, i - 1] *= sqrt2
    Ay[:, i - 1] *= sqrt2
    
    # Symmetrize matrices
    Ax = (Ax + Ax.T) / 2
    Ay = (Ay + Ay.T) / 2   

    return Ax, Ay

def d_param(l,k):
    return  torch.sqrt(((l-k)*(l-k-1))/((2*l+1)*(2*l-1)))

def f_param(l,k):
    return  torch.sqrt(((l+k)*(l+k-1))/((2*l+1)*(2*l-1)))

def upwind_flux(N,num_basis,psi,params) :
    IC_idx = params['IC_idx']
    dx = params['dx']
    dy = params['dy']
    num_x = params['num_x']
    num_y = params['num_y']
    batch_size = params['batch_size']
    device = params['device']

    Ax, Ay = compute_PN_matrices(N)
    Ax = Ax.to(device)
    Ay = Ay.to(device)

    eig_Ax,Vx     = torch.linalg.eig(Ax)
    eig_Ax        = torch.real(eig_Ax)
    Vx            = torch.real(Vx)
    eig_Ax_plus   = torch.diag(torch.clamp(eig_Ax, min=0))
    eig_Ax_minus  = torch.diag(torch.clamp(eig_Ax, max=0))

    Ax_plus       = torch.matmul(torch.matmul(Vx,eig_Ax_plus),Vx.T)
    Ax_minus      = torch.matmul(torch.matmul(Vx,eig_Ax_minus),Vx.T)  

    eig_Ay,Vy     = torch.linalg.eig(Ay)
    eig_Ay        = torch.real(eig_Ay)
    Vy            = torch.real(Vy)
    eig_Ay_plus   = torch.diag(torch.clamp(eig_Ay, min=0))
    eig_Ay_minus  = torch.diag(torch.clamp(eig_Ay, max=0))

    Ay_plus       = torch.matmul(torch.matmul(Vy,eig_Ay_plus),Vy.T) 
    Ay_minus      = torch.matmul(torch.matmul(Vy,eig_Ay_minus),Vy.T)

    #Clean flux matrices
    threshold = 1e-6
    Ax_plus   = torch.where(torch.abs(Ax_plus)  < threshold, torch.zeros_like(Ax_plus, dtype=torch.float32),  Ax_plus.to(torch.float32))
    Ax_minus  = torch.where(torch.abs(Ax_minus) < threshold, torch.zeros_like(Ax_minus, dtype=torch.float32), Ax_minus.to(torch.float32))
    Ay_plus   = torch.where(torch.abs(Ay_plus)  < threshold, torch.zeros_like(Ay_plus, dtype=torch.float32),  Ay_plus.to(torch.float32))
    Ay_minus  = torch.where(torch.abs(Ay_minus) < threshold, torch.zeros_like(Ay_minus, dtype=torch.float32), Ay_minus.to(torch.float32))

    f_plus  = torch.zeros([batch_size,num_y,num_x,num_basis], device = device)
    f_minus = torch.zeros([batch_size,num_y,num_x,num_basis], device = device)
    g_plus  = torch.zeros([batch_size,num_y,num_x,num_basis], device = device)
    g_minus = torch.zeros([batch_size,num_y,num_x,num_basis], device = device)

    dx_left  = torch.zeros([batch_size,num_y,num_x,num_basis], device = device)
    dx_right = torch.zeros([batch_size,num_y,num_x,num_basis], device = device)
    dy_up    = torch.zeros([batch_size,num_y,num_x,num_basis], device = device)
    dy_down  = torch.zeros([batch_size,num_y,num_x,num_basis], device = device)

    #Compute differences for slopes in x-direction 
    dx_left[:,1:num_y-1,1:num_x,:]    = (psi[:,1:num_y-1,1:num_x,:] - psi[:,1:num_y-1,0:num_x-1,:])/dx 
    dx_right[:,1:num_y-1,0:num_x-1,:] = (psi[:,1:num_y-1,1:num_x,:] - psi[:,1:num_y-1,0:num_x-1,:])/dx  

    #Compute differences for slopes in y-direction
    dy_down[:,1:num_y, 1:num_x-1,:]  = (psi[:,1:num_y,1:num_x-1,:] - psi[:,0:num_y-1,1:num_x-1,:])/dy
    dy_up[:,0:num_y-1,1:num_x-1,:]   = (psi[:,1:num_y,1:num_x-1,:] - psi[:,0:num_y-1,1:num_x-1,:])/dy

    lim_x  = minmod(dx_left,dx_right)
    lim_y  = minmod(dy_down,dy_up)
    lim_x_plus  = torch.zeros_like(lim_x)
    lim_x_minus = torch.zeros_like(lim_x)
    lim_y_plus  = torch.zeros_like(lim_y)
    lim_y_minus = torch.zeros_like(lim_y)
    lim_x_plus[:,1:num_y-1,1:num_x-1,:]  = lim_x[:,1:num_y-1,2:num_x,:]   - lim_x[:,1:num_y-1,1:num_x-1,:] 
    lim_x_minus[:,1:num_y-1,1:num_x-1,:] = lim_x[:,1:num_y-1,1:num_x-1,:] - lim_x[:,1:num_y-1,0:num_x-2,:]   
    lim_y_plus[:,1:num_y-1,1:num_x-1,:]  = lim_y[:,2:num_y,1:num_x-1,:]   - lim_y[:,1:num_y-1,1:num_x-1,:] 
    lim_y_minus[:,1:num_y-1,1:num_x-1,:] = lim_y[:,1:num_y-1,1:num_x-1,:] - lim_y[:,0:num_y-2,1:num_x-1,:] 

    f_plus  = torch.matmul(dx_right - 0.5*lim_x_plus, Ax_minus.T)
    f_minus = torch.matmul(dx_left  + 0.5*lim_x_minus, Ax_plus.T)
    g_plus  = torch.matmul(dy_up    - 0.5*lim_y_plus, Ay_minus.T)
    g_minus = torch.matmul(dy_down  + 0.5*lim_y_minus, Ay_plus.T)

    f_plus[:,:,-1,:]  = torch.zeros([batch_size,num_y,num_basis], device = device)
    f_minus[:,:,0,:]  = torch.zeros([batch_size,num_y,num_basis], device = device)
    g_plus[:,-1,:,:]  = torch.zeros([batch_size,num_x,num_basis], device = device)
    g_minus[:,0,:,:]  = torch.zeros([batch_size,num_x,num_basis], device = device)
    
    if IC_idx == 5:
        source_boundary = params['source_boundary']
        f_minus[:,:,0,0] = (psi[0,:,1,0]-source_boundary)/dx
        
    fluxes  = f_plus + f_minus + g_plus + g_minus

    #partial derivatives for model features
    dx_psi = torch.zeros([batch_size,num_y,num_x,num_basis], device = device)
    dy_psi = torch.zeros([batch_size,num_y,num_x,num_basis], device = device)

    dx_psi[:,1:num_y-1, 1:num_x-1, :] = (psi[:,1:num_y-1, 2:num_x , :] - psi[:,1:num_y-1, 0:num_x-2, :])/(2*dx) 
    dy_psi[:,1:num_y-1, 1:num_x-1, :] = (psi[:,2:num_y , 1:num_x-1, :] - psi[:,0:num_y-2, 1:num_x-1, :])/(2*dy) 

    A_dxpsi = torch.matmul(dx_psi, Ax.T) 
    A_dypsi = torch.matmul(dy_psi, Ay.T)

    return fluxes, A_dxpsi, A_dypsi

def preprocess_features(N,psi, dxpsi, dypsi, scattering, source, params):
    num_x = params['num_x']
    num_y = params['num_y']
    batch_size = params['batch_size']
    device     = params['device']
    psi_norms  = torch.zeros([batch_size,num_y,num_x,N+1],device = device)
    dpsi_norms = torch.zeros([batch_size,num_y,num_x,N+1],device = device)

    index  = 0  

    for ell in range(N + 1):

        num_m   = ell + 1  
        ell_psi = psi[:,:,:,index : index + num_m]  
        
        norm_l_psi   = torch.linalg.norm(ell_psi, ord=2, dim = -1)  
        psi_norms[...,ell] = norm_l_psi 

        ell_dx = dxpsi[..., index : index + num_m]
        ell_dy = dypsi[..., index : index + num_m]

        dpsi_norms[..., ell] =  torch.linalg.norm(torch.sqrt(torch.clamp(ell_dx**2 + ell_dy**2, min=1e-12)), ord=2, dim=-1)

        index += num_m  

    scattering_in = NN_normalization(scattering[:,:,:,None]) 
    source_in     = NN_normalization(source[:,:,:,None])
    psi_in        = NN_normalization(psi_norms)
    dpsi_in       = NN_normalization(dpsi_norms)

    inputs = torch.cat((psi_in, dpsi_in, scattering_in, source_in),dim=-1)

    return inputs

def NN_normalization(f):
    f_mean = torch.mean(f, dim =[1,2],keepdim=True)
    f_std  = torch.std(f,dim=[1,2],keepdim=True)

    f_normalized = (f - f_mean)/(f_std + 1e-10)
    return f_normalized

def timestepping(psi0, filt_switch, NN_model, params, sigs, sigt, N,num_basis, source):

    num_x = params['num_x']
    num_y = params['num_x']
    num_t = params['num_t']
    dt    = params['dt']
    batch_size = params['batch_size']
    device       = params['device']
    

    psi_prev = torch.zeros([batch_size,num_y,num_x,num_basis],device=device)
    psi_prev[:,:,:,0] = psi0

    for k in range(1, num_t + 1):
        psi1_update       = PN_update(psi_prev,N,params,num_basis,sigt,sigs,filt_switch,source,NN_model)[0]
        psi1 = psi_prev + dt*psi1_update 
        psi2_update, sigf = PN_update(psi1,N,params,num_basis,sigt,sigs,filt_switch,source,NN_model)  
        psi = psi_prev + 0.5*dt*(psi1_update + psi2_update)
        psi_prev = psi
    return psi_prev[:,:,:,0], sigf

def PN_update(psi_prev,N,params,num_basis,sigt,sigs,filt_switch,source,NN_model): 

    num_x = params['num_x']
    num_y = params['num_y']
    num_features = params['num_features']
    batch_size = params['batch_size']
    filter = params['filter']
    tt_flag = params['tt_flag']
    IC_idx  = params['IC_idx']
    device       = params['device']
    filter = filter.to(device)
    
    fluxes, A_dxpsi, A_dypsi  = upwind_flux(N,num_basis,psi_prev,params)

    if tt_flag == 0:
        sigt_psi   = sigt[:,None,None,None]*psi_prev 
        scattering = sigs[:,None,None]**psi_prev[:,:,:,0]        
    elif tt_flag == 1: 
        sigt_psi    = sigt[:,:,:,None]*psi_prev
        scattering  = sigs*psi_prev[:,:,:,0]

    sigf       = torch.zeros([batch_size,num_y,num_x],device =device)
    psi_update = torch.zeros([batch_size,num_y,num_x,num_basis],device =device)
    inputs     = torch.zeros([batch_size,num_y,num_x,num_features],device =device)

    if filt_switch == 1:
        inputs =  preprocess_features(N,sigt_psi,A_dxpsi,A_dypsi,scattering,source,params)     
        #sigf = NN_model(inputs).squeeze(-1)
        sigf = NN_model()

    psi_update  =  - fluxes - sigt_psi 
    psi_update[:,:,:,0]  = psi_update[:,:,:,0] + scattering + source

    if filt_switch == 1:
        sigf_psi   = sigf*psi_prev*filter[0:num_basis]
        psi_update = psi_update - sigf_psi

    if IC_idx != 5:
      psi_update[:,:,0,:]   =  psi_update[:,:,1,:]

    psi_update[:,0,:,:]   =  psi_update[:,1,:,:]
    psi_update[:,-1,:,:]  =  psi_update[:,-2,:,:]
    psi_update[:,:,-1,:]  =  psi_update[:,:,-2,:]
    psi_update[:,0,0,:]   = 0.5*(psi_update[:,1,0,:] + psi_update[:,0,1,:] )
    psi_update[:,0,-1,:]  = 0.5*(psi_update[:,0,-2,:] + psi_update[:,1,-1,:] )
    psi_update[:,-1,0,:]  = 0.5*(psi_update[:,-2,0,:] + psi_update[:,-1,1,:] )
    psi_update[:,-1,-1,:] = 0.5*(psi_update[:,-2,-1,:] + psi_update[:,-1,-2,:] )    
        
    return psi_update, sigf

    
def compute_cell_average(f,num_x,num_y,num_funcs):
    average = torch.zeros(num_funcs,num_x,num_y,dtype = torch.float32)
    for l in range(0,num_y):
        for m in range(0,num_x):
            average[:,l,m] = 0.25*(f[:,l,m] + f[:,l,m+1] + f[:,l+1,m]  + f[:,l+1,m+1])
    return average

def reconstruction(f_coarse,x_fine,y_fine,x_coarse,y_coarse,num_x,num_y,dx,dy,
                   num_y_fine,num_x_fine,num_y_fine_factor,num_x_fine_factor):
    f_coarse_expanded = torch.zeros(num_y+2,num_x+2, dtype=f_coarse.dtype, device=f_coarse.device)
    f_coarse_expanded[1:-1,1:-1] = f_coarse

    # f_coarse_expanded[0,1:-1]  = f_coarse[0,:]
    # f_coarse_expanded[-1,1:-1] = f_coarse[-1,:]
    # f_coarse_expanded[1:-1,0]  = f_coarse[:,0]
    # f_coarse_expanded[1:-1,-1] = f_coarse[:,-1]
    # f_coarse_expanded[0, 0]    = f_coarse[0, 0]  
    # f_coarse_expanded[0, -1]   = f_coarse[0, -1] 
    # f_coarse_expanded[-1, 0]   = f_coarse[-1, 0] 
    # f_coarse_expanded[-1, -1]  = f_coarse[-1, -1]  

    grad_x = torch.zeros(num_y,num_x)
    grad_y = torch.zeros(num_y,num_x)

    grad_x = (f_coarse_expanded[1:-1,2:] - f_coarse_expanded[1:-1,:-2])/(2*dx) 
    grad_y = (f_coarse_expanded[2:,1:-1] - f_coarse_expanded[:-2,1:-1])/(2*dy) 

    grad_x[:,0] = (f_coarse[:,1] - f_coarse[:,0])/dx 
    grad_x[:,num_x-1] = (f_coarse[:,num_x-1] - f_coarse[:,num_x-2])/dx 
    grad_y[0,:] = (f_coarse[1,:] - f_coarse[0,:])/dx 
    grad_y[num_y-1,:] = (f_coarse[num_y-1,:] - f_coarse[num_y-2,:])/dx 

    f_fine = torch.zeros(num_y_fine,num_x_fine, dtype=f_coarse.dtype, device=f_coarse.device)

    for m in range(num_y):
        for n in range(num_x):
            for m_ref in range(num_y_fine_factor):
                for n_ref in range(num_x_fine_factor):
                    fine_m = m * num_y_fine_factor + m_ref
                    fine_n = n * num_x_fine_factor + n_ref  
                    f_fine[fine_m,fine_n] = (
                    f_coarse[m,n]
                    + grad_x[m,n]*(x_fine[fine_n] - x_coarse[n])
                    + grad_y[m,n]*(y_fine[fine_m] - y_coarse[m])
                    )


    for m in range(num_y):
        for m_ref in range(num_y_fine_factor):
            fine_m = m*num_y_fine_factor + m_ref 
            f_fine[fine_m,num_x_fine-1] = (
                    f_coarse[m,num_x-1]
                    + grad_x[m,num_x-1]*(x_fine[num_x_fine-1] - x_coarse[num_x-1])
                    + grad_y[m,num_x-1]*(y_fine[fine_m] - y_coarse[m])
                    )

    for n in range(num_x):
        for n_ref in range(num_x_fine_factor):
            fine_n = n * num_x_fine_factor + n_ref  
            f_fine[num_y_fine-1,fine_n] = (
            f_coarse[num_y-1,n]
            + grad_x[num_y-1,n]*(x_fine[fine_n] - x_coarse[n])
            + grad_y[num_y-1,n]*(y_fine[num_y_fine-1] - y_coarse[num_y-1])
            )    
    

    f_fine[num_y_fine-1,num_x_fine-1] =  (
            f_coarse[num_y-1,num_x-1]
            + grad_x[num_y-1,num_x-1]*(x_fine[num_x_fine-1] - x_coarse[num_x-1])
            + grad_y[num_y-1,num_x-1]*(y_fine[num_y_fine-1] - y_coarse[num_y-1])
            ) 

    return f_fine


def rotation_test(psi):
    rot_error = np.zeros(2)
    psi_rot = np.rot90(psi)
    rot_error[0] = np.max(np.abs(psi_rot - psi))
    psi_rot = np.rot90(psi)
    rot_error[1] = np.max(np.abs(psi_rot - psi))
    return rot_error