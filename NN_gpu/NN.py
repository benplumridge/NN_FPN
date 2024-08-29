import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from numpy import asarray
from numpy import savetxt

class SimpleNN(nn.Module):
    def __init__(self,N):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(2*N+2, N+1)  # (inputs,hidden)
        self.output = nn.Linear(N+1, 1)      # (hidden,output)

    def forward(self, x):
        x = torch.relu(self.hidden(torch.abs(x)))  # Activation hidden layer
        x = torch.relu(self.output(torch.abs(x)))  # Activation output layer 
        return x

def minmod(a, b):
    mm = torch.zeros_like(a)
    mm = torch.where((torch.abs(a) <= torch.abs(b)) & (a * b > 0), a, mm)
    mm = torch.where((torch.abs(b) < torch.abs(a)) & (a * b > 0), b, mm)
    return mm

def initial_data_linesource(N,num_x,num_mu,glw,mu,x):
    z     = torch.zeros([num_x,num_mu])
    z_leg = torch.zeros([num_x,N+1])
    c     = 0   
    s     = 0.03

    for n in range(0,num_mu):
        z[:,n] = torch.exp(-((x-c)**2)/(2*s**2))

    for m in range(0,num_x):
         z_leg[m,:] = func2leg(z[m,:],N,glw,num_mu,mu)
   
    return z_leg

def initial_data_heaviside(N,num_x,num_mu,glw,mu,x):
    z     = torch.zeros([num_x,num_mu])
    z_leg = torch.zeros([num_x,N+1])

    for m in range(0,num_x):
        for n in range(0,num_mu):        
            if x[m] > -0.2 and x[m] < .2:
                z[m,n] = 1 

    for m in range(0,num_x):
         z_leg[m,:] = func2leg(z[m,:],N,glw,num_mu,mu)
   
    return z_leg

def initial_data_sin(N,num_x,num_mu,glw,mu,x):
    z     = torch.zeros([num_x,num_mu])
    z_leg = torch.zeros([num_x,N+1])
    k = 10
    for m in range(0,num_x):
        for n in range(0,num_mu): 
            z[m,n] = torch.sin(torch.pi*k*x[m])       

    for m in range(0,num_x):
         z_leg[m,:] = func2leg(z[m,:],N,glw,num_mu,mu)
   
    return z_leg

def initial_data_vanishing(N,num_x,num_mu,glw,mu,x):
    z      = torch.zeros([num_x,num_mu])
    sigs   = torch.zeros(num_x)  
    source = torch.zeros([num_t+1,num_x,N+1])
    z_leg  = torch.zeros([num_x,N+1])
    sigs   = 100*x**4  
    sigt  = sigs

    for m in range(0,num_x):
        for n in range(0,num_mu):     
            if x[m] > -0.2 and x[m] < .2:
                z[m,n] = 1 

    CN = 1/(N*sigs+1)

    for m in range(0,num_x):
         z_leg[m,:] = func2leg(z[m,:],N,glw,num_mu,mu)

    return z_leg, sigs, sigt,source,CN

def filter_func(z,p):
    return torch.exp(-z**p)

def legendre_poly(N,mu,num_mu):
    L = torch.zeros([num_mu,N+1])
    L[:,0] = torch.ones(num_mu)
    L[:,1] = mu
    for n in range(2,N+1):
        L[:,n]   = ((2*n-1)*torch.multiply(mu,L[:,n-1]) - (n-1)*L[:,n-2])/n
    for n in range(0,N+1):
        L[:,n]   = np.sqrt((2*n+1)/2)*L[:,n]
    return L

def func2leg(z,N,glw,num_mu,mu):
    L  = legendre_poly(N+1,mu,num_mu) 
    c = torch.zeros([N+1]) 
    for n in range(0,N+1):
        c[n] = torch.matmul(glw,torch.multiply(z,L[:,n]))
    return c 

def obj_func(z,params,mesh):
    obj_value = 0.5*dx*torch.mean(z**2)
    print(obj_value)
    return obj_value

def trapezoid(dx,z): 
    n = np.size(z)
    I  = (dx/2)*(z[0] + 2 * sum(z[1:n-1]) + z[n-1])
    return I

def timestepping_state(y0, filt_switch, NN_model, params, mesh, funcs, sigs, sigt, N, source,CN):
    dt = mesh['dt']
    dx = mesh['dx']

    num_x = params['num_x']
    num_t = params['num_t']
    
    filter_func = funcs['filter']

    # CONSTRUCT A vector: does not need updating
    a = torch.zeros(N)

    for n in range(1, N+1):
        a[n-1] = n / np.sqrt((2*n-1)*(2*n+1))
    A = torch.diag(a, 1) + torch.diag(a, -1)

    eigA, V = torch.linalg.eig(A)
    eigA    = torch.real(eigA)
    V       = torch.real(V)
    absA    = torch.matmul(torch.matmul(V,torch.diag(torch.abs(eigA))),V.T)  
    dtdx    = dt/dx

    B = torch.eye(N+1) - dtdx*absA
    C = 0.5*dtdx*(absA - A)
    D = 0.5*dtdx*(absA + A)

    y  = torch.zeros([num_t + 1, num_x,N+1])
    Dy = torch.zeros([num_t + 1, num_x,N+1])
    y_prev = y0
    y[0,:, :] = y0

    d = torch.zeros(num_x-1)
    d[1:num_x-1]  = 0.5/dx
    Der     = torch.diag(d,1) - torch.diag(d,-1)
    
    sigf = torch.zeros([num_t + 1,num_x])
    for k in range(1, num_t + 1):
        if filt_switch == 1:
            for n  in range(0,N+1):
                Dy[k,:,n] = torch.matmul(Der,torch.squeeze(y[k-1,:,n]))
                      
            for m in range(0,num_x):
                model_inputs = torch.concatenate((CN[m]*y[k-1,m,:],CN[m]*Dy[k-1,m,:]))
                sigf[k-1,m] = NN_model(model_inputs)
            
        y1 = PN_state(y_prev, num_x, sigf[k-1,:], sigs, sigt, A, absA, B, C, D, dt, 
                      filter_func, N, source[k-1,:, :])
        y[k,:, :] = 0.5*(y1 + PN_state( y1, num_x, sigf[k-1,:], sigs, sigt, 
                                        A, absA, B, C, D, dt, filter_func, N, 
                                        source[k-1,:, :]))
        y_prev = y[k,:, :]

    return y[-1,:,0]

def PN_state(y_prev, num_x, sigf, sigs, sigt, A, absA, B, C, D, dt, filter, N, source):

    yT = y_prev.T
    sourceT = source.T
    flux_limiter = torch.zeros([N+1,num_x])
    y = torch.zeros([N+1,num_x])
    flux_limiter[:,0] = minmod(yT[:,1] - yT[:,0], yT[:,0] - yT[:,num_x-1])
    for m in range(1, num_x - 1):
        flux_limiter[:,m] = minmod(yT[:,m+1] - yT[:,m], yT[:,m] - yT[:,m-1])
    
    flux_limiter[:,num_x - 1] = minmod(yT[:,0] - yT[:,num_x-1], yT[:,num_x-1] - yT[:,num_x-2])
    
    y[:,0] = (torch.matmul(B,yT[:,0]) + torch.matmul(C,yT[:,1]) + torch.matmul(D,yT[:,num_x-1]) -
               dt * sigf[0] * torch.matmul(torch.diag(filter[0:N+1]),yT[:,0]) -
               dt * sigt[0] * torch.matmul(torch.eye(N+1), yT[:,0]) -
               0.25 * dt * torch.matmul(A,flux_limiter[:,num_x-1] - 2 * flux_limiter[:,0] + flux_limiter[:,1]) +
               0.25 * dt * torch.matmul(absA,flux_limiter[:,1] - flux_limiter[:,num_x - 1]) 
               + dt * sourceT[:,0])
    y[0,0] += dt * sigs[0] * y_prev[0,0]

    for m in range(1, num_x -1):
        y[:,m] = (torch.matmul(B,yT[:,m]) + torch.matmul(C,yT[:,m+1]) + torch.matmul(D,yT[:,m-1]) -
                   dt * sigf[m] * torch.matmul(torch.diag(filter[0:N+1]),yT[:,m]) -
                   dt * sigt[m] * torch.matmul(torch.eye(N+1),yT[:,m]) -
                   0.25 * dt * torch.matmul(A,flux_limiter[:,m-1] - 2 * flux_limiter[:,m] + flux_limiter[:,m+1]) +
                   0.25 * dt * torch.matmul(absA,flux_limiter[:,m+1] - flux_limiter[:,m-1]) 
                   + dt * sourceT[:, m])
        y[0, m] += dt * sigs[m] * yT[0, m]
    
    y[:,num_x-1] = (torch.matmul(B,yT[:, num_x-1]) + torch.matmul(C,yT[:,0]) + torch.matmul(D,yT[:,num_x-2]) -
                       dt * sigf[num_x - 1] * torch.matmul(torch.diag(filter[0:N+1]),yT[:,num_x-1]) -
                       dt * sigt[num_x - 1] * torch.matmul(torch.eye(N+1),yT[:,num_x-1]) -
                       0.25 * dt * torch.matmul(A,flux_limiter[:,num_x-2] - 2 * flux_limiter[:,num_x-1] + flux_limiter[:, 0]) +
                       0.25 * dt * torch.matmul(absA,flux_limiter[:,0] - flux_limiter[:,num_x-2]) 
                       + dt * sourceT[:,num_x-1])
    y[0,num_x-1] += dt * sigs[num_x-1] * yT[0, num_x-1]

    y = y.T
    return y

def testing(params,mesh,funcs):
    
    num_x   = params['num_x']
    num_t   = params['num_t']
    num_mu  = params['num_mu']
    N_exact = params['N_exact']
    N       = params['N']
    glw     = mesh['glw']
    mu      = mesh['mu']
    x       = mesh['x']

    NN_model = torch.load('model_scripted.pth')
    #NN_model = torch.jit.load('model_scripted.pth')
    NN_model.eval()
    
    exact  = torch.zeros([num_t+1,num_x,N_exact+1])
    exact0,sigs,sigt,source_exact,CN = initial_data_vanishing(N_exact,num_x,num_mu,glw,mu,x)
    exact  = timestepping_state(exact0, 0, 0, params, mesh, funcs, sigs, sigt, N_exact, source_exact,0)

    y      = torch.zeros([num_t+1,num_x,N+1])
    y0, sigs, sigt,source,CN     = initial_data_vanishing(N,num_x,num_mu,glw,mu,x)

    y_PN   = timestepping_state(y0, 0, 0, params, mesh, funcs, sigs, sigt, N, source,0)
    sq_error0     = obj_func(y_PN-exact,params,mesh)

    y      = timestepping_state(y0, 1, NN_model, params, mesh, funcs, sigs, sigt, N, source,CN)

    sq_errorf  = obj_func(y -exact,params,mesh)

    error_reduction = torch.sqrt(sq_errorf/sq_error0)
    print('error_reduction = ', error_reduction)
    
    plt.figure(1)
    plt.title('N = %i' %(N))
    plt.plot(x, exact/np.sqrt(2), color='r', label='exact')
    plt.plot(x, y_PN/np.sqrt(2), color='b', label='y_PN')
    plt.plot(x, y.detach().numpy()/np.sqrt(2), color='g', label='y_FPN')  
    plt.legend()
    plt.show() 


def training(params,mesh,funcs,sigs_max):

    num_x   = params['num_x']
    num_t   = params['num_t']
    num_mu  = params['num_mu']
    N_exact = params['N_exact']
    N       = params['N']
    num_epochs  = params['num_epochs']
    batch_size  = params['batch_size']
    learning_rate = params['learning_rate']
    momentum_factor = params['momentum_factor']
    glw     = mesh['glw']
    mu      = mesh['mu']
    x       = mesh['x']
    source_exact = funcs['source_exact']
    source       = funcs['source']

    NN_model = SimpleNN(N)
    
    opt = optim.SGD(NN_model.parameters(), lr=learning_rate, momentum=momentum_factor)
    #opt = optim.Adam(NN_model.parameters(), lr=learning_rate)

    y0_ls     = initial_data_linesource(N,num_x,num_mu,glw,mu,x)
    exact0_ls = initial_data_linesource(N_exact,num_x,num_mu,glw,mu,x)
        
    y0_hs     = initial_data_heaviside(N,num_x,num_mu,glw,mu,x)
    exact0_hs = initial_data_heaviside(N_exact,num_x,num_mu,glw,mu,x) 

    y0_sin     = initial_data_sin(N,num_x,num_mu,glw,mu,x)
    exact0_sin = initial_data_sin(N_exact,num_x,num_mu,glw,mu,x) 

    for l in range(num_epochs):
        opt.zero_grad()

        sigs  = torch.rand(batch_size)*sigs_max
        sigt  = sigs
        sigs.detach().numpy
        CN    = 1/(N*sigs+1) 

        exact_ls  = torch.zeros([batch_size,num_x])
        y_ls      = torch.zeros([batch_size,num_x])
        exact_hs  = torch.zeros([batch_size,num_x])
        y_hs      = torch.zeros([batch_size,num_x])
        exact_sin  = torch.zeros([batch_size,num_x])
        y_sin      = torch.zeros([batch_size,num_x])

        for j in range(batch_size):
            exact_ls[j,:] = timestepping_state(exact0_ls, 0, 0, params, 
                mesh, funcs, sigs[j]*torch.ones(num_x), sigt[j]*torch.ones(num_x), 
                N_exact, source_exact,0)
            y_ls[j,:] = timestepping_state(y0_ls, 1, NN_model,
                 params, mesh, funcs, sigs[j]*torch.ones(num_x),sigt[j]*torch.ones(num_x),
                 N, source,CN[j]*torch.ones(num_x))
            
            exact_hs[j,:] = timestepping_state(exact0_hs, 0, 0, params, 
                mesh, funcs, sigs[j]*torch.ones(num_x), sigt[j]*torch.ones(num_x), 
                N_exact, source_exact,0)
            y_hs[j,:] = timestepping_state(y0_hs, 1, NN_model,
                 params, mesh, funcs, sigs[j]*torch.ones(num_x),sigt[j]*torch.ones(num_x),
                 N, source,CN[j]*torch.ones(num_x))
            
            exact_sin[j,:] = timestepping_state(exact0_sin, 0, 0, params, 
                mesh, funcs, sigs[j]*torch.ones(num_x), sigt[j]*torch.ones(num_x), 
                N_exact, source_exact,0)
            y_sin[j,:] = timestepping_state(y0_sin, 1, NN_model,
                 params, mesh, funcs, sigs[j]*torch.ones(num_x),sigt[j]*torch.ones(num_x),
                 N, source,CN[j]*torch.ones(num_x))

        y     = torch.cat([y_ls, y_hs,y_sin])
        exact = torch.cat([exact_ls,exact_hs,exact_sin])
        loss  = obj_func(y -exact,params,mesh)     
        
        loss.backward()
        opt.step()
        print('epoch', l)

    return NN_model

N       = 3
N_exact = 63
num_x   = 64
num_t   = num_x
num_mu  = N_exact +1 

xl      = -1
xr      =  1
T       =  0.5

filter_order = 4 

dx      = (xr-xl)/num_x
dt      = T/num_t

t       = torch.arange(0,T+dt,dt)
x       = torch.arange(xl,xr,dx)

#batch size is number of sigs values per IC per epoch
batch_size  = 5
num_epochs  = 100
learning_rate = 1e6
momentum_factor = 0.9
sigs_max  = 1

glw_glp  = np.polynomial.legendre.leggauss(num_mu)
mu       = torch.tensor(glw_glp[0],dtype = torch.float32)
glw      = torch.tensor(glw_glp[1],dtype = torch.float32)
source         = torch.zeros([num_t+1,num_x,N+1])
source_exact   = torch.zeros([num_t+1,num_x,N_exact+1])

filter = torch.zeros(N_exact+1)

filt_input        = torch.zeros(N+1)
filt_input[0:N+1] = torch.arange(0,N+1,1)/(N+1)
filter[0:N+1]     = -torch.log(filter_func(filt_input,filter_order))

params = {'num_x'    : num_x,
          'num_t'    : num_t,
          'num_mu'   : num_mu,
           'N'       : N,
           'N_exact' : N_exact,
           'num_epochs'  : num_epochs,
           'learning_rate' : learning_rate,
           'batch_size'    : batch_size,
           'momentum_factor' : momentum_factor}

mesh = {'dx'  : dx, 
        'dt'  : dt,
        'x'   : x,
        't'   : t,
        'glw' : glw,
        'mu'  : mu}

funcs = {'filter'      : filter,
         'source'      : source,
         'source_exact': source_exact,
         }

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NN_model = training(params,mesh,funcs,sigs_max)
    # NN_model = mp.spawn(training, args = (num_gpus,params,mesh,
    #          funcs,sigs_max), nprocs=num_gpus, join=True)
    torch.save(NN_model, "model_scripted.pth")   
    testing(params,mesh,funcs)