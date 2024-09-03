import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from numpy import asarray
from numpy import savetxt

class SimpleNN(nn.Module):
    def __init__(self,N,num_features):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(num_features, N+1)  # (inputs,hidden)
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

def obj_func(z):
    obj_value = 0.5*dx*torch.mean(z**2)
    print(obj_value)
    return obj_value


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

    for n in range(0,num_mu):    
        for m in range(0,num_x):         
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

    B = B.to(device)
    C = C.to(device)
    D = D.to(device)
    source = source.to(device)
    filter_func = filter_func.to(device)
    A = A .to(device)
    absA = absA.to(device)
    sigt = sigt.to(device)
    sigs = sigs.to(device)

    y  = torch.zeros([batch_size,num_x,N+1],device=device)
    Dy = torch.zeros([batch_size,num_x,N+1],device=device)

    y_prev = y0
    y  = y0

    d = torch.zeros(num_x-1)
    d[1:num_x-1]  = 0.5/dx
    Der     = torch.diag(d,1) - torch.diag(d,-1)
    Der     = Der.to(device)
    
    sigf = torch.zeros([batch_size,num_x],device=device)
    for k in range(1, num_t + 1):
        for j in range(batch_size):     
            if filt_switch == 1:
                for n  in range(0,N+1):
                    Dy[j,:,n] = torch.matmul(Der,y[j,:,n])
                      
                for m in range(0,num_x):
                    model_inputs = torch.concatenate((y[j,m,:]/N,CN[j]*Dy[j,m,:]))
                    sigf[j,m] = NN_model(model_inputs)
            
        y1 = PN_solve(y_prev, num_x, sigf, sigs, sigt, A, absA, B, C, D, dt, 
                      filter_func, N, source[k-1,:,:])
        y = 0.5*(y1 + PN_solve( y1, num_x, sigf, sigs, sigt, A, absA, B, C, D, dt, 
                               filter_func, N, source[k-1,:,:]))
        y_prev = y

    return y[:,:,0]

def PN_solve(y_prev, num_x, sigf, sigs, sigt, A, absA, B, C, D, dt, filter, N, source):

    flux_limiter = torch.zeros([batch_size,num_x+2,N+1],device=device)
    y = torch.zeros([batch_size,num_x,N+1],device=device)
    y_expand = torch.zeros([batch_size,num_x+2,N+1],device=device)
    y_expand[:,0,:] = y_prev[:,num_x-1,:]
    y_expand[:,1:num_x+1,:] = y_prev
    y_expand[:,num_x+1,:] = y_prev[:,0,:]

    flux_limiter[:,1:num_x+1,:] = minmod(y_expand[:,2:num_x+2,:] - y_expand[:,1:num_x+1,:], 
                y_expand[:,1:num_x+1,:] - y_expand[:,0:num_x,:])
    flux_limiter[:,0,:] = flux_limiter[:,num_x+1,:]
    flux_limiter[:,num_x+1,:] = flux_limiter[:,0,:]

    B_y = torch.matmul(y_expand[:, 1:num_x+1, :], B.T) 
    C_y = torch.matmul(y_expand[:, 2:num_x+2, :], C.T)  
    D_y = torch.matmul(y_expand[:, :num_x, :], D.T)     

    sigf_y = dt * sigf[:, :, None] * filter[None, None, :N+1] * y_expand[:, 1:num_x+1, :] 
    sigt_y = dt * sigt[:, None, None] * y_expand[:, 1:num_x+1, :]                         

    flux_diff1 = flux_limiter[:, :num_x, :] - 2 * flux_limiter[:, 1:num_x+1, :] + flux_limiter[:, 2:num_x+2, :]
    flux_diff2 = flux_limiter[:, 2:num_x+2, :] - flux_limiter[:, :num_x, :]
    flux1 = 0.25 * dt * torch.matmul(flux_diff1, A.T)    
    flux2 = 0.25 * dt * torch.matmul(flux_diff2, absA.T)   

    y[:,0:num_x,:] = (B_y + C_y + D_y - sigf_y - sigt_y - flux1 + flux2 +
    dt * source[None, :num_x, :])

    y[:, :, 0] += dt * sigs[:, None] * y_expand[:, 1:num_x+1, 0]

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
    NN_model.eval()

    exact0,sigs,sigt,source_exact,CN = initial_data_vanishing(N_exact,num_x,num_mu,glw,mu,x)
    exact  = timestepping_state(exact0, 0, 0, params, mesh, funcs, sigs, sigt, N_exact, source_exact,0)

    y0, sigs, sigt,source,CN     = initial_data_vanishing(N,num_x,num_mu,glw,mu,x)

    y_PN   = timestepping_state(y0, 0, 0, params, mesh, funcs, sigs, sigt, N, source,0)
    sq_error0 = obj_func(y_PN -exact)

    yf      = timestepping_state(y0, 1, NN_model, params, mesh, funcs, sigs, sigt, N, source,CN)
    sq_errorf = obj_func(yf -exact)
 
    error_reduction = torch.sqrt(sq_errorf/sq_error0)
    print('error_reduction = ', error_reduction)
    
    plt.figure(1)
    plt.title('N = %i' %(N))
    plt.plot(x, exact/np.sqrt(2), color='r', label='exact')
    plt.plot(x, y_PN/np.sqrt(2), color='b', label='y_PN')
    plt.plot(x, yf.detach().numpy()/np.sqrt(2), color='g', label='y_FPN')  
    plt.legend()
    plt.show() 

def training(device,params,mesh,funcs,sigs_max):

    num_x   = params['num_x']
    num_t   = params['num_t']
    num_mu  = params['num_mu']
    N_exact = params['N_exact']
    N       = params['N']
    num_epochs  = params['num_epochs']
    batch_size  = params['batch_size']
    learning_rate = params['learning_rate']
    momentum_factor = params['momentum_factor']
    num_features = params['num_features']
    glw     = mesh['glw']
    mu      = mesh['mu']
    x       = mesh['x']
    source_exact = funcs['source_exact']
    source       = funcs['source']

    NN_model = SimpleNN(N,num_features)
    opt = optim.SGD(NN_model.parameters(), lr=learning_rate, momentum=momentum_factor)
    #opt = optim.Adam(NN_model.parameters(), lr=learning_rate)
    NN_model = NN_model.to(device)
    num_IC = 3
    y0 = torch.zeros([batch_size,num_x,N+1], device=device)
    exact0 = torch.zeros([batch_size,num_x,N_exact+1], device=device)
    y0[0:round(batch_size/num_IC),:,:]     = initial_data_linesource(N,num_x,num_mu,glw,mu,x)
    exact0[0:round(batch_size/num_IC),:,:] = initial_data_linesource(N_exact,num_x,num_mu,glw,mu,x)
        
    y0[round(batch_size/num_IC)+1:round(2*batch_size/num_IC),:,:]     = initial_data_heaviside(N,num_x,num_mu,glw,mu,x)
    exact0[round(batch_size/num_IC)+1:round(2*batch_size/num_IC),:,:] = initial_data_heaviside(N_exact,num_x,num_mu,glw,mu,x) 

    y0[round(2*batch_size/num_IC)+1:round(3*batch_size/num_IC),:,:]     = initial_data_sin(N,num_x,num_mu,glw,mu,x)
    exact0[round(2*batch_size/num_IC)+1:round(3*batch_size/num_IC),:,:] = initial_data_sin(N_exact,num_x,num_mu,glw,mu,x) 

    for l in range(num_epochs):
        opt.zero_grad()

        sigs  = torch.rand(batch_size)*sigs_max
        sigt  = sigs
        sigs.detach().numpy
        CN    = 1/(N*sigs+1) 

        exact = timestepping_state(exact0, 0, 0, params, mesh, funcs, 
                sigs, sigt, N_exact, source_exact,0)
        y  = timestepping_state(y0, 1, NN_model, params, mesh, funcs,
                sigs,sigt,N, source,CN)
  
        y  = y.to('cpu')
        exact = exact.to('cpu')
        loss = obj_func(y -exact)
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


num_features = 2*N+2

batch_size  = 60  ## make batch size a multiple of the number of Initial Conditions
num_epochs  = 200
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
           'momentum_factor' : momentum_factor,
           'num_features'  : num_features}

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NN_model = training(device,params,mesh,funcs,sigs_max)
torch.save(NN_model, "model_scripted.pth")   


