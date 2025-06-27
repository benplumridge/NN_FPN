import torch
import torch.nn as nn
import torch.optim as optim

def filter_func(z,p):
    return torch.exp(-z**p)

# N = 3: 27.1199
# N = 7: 16.1425
# N = 9: 10.2298

N       = 9
N_exact = 127
num_x   = 128
T       = 1

# filter type 
# 0 - Neural network 
# 1 - Constant
filter_type = 0

xl      = -1
xr      =  1


filter_order = 4

L       = xr - xl
dx      = L/num_x

dt      = dx/2
num_t   = int((T+dt)//dt) 

x_edges = torch.linspace(xl,xr,num_x + 1)
x       = torch.linspace(xl+dx/2, xr- dx/2, num_x)

num_features = 2*N+4
num_hidden   = N+2

filter = torch.zeros(N_exact+1)

filt_input        = torch.zeros(N+1)
filt_input[0:N+1] = torch.arange(0,N+1,1)/(N+1)
filter[0:N+1]     = -torch.log(filter_func(filt_input,filter_order))

params = {'num_x'      : num_x,
        'num_t'        : num_t,
        'N'            : N,
        'N_exact'      : N_exact,
        'num_features' : num_features,
        'num_hidden'   : num_hidden,
        'filter'       : filter,
        'dx'           : dx, 
        'dt'           : dt,
        'x'            : x,
        'xl'           : xl,
        'xr'           : xr,
        'x_edges'      : x_edges,
        'L'            : L,
        'T'            : T,
        'filter_type'  : filter_type}



