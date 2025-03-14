import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def frame_source(num_x, num_y, x, y):
    source = torch.zeros([num_y+1,num_x+1])

    source_x_min, source_x_max = 0.6, 0.8
    source_y_min, source_y_max = 0.6, 0.8
    
    for l in range(num_y+1):
        for m in range(num_x+1):
            
            if (
                (source_x_min <= torch.abs(x[m]) <= source_x_max and torch.abs(y[l]) <= source_y_max) or
                (source_y_min <= torch.abs(y[l]) <= source_y_max and  torch.abs(x[m]) <= source_x_max)
            ):
                source[l, m] = 5 

    return source

def two_rect_source(num_x, num_y, x, y):
    source = torch.zeros([num_y+1,num_x+1])


    c1 = 20
    c2 = 40
    source_x_min, source_x_max   = -0.25, 0.25
    source_y1_min, source_y1_max = 0.7, 0.8
    source_y2_min, source_y2_max = -0.8,-0.7
    
    for l in range(num_y+1):
        for m in range(num_x+1):
            
            if (
                (source_x_min <= x[m] <= source_x_max and  source_y1_min <= y[l] <= source_y1_max ) 
            ):
                source[l, m] = c1

            if (
                (source_x_min <= x[m] <= source_x_max and  source_y2_min <= y[l] <= source_y2_max ) 
            ):
                source[l, m] = c2

    return source


def gaussian_source(num_x,num_y,x,y):
    cx    = 0
    cy    = 0
    s     = 0.05
    source = torch.zeros([num_y+1,num_x+1],dtype = torch.float32)
    scale = 1/(s*torch.sqrt(torch.tensor(2*np.pi)))
    for l in range(0,num_y+1):
        for m in range(0,num_x+1):
            source[l,m] = scale*torch.exp(-(((x[m]-cx)**2) + (y[l]-cy)**2)/(2*s**2))
    

    return source

def pulse_source(num_x,num_y,x,y):

    source = torch.zeros([num_y+1,num_x+1])
    c = 10
    rad = 0.1
    for l in range(num_y+1):
        for m in range(num_x+1):
            r = torch.sqrt(x[m]**2 + y[l]**2)  
            if r < rad:
                source[l, m] = c

    # psi0_np = source.numpy()
    # fig, axs = plt.subplots(1, 1, figsize=(4.5, 4.5), constrained_layout=True)
    # contour1 = axs.contourf(y, x, psi0_np, levels = 100)
    # fig.colorbar(contour1, ax=axs, orientation='vertical', shrink=0.8)
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.show()

    return source


