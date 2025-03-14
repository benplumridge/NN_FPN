import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def gaussian(num_x,num_y,x,y):
    #isotropic initial conditions: Don't need to include angles in initial data 
    psi0  = torch.zeros([num_y+1,num_x+1],dtype = torch.float32)
    cx    = 0
    cy    = 0
    #s     = 0.1 #Training value
    s     = 0.03
    sigs   = torch.ones([num_y+1,num_x+1],dtype = torch.float32)
    source = torch.zeros([num_y+1,num_x+1],dtype = torch.float32)
    scale = 1/(s*torch.sqrt(torch.tensor(2*np.pi)))
    for l in range(0,num_y+1):
        for m in range(0,num_x+1):
            psi0[l,m] = scale*torch.exp(-(((x[m]-cx)**2) + (y[l]-cy)**2)/(2*s**2))
    
    sigt  = sigs

    plt.show()


    return psi0, source, sigs, sigt

def step(num_x, num_y, x, y):


    psi0   = torch.zeros([num_y+1,num_x+1])
    sigs   = torch.ones([num_y+1,num_x+1])
    source = torch.zeros([num_y+1,num_x+1])

    x_min, x_max = -0.2, 0.2
    y_min, y_max = -0.2, 0.2
    
    for l in range(num_y+1):
        for m in range(num_x+1):
            if x_min <= x[m] <= x_max and y_min <= y[l] <= y_max:
                psi0[l, m] = 20

    sigt = sigs  
    return psi0, source, sigs, sigt

def disc_source(num_x, num_y, x, y):

    psi0 = torch.zeros([num_y+1,num_x+1])
    sigs = torch.ones([num_y+1,num_x+1])
    source = torch.zeros([num_y+1,num_x+1])

    step_x_min, step_x_max = -0.2, 0.2
    step_y_min, step_y_max = -0.2, 0.2
    
    source_x_min, source_x_max = 0.6, 0.8
    source_y_min, source_y_max = 0.6, 0.8
    
    for l in range(num_y+1):
        for m in range(num_x+1):
            if step_x_min <= x[m] <= step_x_max and step_y_min <= y[l] <= step_y_max:
                psi0[l, m] = 10 
                
            if (
                (source_x_min <= torch.abs(x[m]) <= source_x_max and torch.abs(y[l]) <= source_y_max) or
                (source_y_min <= torch.abs(y[l]) <= source_y_max and  torch.abs(x[m]) <= source_x_max)
            ):
                source[l, m] = 5 

    sigt = sigs  

    return psi0, source, sigs, sigt


def bump(num_x, num_y, x, y):

    sigs = torch.ones([num_y+1,num_x+1])
    sigt = sigs
    psi0 = torch.zeros([num_y+1,num_x+1])
    source = torch.zeros([num_y+1,num_x+1])

    sx = 0.05
    sy = sx

    scale = 10

    for l in range(num_y+1):
        for m in range(num_x+1):
            r = 2*torch.pi*torch.sqrt(x[m]**2 + y[l]**2)  
            cosr = torch.cos(r)
            if cosr > 0 and r < torch.pi*3/4:
                psi0[l, m] = scale*cosr

    return psi0, source, sigs, sigt


def hat(num_x,num_y,x,y):
    #isotropic initial conditions: Don't need to include angles in initial data 
    psi0  = torch.zeros([num_y+1,num_x+1])

    sigs   = torch.zeros([num_y+1,num_x+1])
    source = torch.zeros([num_y+1,num_x+1])

    
    for l in range(num_y+1):
        for m in range(num_x+1):
            psi0[l, m] = torch.max(torch.tensor(0.0), 1 - torch.abs(x[m]) - torch.abs(y[l]))
    
    sigt  = sigs

    return psi0, source, sigs, sigt

def holhraum(params):

    num_x = 130
    num_y = 130
    sigt_wall  = 100
    sigs_wall  = 50
    sigt_mat  = 100
    sigs_mat  = 0
    sigt_vac  = 0.1
    sigs_vac  = 0.1
    sigt_source  = 100
    sigs_source  = 95
    sigs_center_plate = 90
    source_boundary  = 1

    T = params['T']
    xl = 0
    xr = 1.3
    yl = 0 
    yr = 1.3
    Lx = xr - xl
    Ly = yr - yl
    dx      = Lx/num_x
    dy      = Ly/num_y
    dt      = dx/2
    num_t   = int((T+dt)//dt) 
    x_edges     = torch.linspace(xl,xr,num_x + 1, dtype=torch.float32)
    y_edges     = torch.linspace(yl,yr,num_y + 1, dtype=torch.float32)
    x           = torch.linspace(xl+dx/2, xr-dx/2, num_x, dtype=torch.float32)
    y           = torch.linspace(yl+dy/2, yr-dy/2, num_y, dtype=torch.float32)

    psi0   = torch.zeros([num_y+1, num_x+1])
    sigs   = sigt_vac*torch.ones([num_y+1, num_x+1])
    sigt   = sigs_vac*torch.ones([num_y+1, num_x+1])
    source = torch.zeros([num_y+1, num_x+1])

    y_bottom        = int(0.25*num_y//1.3)+1
    y_top           = int(1.05*num_y//1.3)+2
    x_left          = int(0.45*num_x//1.3)+1
    x_right         = int(0.85*num_x//1.3)+2
    y_slice         = slice(y_bottom, y_top)
    x_slice         = slice(x_left, x_right)
    wall_thickness  = int(0.05//dx)
    sigt[y_slice,x_slice]     = sigt_mat
  
    sigt[:wall_thickness, :]  = sigt_wall
    sigt[-wall_thickness:, :] = sigt_wall
    sigt[y_slice, :wall_thickness]  = sigt_source
    sigt[:, -wall_thickness:] = sigt_wall    

    sigs[y_slice,x_slice]     = sigs_mat
    sigs[:wall_thickness, :]  = sigs_wall
    sigs[-wall_thickness:, :] = sigs_wall
    sigs[y_slice, :wall_thickness]  = sigs_source
    sigs[:, -wall_thickness:] = sigs_wall

    sigs[y_bottom:y_bottom+wall_thickness, x_slice] = sigs_center_plate
    sigs[y_top-wall_thickness:y_top, x_slice]       = sigs_center_plate
    sigs[y_slice,x_left:x_left+wall_thickness]      = sigs_center_plate

    #source[y_slice,:wall_thickness] = source_boundary
                
    # sigt_np = sigt.numpy()
    # sigs_np = sigs.numpy()
    # source_np = source.numpy()

    # sym_data = source_np
    # source_sym = sym_data - np.flip(sym_data,axis = 0)
    # print(np.abs(np.max(source_sym)))
    # print(np.argmax(np.abs(np.max(source_sym))))
    # print(source_sym)

    # #Create a contour plot

    # fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.5), constrained_layout=True)
    
    # contour1 = axs[0].contourf(y_edges,x_edges, sigt_np, levels=20, cmap='viridis')
    # axs[0].set_title('sigt')
    # plt.xlabel('x')
    # plt.ylabel('y')

    # contour2 =  axs[1].contourf(y_edges,x_edges, sigs_np, levels=20, cmap='viridis')
    # axs[1].set_title('sigs')
    # plt.xlabel('x')
    # plt.ylabel('y')

    # contour3 =  axs[2].contourf(y_edges,x_edges, source_np, levels=20, cmap='viridis')
    # axs[2].set_title('source')
    # plt.xlabel('x')
    # plt.ylabel('y')

    # fig.colorbar(contour1, ax=axs, orientation='vertical', shrink=0.8)
    # plt.show()
    

    params['xl'] = xl
    params['xr'] = xr
    params['yl'] = yl
    params['yr'] = yr
    params['Lx']   = Lx
    params['Ly']   = Ly
    params['x']    = x
    params['y']    = y
    params['dx']   = dx
    params['dy']   = dy
    params['dt']   = dt
    params['x_edges']  = x_edges
    params['y_edges']  = y_edges
    params['num_t'] = num_t
    params['num_x'] = num_x
    params['num_y'] = num_y
    params['source_boundary'] = source_boundary

    return psi0, source, sigs, sigt, params