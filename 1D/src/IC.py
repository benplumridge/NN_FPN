import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def gaussian_training(num_x, x):
    z = torch.zeros([num_x + 1])
    c = 0
    s = 0.10
    sigs = 0.5 * torch.ones(num_x + 1)
    source = torch.zeros([num_x + 1])

    scale = 1 / np.sqrt(2 * np.pi * s**2)
    z = scale * torch.exp(-((x - c) ** 2) / (2 * s**2))

    sigt = sigs

    return z, sigs, sigt, source

def gaussian_testing(num_x, x):
    z = torch.zeros([num_x + 1])
    c = 0
    s = 0.05
    sigs = 0.5 * torch.ones(num_x + 1)
    source = torch.zeros([num_x + 1])

    scale = 1 / np.sqrt(2 * np.pi * s**2)
    z = scale * torch.exp(-((x - c) ** 2) / (2 * s**2))

    sigt = sigs

    return z, sigs, sigt, source

def heaviside(num_x, x):
    z = torch.zeros([num_x + 1])
    sigs = torch.zeros(num_x + 1)
    source = torch.zeros([num_x + 1])

    z = torch.zeros([num_x + 1])
    z[(x > -0.2) & (x < 0.2)] = 1
    sigt = sigs

    return z, sigs, sigt, source


def bump(num_x, x):
    z = torch.zeros([num_x + 1])
    sigs = torch.zeros(num_x + 1)
    source = torch.zeros([num_x + 1])

    z = torch.zeros([num_x + 1])
    abs_x = torch.abs(x)
    z[abs_x < 0.5] = torch.cos(torch.pi * x[abs_x < 0.5])
    sigt = sigs

    return z, sigs, sigt, source


def disc_cs(num_x, x):
    z = torch.zeros([num_x + 1])
    sigs = torch.ones(num_x + 1)
    source = torch.zeros([num_x + 1])
    z[(x > -0.2) & (x < 0.2)] = 1
    sigs[((x >= -0.65) & (x <= -0.35)) | ((x >= 0.35) & (x <= 0.65))] = 0.02
    sigt = sigs

    return z, sigs, sigt, source


def disc_source(num_x, x):
    z = torch.zeros(num_x + 1)
    source = torch.zeros(num_x + 1)
    sigs = torch.zeros(num_x + 1)
    abs_x = torch.abs(x)
    source[abs_x > 0.75] = 2
    source[abs_x < 0.5] = 1
    z[abs_x < 0.25] = 1
    sigt = sigs
    return z, sigs, sigt, source

    # plt.rcParams.update({'font.size': 16})
    # plt.figure(1)
    # plt.plot(x, z[:,0], color='b', label='y0' )
    # plt.plot(x, source[0,:,0],color='r', label = 'source',linestyle="dashed")
    # plt.xlim([-1,1])
    # plt.show()


def vanishing_cs(num_x, x):
    z = torch.zeros(num_x + 1)
    sigs = torch.zeros(num_x + 1)
    source = torch.zeros([num_x + 1])
    sigs = 100 * x**4
    sigt = sigs

    z[(x > -0.2) & (x < 0.2)] = 100
    return z, sigs, sigt, source


def reeds(params):
    num_x = params["num_x"]
    T = params["T"]

    xl = 0
    xr = 8

    L = xr - xl
    dx = L / num_x
    dt = dx / 2
    num_t = int((T + dt) // dt)

    x_edges = torch.linspace(xl, xr, num_x + 1)
    x = torch.linspace(xl + dx / 2, xr - dx / 2, num_x)

    sigs = torch.zeros(num_x + 1)
    sigt = torch.zeros(num_x + 1)
    source = torch.zeros([num_x + 1])
    z = torch.zeros([num_x + 1])

    reg1 = np.arange(num_x / 4)
    reg2 = np.arange(num_x / 4, 3 * num_x / 8)
    reg3 = np.arange(3 * num_x / 8, 5 * num_x / 8)
    reg4 = np.arange(5 * num_x / 8, 3 * num_x / 4)
    reg5 = np.arange(3 * num_x / 4, num_x + 1)

    sigs[reg1] = 0
    sigt[reg1] = 50
    sigs[reg2] = 0
    sigt[reg2] = 5
    sigs[reg3] = 0
    sigt[reg3] = 0
    sigs[reg4] = 0.9
    sigt[reg4] = 1
    sigs[reg5] = 0.9
    sigt[reg5] = 1

    source[reg1] = 50
    source[reg4] = 1

    params["xl"] = xl
    params["xr"] = xr
    params["L"] = L
    params["x"] = x
    params["dx"] = dx
    params["dt"] = dt
    params["x_edges"] = x_edges
    params["num_t"] = num_t

    return z, sigs, sigt, source, params
