import torch
import torch.optim as optim
import numpy as np



N = 3
N_exact = 37

# note num_x and num_y refer to cell centers:  so there will be num_x + 1 nodes in x
num_x = 100
T = 0.5

filter_order = 4

# filter type
# 0 - Neural network
# 1 - Constant trainedd
# 2 - Constant input
filter_type = 0

#constant filter strength for filter_type = 2
sigf_const =  15

show_sym_errors = 0
show_plots = 1
show_slices = 1

xl = -1
xr = 1

yl = -1
yr = 1

num_features = 2 * (N + 1) + 2
num_hidden = N+2

num_basis = (N + 1) * (N + 2) // 2
num_basis_exact = (N_exact + 1) * (N_exact + 2) // 2

num_x_fine_factor = 1
num_y = num_x
num_y_fine_factor = num_x_fine_factor

Lx = xr - xl
Ly = yr - yl
dx = Lx / num_x
dy = Ly / num_y
dt = dx / 2
num_t = int((T + dt) // dt)

x_edges = torch.linspace(xl, xr, num_x + 1, dtype=torch.float32)
y_edges = torch.linspace(yl, yr, num_y + 1, dtype=torch.float32)
x = torch.linspace(xl + dx / 2, xr - dx / 2, num_x, dtype=torch.float32)
y = torch.linspace(yl + dy / 2, yr - dy / 2, num_y, dtype=torch.float32)

num_x_fine = num_x * num_x_fine_factor + 1
num_y_fine = num_y * num_y_fine_factor + 1
x_fine = torch.linspace(xl, xr, num_x_fine, dtype=torch.float32)
y_fine = torch.linspace(yl, yr, num_y_fine, dtype=torch.float32)

plot_idx = int(np.round(num_x // 2))

params = {
    "num_x": num_x,
    "num_y": num_y,
    "num_x_fine_factor": num_x_fine_factor,
    "num_y_fine_factor": num_y_fine_factor,
    "num_x_fine": num_x_fine,
    "num_y_fine": num_y_fine,
    "num_t": num_t,
    "N": N,
    "N_exact": N_exact,
    "dx": dx,
    "dy": dy,
    "dt": dt,
    "x": x,
    "y": y,
    "x_fine": x_fine,
    "y_fine": y_fine,
    "x_edges": x_edges,
    "y_edges": y_edges,
    "xl": xl,
    "xr": xr,
    "T": T,
    "Lx": Lx,
    "Ly": Ly,
    "num_features": num_features,
    "num_hidden": num_hidden,
    "num_basis": num_basis,
    "filter": filter,
    "filter_order": filter_order,
    "filter_type": filter_type,
     "sigf_const" : sigf_const,
    "num_basis_exact": num_basis_exact,
    "plot_idx": plot_idx,
    "show_plots": show_plots,
    "show_sym_errors": show_sym_errors,
    "show_slices": show_slices
}
