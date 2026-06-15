import torch
import torch.optim as optim
import numpy as np


def resolve_device(value=None):
    if isinstance(value, torch.device):
        return value
    if value is None or str(value).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


bias_init = 0

N = 3
N_exact = 37

# note num_x and num_y refer to cell centers:  so there will be num_x + 1 nodes in x
num_x = 100
T = 0.5

# objective functional flag
# 0 - final time scalar flux
# 1 - scalar flux in time, summed over time steps
# 2 - scalar flux in time, averaged over time steps
# 3 - scalar flux in time, averaged over time steps (1D-compatible alias)
obj_idx = 0

filter_order = 4

feature_variant = "baseline_norm"
feature_normalization = "sample"
material_feature_normalization = "none"
feature_log_scale = 1.0
feature_log_clip = [0.0, 20.0]
feature_eps = 1e-8
include_material_scale_features = False
include_material_ratios = False

# filter type
# 0 - Neural network
# 1 - Constant trained
# 2 - Constant input
filter_type = 0

# constant filter strength for filter_type = 2
sigf_const = 15

show_sym_errors = 0
show_plots = 0
show_slices = 1

xl = -1
xr = 1

yl = -1
yr = 1

num_features = 2 * (N + 1) + 2
num_hidden = num_features // 2

num_basis = (N + 1) * (N + 2) // 2
num_basis_exact = (N_exact + 1) * (N_exact + 2) // 2

num_y = num_x

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

plot_idx = int(np.round(num_x // 2))

params = {
    "num_x": num_x,
    "num_y": num_y,
    "num_t": num_t,
    "N": N,
    "N_exact": N_exact,
    "dx": dx,
    "dy": dy,
    "dt": dt,
    "x": x,
    "y": y,
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
    "filter_order": filter_order,
    "filter_type": filter_type,
    "sigf_const": sigf_const,
    "num_basis_exact": num_basis_exact,
    "plot_idx": plot_idx,
    "show_plots": show_plots,
    "show_sym_errors": show_sym_errors,
    "show_slices": show_slices,
    "obj_idx": obj_idx,
    "feature_variant": feature_variant,
    "feature_normalization": feature_normalization,
    "material_feature_normalization": material_feature_normalization,
    "feature_log_scale": feature_log_scale,
    "feature_log_clip": feature_log_clip,
    "feature_eps": feature_eps,
    "include_material_scale_features": include_material_scale_features,
    "include_material_ratios": include_material_ratios,
    "bias_init": bias_init,
}
