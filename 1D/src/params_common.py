import re

import torch
import torch.nn as nn
import torch.optim as optim


def resolve_device(value=None):
    if isinstance(value, torch.device):
        return value
    if value is None or str(value).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def model_tag_from_params(params_or_value=None):
    if isinstance(params_or_value, dict):
        value = (
            params_or_value.get("model_tag")
            or params_or_value.get("training_data")
            or params_or_value.get("training_setup")
        )
    else:
        value = params_or_value

    tag = str(value or "paper").lower()
    tag = re.sub(r"[^a-z0-9_-]+", "-", tag).strip("-")
    return tag or "paper"


def tagged_model_path(N, model_idx, filter_type, model_tag=None):
    tag = model_tag_from_params(model_tag)
    if filter_type in (1, 2):
        return f"trained_models/model_{tag}_N{N}_{model_idx}.pth"
    if filter_type == 3:
        return f"trained_models_const/model_{tag}_N{N}_{model_idx}.pth"
    raise ValueError(f"Unsupported filter_type: {filter_type}")


N = 3
N_exact = 127
num_x = 128
T = 0.5
# num_x = 128*4
# T = 5

# show plot = 1 -> plot
show_plot = 0

method_order = 2

# filter type
# 0 - Off
# 1 -  abs on every input
# 2 - Alternating abs on moments
# 3 - Constant
filter_type = 1

# objective functional index
# 0 - scalar flux at final time
# 1 - all moments at final time
# 2 - scalar flux in time, summed over time steps
# 3 - scalar flux in time, averaged over time steps
obj_idx = 0

xl = -1
xr = 1

filter_order = 4

feature_variant = "baseline_norm"
feature_normalization = "sample"
material_feature_normalization = "none"
feature_log_scale = 1.0
feature_log_clip = [0.0, 20.0]
feature_eps = 1e-8
include_material_scale_features = False
include_material_ratios = False

L = xr - xl
dx = L / num_x

dt = dx / 2
num_t = int((T + dt) // dt)

x_edges = torch.linspace(xl, xr, num_x + 1)
x = torch.linspace(xl + dx / 2, xr - dx / 2, num_x)

if filter_type in (1, 2):
    num_features = 2 * N + 4
    num_hidden = 50
    weight_decay = 1e-5
elif filter_type == 3:
    num_hidden = 0
    num_features = 0
    weight_decay = 1e-6

params = {
    "num_x": num_x,
    "num_t": num_t,
    "N": N,
    "N_exact": N_exact,
    "num_features": num_features,
    "num_hidden": num_hidden,
    "dx": dx,
    "dt": dt,
    "x": x,
    "xl": xl,
    "xr": xr,
    "x_edges": x_edges,
    "L": L,
    "T": T,
    "filter_type": filter_type,
    "weight_decay": weight_decay,
    "show_plot": show_plot,
    "filter_order": filter_order,
    "method_order": method_order,
    "obj_idx": obj_idx,
    "feature_variant": feature_variant,
    "feature_normalization": feature_normalization,
    "material_feature_normalization": material_feature_normalization,
    "feature_log_scale": feature_log_scale,
    "feature_log_clip": feature_log_clip,
    "feature_eps": feature_eps,
    "include_material_scale_features": include_material_scale_features,
    "include_material_ratios": include_material_ratios,
}
