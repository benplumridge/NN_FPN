import torch
import torch.nn as nn
import numpy as np


_SOLVER_TENSOR_CACHE = {}
FEATURE_VARIANTS = {
    "baseline_norm",
    "log_norm",
    "baseline_plus_log",
    "log_material_only",
    "no_norm_log",
}
FEATURE_VARIANT_ALIASES = {
    "baseline": "baseline_norm",
    "current": "baseline_norm",
    "paper": "baseline_norm",
    "log": "log_norm",
    "baseline+log": "baseline_plus_log",
    "plus_log": "baseline_plus_log",
    "material_log": "log_material_only",
    "material_only": "log_material_only",
}


def feature_variant(params=None):
    params = params or {}
    value = str(params.get("feature_variant", "baseline_norm")).lower()
    value = FEATURE_VARIANT_ALIASES.get(value, value)
    if value not in FEATURE_VARIANTS:
        raise ValueError(
            f"Unsupported feature_variant={value!r}. "
            f"Expected one of: {', '.join(sorted(FEATURE_VARIANTS))}."
        )
    return value


def _material_features_enabled(params):
    variant = feature_variant(params)
    return bool(
        variant == "log_material_only"
        or params.get("include_material_scale_features", False)
        or params.get("include_material_features", False)
        or params.get("include_material_ratios", False)
    )


def _material_feature_count(params):
    if not _material_features_enabled(params):
        return 0
    return 3 + (2 if params.get("include_material_ratios", False) else 0)


def nn_feature_count(N, params=None):
    params = params or {}
    base_count = 2 * (int(N) + 1) + 2
    variant = feature_variant(params)
    material_count = _material_feature_count(params)

    if variant in {"baseline_norm", "log_norm", "no_norm_log"}:
        return base_count + material_count
    if variant == "baseline_plus_log":
        return 2 * base_count + material_count
    if variant == "log_material_only":
        return base_count + material_count
    raise AssertionError(f"Unhandled feature_variant={variant!r}")


def _device_cache_key(device):
    device = torch.device(device)
    return device.type, device.index


class SimpleNN_const(nn.Module):
    def __init__(self):
        super().__init__()
        self.const = nn.Parameter(0.01 * torch.rand(()))

    def forward(self):
        return self.const


class SimpleNN(nn.Module):
    def __init__(self, num_features, num_hidden):
        super(SimpleNN, self).__init__()
        self.input_norm = nn.BatchNorm1d(num_features)
        self.hidden1 = nn.Linear(num_features, num_hidden)  # (inputs,hidden)
        self.hidden2 = nn.Linear(num_hidden, num_hidden)  # (inputs,hidden)
        self.hidden3 = nn.Linear(num_hidden, num_hidden)  # (inputs,hidden)
        self.hidden4 = nn.Linear(num_hidden, num_hidden)  # (inputs,hidden)

        self.bn2 = nn.LayerNorm(num_hidden)
        self.bn3 = nn.LayerNorm(num_hidden)
        self.bn4 = nn.LayerNorm(num_hidden)
        self.bn5 = nn.LayerNorm(num_hidden)
        self.output = nn.Linear(num_hidden, 1)  # (hidden,output)
        # print(self)

    def forward(self, x):
        # print("Input shape:", x.shape)  # Debugging line
        original_shape = x.shape
        x = torch.flatten(x, start_dim=0, end_dim=2)
        # print("Flattened input shape:", x.shape)  # Debugging line
        x = self.input_norm(x)
        x = torch.tanh(self.hidden1(x))  # Activation hidden layer
        x = self.bn2(x)
        x = torch.tanh(self.hidden2(x)) + x  # Activation hidden layer
        x = self.bn3(x)
        x = torch.tanh(self.hidden3(x)) + x  # Activation hidden layer
        x = self.bn4(x)
        x = torch.tanh(self.hidden4(x)) + x  # Activation hidden layer
        x = self.bn5(x)
        x = torch.relu(self.output(x))  # Activation output layer
        output_shape = [original_shape[0], original_shape[1], original_shape[2], 1]
        return x.reshape(output_shape)


def obj_func(z):
    return torch.mean(z**2)


def obj_func_time(z):
    dims = tuple(i for i in range(z.ndim) if i != 1)
    return torch.sum(torch.mean(z**2, dim=dims))


def obj_func_time_average(z):
    dims = tuple(i for i in range(z.ndim) if i != 1)
    return torch.mean(torch.mean(z**2, dim=dims))


def minmod(a, b):
    return 0.5 * (torch.sign(a) + torch.sign(b)) * torch.min(torch.abs(a), torch.abs(b))


def filter_func(z, p):
    return torch.exp(-(z**p))


def filter_coefficients(filter_order, N, num_basis, device=None):
    filter = torch.zeros(N + 1, device=device)
    filter[1 : N + 1] = -torch.log(
        filter_func(torch.arange(1, N + 1, device=device) / (N + 1), filter_order)
    )

    filter_expand = torch.zeros(num_basis, device=device)
    idx = 0
    for l in range(1, N + 2):
        filter_expand[idx : idx + l] = filter[l - 1]
        idx += l
    return filter_expand


def compute_PN_matrices(N, device=None):
    n_sys = (N + 1) * (N + 2) // 2

    # Initialize Mx, My as sparse matrices
    Ax = torch.zeros((n_sys, n_sys), dtype=torch.float32, device=device)
    Ay = torch.zeros((n_sys, n_sys), dtype=torch.float32, device=device)
    sqrt2 = torch.sqrt(torch.tensor(2, dtype=torch.float32, device=device))

    # Loop through values of m
    for m in range(1, N + 1):
        i = torch.arange(1, m + 1, device=device)
        p = (m * (m - 1)) // 2 + i
        v = d_param(m, -m + 2 * (torch.ceil(i / 2) - 1))
        Ax[p - 1, p + m - 1] = v
        Ay[p - 1, p + m - 1 - (-1) ** i] = -((-1) ** i) * v

        i = torch.arange(1, m, device=device)  # m - 1
        p = (m * (m - 1)) // 2 + i
        v = f_param(m, -m + 2 + 2 * (torch.ceil(i / 2) - 1))
        Ax[p - 1, p + m + 1] = -v
        Ay[p - 1 - (-1) ** i, p + m + 1] = (-1) ** i * v

    # Apply sqrt(2) scaling to appropriate indices
    m = torch.arange(1, N + 1, 2, device=device)
    i = (m * (m + 1)) // 2
    Ax[i - 1, :] *= sqrt2
    Ay[i - 1, :] *= sqrt2

    m = torch.arange(2, N + 1, 2, device=device)
    i = ((m + 1) * (m + 2)) // 2
    Ax[:, i - 1] *= sqrt2
    Ay[:, i - 1] *= sqrt2

    # Symmetrize matrices
    Ax = (Ax + Ax.T) / 2
    Ay = (Ay + Ay.T) / 2

    return Ax, Ay


def d_param(l, k):
    return torch.sqrt(((l - k) * (l - k - 1)) / ((2 * l + 1) * (2 * l - 1)))


def f_param(l, k):
    return torch.sqrt(((l + k) * (l + k - 1)) / ((2 * l + 1) * (2 * l - 1)))


def compute_upwind_matrices(N, device):
    Ax, Ay = compute_PN_matrices(N, device=device)

    eig_Ax, Vx = torch.linalg.eigh(Ax)
    Ax_plus = torch.matmul(
        torch.matmul(Vx, torch.diag(torch.clamp(eig_Ax, min=0))), Vx.T
    )
    Ax_minus = torch.matmul(
        torch.matmul(Vx, torch.diag(torch.clamp(eig_Ax, max=0))), Vx.T
    )

    eig_Ay, Vy = torch.linalg.eigh(Ay)
    Ay_plus = torch.matmul(
        torch.matmul(Vy, torch.diag(torch.clamp(eig_Ay, min=0))), Vy.T
    )
    Ay_minus = torch.matmul(
        torch.matmul(Vy, torch.diag(torch.clamp(eig_Ay, max=0))), Vy.T
    )

    threshold = 1e-6
    Ax_plus = torch.where(
        torch.abs(Ax_plus) < threshold, torch.zeros_like(Ax_plus), Ax_plus
    )
    Ax_minus = torch.where(
        torch.abs(Ax_minus) < threshold, torch.zeros_like(Ax_minus), Ax_minus
    )
    Ay_plus = torch.where(
        torch.abs(Ay_plus) < threshold, torch.zeros_like(Ay_plus), Ay_plus
    )
    Ay_minus = torch.where(
        torch.abs(Ay_minus) < threshold, torch.zeros_like(Ay_minus), Ay_minus
    )

    return Ax, Ay, Ax_plus, Ax_minus, Ay_plus, Ay_minus


def solver_tensors(filter_order, N, num_basis, device):
    key = (int(filter_order), int(N), int(num_basis), _device_cache_key(device))
    cached = _SOLVER_TENSOR_CACHE.get(key)
    if cached is not None:
        return cached

    filter_coeffs = filter_coefficients(filter_order, N, num_basis, device=device)
    upwind_matrices = compute_upwind_matrices(N, device=device)
    cached = filter_coeffs, upwind_matrices
    _SOLVER_TENSOR_CACHE[key] = cached
    return cached


def upwind_flux(N, num_basis, psi, params, upwind_matrices):
    IC_idx = params["IC_idx"]
    dx = params["dx"]
    dy = params["dy"]
    num_x = params["num_x"]
    num_y = params["num_y"]
    batch_size = params["batch_size"]
    device = psi.device
    Ax, Ay, Ax_plus, Ax_minus, Ay_plus, Ay_minus = upwind_matrices

    dx_left = torch.zeros([batch_size, num_y, num_x, num_basis], device=device)
    dx_right = torch.zeros([batch_size, num_y, num_x, num_basis], device=device)
    dy_up = torch.zeros([batch_size, num_y, num_x, num_basis], device=device)
    dy_down = torch.zeros([batch_size, num_y, num_x, num_basis], device=device)

    # Compute differences for slopes in x-direction
    dx_left[:, 1 : num_y - 1, 1:num_x, :] = (
        psi[:, 1 : num_y - 1, 1:num_x, :] - psi[:, 1 : num_y - 1, 0 : num_x - 1, :]
    ) / dx
    dx_right[:, 1 : num_y - 1, 0 : num_x - 1, :] = (
        psi[:, 1 : num_y - 1, 1:num_x, :] - psi[:, 1 : num_y - 1, 0 : num_x - 1, :]
    ) / dx

    # Compute differences for slopes in y-direction
    dy_down[:, 1:num_y, 1 : num_x - 1, :] = (
        psi[:, 1:num_y, 1 : num_x - 1, :] - psi[:, 0 : num_y - 1, 1 : num_x - 1, :]
    ) / dy
    dy_up[:, 0 : num_y - 1, 1 : num_x - 1, :] = (
        psi[:, 1:num_y, 1 : num_x - 1, :] - psi[:, 0 : num_y - 1, 1 : num_x - 1, :]
    ) / dy

    lim_x = minmod(dx_left, dx_right)
    lim_y = minmod(dy_down, dy_up)
    lim_x_plus = torch.zeros_like(lim_x)
    lim_x_minus = torch.zeros_like(lim_x)
    lim_y_plus = torch.zeros_like(lim_y)
    lim_y_minus = torch.zeros_like(lim_y)
    lim_x_plus[:, 1 : num_y - 1, 1 : num_x - 1, :] = (
        lim_x[:, 1 : num_y - 1, 2:num_x, :] - lim_x[:, 1 : num_y - 1, 1 : num_x - 1, :]
    )
    lim_x_minus[:, 1 : num_y - 1, 1 : num_x - 1, :] = (
        lim_x[:, 1 : num_y - 1, 1 : num_x - 1, :]
        - lim_x[:, 1 : num_y - 1, 0 : num_x - 2, :]
    )
    lim_y_plus[:, 1 : num_y - 1, 1 : num_x - 1, :] = (
        lim_y[:, 2:num_y, 1 : num_x - 1, :] - lim_y[:, 1 : num_y - 1, 1 : num_x - 1, :]
    )
    lim_y_minus[:, 1 : num_y - 1, 1 : num_x - 1, :] = (
        lim_y[:, 1 : num_y - 1, 1 : num_x - 1, :]
        - lim_y[:, 0 : num_y - 2, 1 : num_x - 1, :]
    )

    f_plus = torch.matmul(dx_right - 0.5 * lim_x_plus, Ax_minus.T)
    f_minus = torch.matmul(dx_left + 0.5 * lim_x_minus, Ax_plus.T)
    g_plus = torch.matmul(dy_up - 0.5 * lim_y_plus, Ay_minus.T)
    g_minus = torch.matmul(dy_down + 0.5 * lim_y_minus, Ay_plus.T)

    f_plus[:, :, -1, :] = 0
    f_minus[:, :, 0, :] = 0
    g_plus[:, -1, :, :] = 0
    g_minus[:, 0, :, :] = 0

    if IC_idx == 5:
        source_boundary = params["source_boundary"]
        f_minus[:, :, 0, 0] = (psi[0, :, 1, 0] - source_boundary) / dx

    fluxes = f_plus + f_minus + g_plus + g_minus

    # partial derivatives for model features
    dx_psi = torch.zeros([batch_size, num_y, num_x, num_basis], device=device)
    dy_psi = torch.zeros([batch_size, num_y, num_x, num_basis], device=device)

    dx_psi[:, 1 : num_y - 1, 1 : num_x - 1, :] = (
        psi[:, 1 : num_y - 1, 2:num_x, :] - psi[:, 1 : num_y - 1, 0 : num_x - 2, :]
    ) / (2 * dx)
    dy_psi[:, 1 : num_y - 1, 1 : num_x - 1, :] = (
        psi[:, 2:num_y, 1 : num_x - 1, :] - psi[:, 0 : num_y - 2, 1 : num_x - 1, :]
    ) / (2 * dy)

    A_dxpsi = torch.matmul(dx_psi, Ax.T)
    A_dypsi = torch.matmul(dy_psi, Ay.T)

    return fluxes, A_dxpsi, A_dypsi


def _feature_eps(params):
    return float(params.get("feature_eps", params.get("log_feature_eps", 1e-8)))


def _log_clip_bounds(params):
    clip = params.get("feature_log_clip", params.get("log_feature_clip", [0.0, 20.0]))
    if clip is None:
        return None
    if isinstance(clip, (int, float)):
        return 0.0, float(clip)
    if len(clip) != 2:
        raise ValueError("feature_log_clip must be null, a scalar, or [min, max]")
    return float(clip[0]), float(clip[1])


def _log_scale(params):
    return max(
        float(params.get("feature_log_scale", params.get("log_feature_scale", 1.0))),
        _feature_eps(params),
    )


def _log_magnitude(value, params):
    logged = torch.log1p(torch.abs(value) / _log_scale(params))
    bounds = _log_clip_bounds(params)
    if bounds is not None:
        logged = torch.clamp(logged, min=bounds[0], max=bounds[1])
    return logged


def _stat_tensor(value, feature, name):
    if value is None:
        raise ValueError(f"{name} is required when feature_normalization='global'")
    stat = torch.as_tensor(value, dtype=feature.dtype, device=feature.device)
    if stat.ndim == 0:
        return stat
    if stat.numel() != feature.shape[-1]:
        raise ValueError(
            f"{name} has {stat.numel()} entries, but feature group has "
            f"{feature.shape[-1]} channels"
        )
    return stat.reshape((1,) * (feature.ndim - 1) + (stat.numel(),))


def _normalize_feature_tensor(feature, params, mode=None):
    mode = str(mode or params.get("feature_normalization", "sample")).lower()
    if mode in {"sample", "per_sample", "batch"}:
        return NN_normalization(feature)
    if mode in {"none", "identity", "raw"}:
        return feature
    if mode in {"global", "training", "train"}:
        mean = _stat_tensor(
            params.get("feature_global_mean"), feature, "feature_global_mean"
        )
        std = _stat_tensor(
            params.get("feature_global_std"), feature, "feature_global_std"
        )
        return (feature - mean) / (std + _feature_eps(params))
    raise ValueError(
        f"Unsupported feature_normalization={mode!r}. Expected sample, none, or global."
    )


def _log_feature_tensor(feature, params, mode=None):
    return _normalize_feature_tensor(_log_magnitude(feature, params), params, mode=mode)


def _expand_2d_material_field(value, params):
    if value.ndim == 1:
        value = value[:, None, None]
    elif value.ndim == 4 and value.shape[-1] == 1:
        value = value[..., 0]
    if value.ndim != 3:
        raise ValueError(
            f"Expected 2D material field with shape [batch, y, x]; got {value.shape}"
        )
    if value.shape[1] == 1 and value.shape[2] == 1:
        value = value.expand(-1, int(params["num_y"]), int(params["num_x"]))
    return value[:, :, :, None]


def _material_feature_tensor(sigs, sigt, params, mode=None):
    sigs = _expand_2d_material_field(sigs, params)
    sigt = _expand_2d_material_field(sigt, params)
    siga = torch.clamp(sigt - sigs, min=0.0)

    features = [
        _log_magnitude(sigs, params),
        _log_magnitude(sigt, params),
        _log_magnitude(siga, params),
    ]
    if params.get("include_material_ratios", False):
        denom = torch.clamp(torch.abs(sigt), min=_feature_eps(params))
        features.extend((sigs / denom, siga / denom))

    material_features = torch.cat(features, dim=-1)
    mode = params.get("material_feature_normalization", mode or "none")
    return _normalize_feature_tensor(material_features, params, mode=mode)


def _invariant_norm_features(N, advective_term, total_term, params):
    num_x = params["num_x"]
    num_y = params["num_y"]
    batch_size = params["batch_size"]
    device = params["device"]
    advective_norms = torch.zeros([batch_size, num_y, num_x, N + 1], device=device)
    total_norms = torch.zeros([batch_size, num_y, num_x, N + 1], device=device)

    index = 0
    for ell in range(N + 1):
        num_m = ell + 1
        ell_advective = advective_term[:, :, :, index : index + num_m]
        ell_total = total_term[:, :, :, index : index + num_m]
        advective_norms[..., ell] = torch.linalg.norm(ell_advective, ord=2, dim=-1)
        total_norms[..., ell] = torch.linalg.norm(ell_total, ord=2, dim=-1)
        index += num_m

    return advective_norms, total_norms


def preprocess_features(
    N, psi, dxpsi, dypsi, scattering, source, params, sigs=None, sigt=None
):
    variant = feature_variant(params)
    advective_term = dxpsi + dypsi
    advective_norms, total_norms = _invariant_norm_features(
        N, advective_term, psi, params
    )
    scattering_field = -scattering[:, :, :, None]
    source_field = source[:, :, :, None]

    base_groups = (
        NN_normalization(advective_norms),
        NN_normalization(total_norms),
        NN_normalization(scattering_field),
        NN_normalization(source_field),
    )

    log_mode = "none" if variant == "no_norm_log" else None
    log_groups = tuple(
        _log_feature_tensor(group, params, mode=log_mode)
        for group in (advective_norms, total_norms, scattering_field, source_field)
    )

    if variant == "baseline_norm":
        groups = list(base_groups)
    elif variant == "log_norm":
        groups = list(log_groups)
    elif variant == "baseline_plus_log":
        groups = list(base_groups) + list(log_groups)
    elif variant == "log_material_only":
        groups = list(base_groups)
    elif variant == "no_norm_log":
        groups = list(log_groups)
    else:
        raise AssertionError(f"Unhandled feature_variant={variant!r}")

    if _material_features_enabled(params):
        if sigs is None or sigt is None:
            raise ValueError("sigs and sigt are required for material feature channels")
        material_mode = "none" if variant == "no_norm_log" else None
        groups.append(_material_feature_tensor(sigs, sigt, params, mode=material_mode))

    return torch.cat(groups, dim=-1)


def NN_normalization(f):
    f_mean = torch.mean(f, dim=[1, 2], keepdim=True)
    f_std = torch.std(f, dim=[1, 2], keepdim=True)

    f_normalized = (f - f_mean) / (f_std + 1e-10)
    return f_normalized


def timestepping(
    psi0,
    filt_switch,
    NN_model,
    params,
    sigs,
    sigt,
    N,
    num_basis,
    source,
    return_filter_stats=False,
):

    num_x = params["num_x"]
    num_y = params["num_y"]
    num_t = params["num_t"]
    dt = params["dt"]
    batch_size = params["batch_size"]
    device = params["device"]
    obj_idx = params["obj_idx"]
    filter_order = params["filter_order"]

    psi0 = psi0.to(device)
    sigs = sigs.to(device)
    sigt = sigt.to(device)
    source = source.to(device)

    psi_prev = torch.zeros([batch_size, num_y, num_x, num_basis], device=device)
    psi_prev[:, :, :, 0] = psi0

    store_time_history = obj_idx in (1, 2, 3)
    if store_time_history:
        psi_out = torch.zeros(
            [batch_size, num_t, num_y, num_x, num_basis], device=device
        )

    filter_coeffs = filter_coefficients(filter_order, N, num_basis, device=device)
    upwind_matrices = compute_upwind_matrices(N, device=device)
    filter_strength_rollout_max = None

    def update_filter_stats(sigf):
        nonlocal filter_strength_rollout_max
        if not return_filter_stats or filt_switch != 1:
            return
        current_max = sigf.detach().max()
        if filter_strength_rollout_max is None:
            filter_strength_rollout_max = current_max
        else:
            filter_strength_rollout_max = torch.maximum(
                filter_strength_rollout_max, current_max
            )

    for k in range(num_t):
        psi1_update, sigf = PN_update(
            psi_prev,
            N,
            params,
            num_basis,
            sigt,
            sigs,
            filt_switch,
            source,
            NN_model,
            filter_coeffs,
            upwind_matrices,
        )
        update_filter_stats(sigf)
        psi1 = psi_prev + dt * psi1_update
        psi2_update, sigf = PN_update(
            psi1,
            N,
            params,
            num_basis,
            sigt,
            sigs,
            filt_switch,
            source,
            NN_model,
            filter_coeffs,
            upwind_matrices,
        )
        update_filter_stats(sigf)
        psi = psi_prev + 0.5 * dt * (psi1_update + psi2_update)
        psi_prev = psi

        if store_time_history:
            psi_out[:, k, :, :, :] = psi
    if not store_time_history:
        psi_out = psi
    if return_filter_stats:
        if filter_strength_rollout_max is None:
            filter_strength_rollout_max = torch.zeros((), device=device)
        return (
            psi_out,
            sigf[0, :, :],
            {
                "filter_strength_rollout_max": filter_strength_rollout_max,
            },
        )
    return psi_out, sigf[0, :, :]


def PN_update(
    psi_prev,
    N,
    params,
    num_basis,
    sigt,
    sigs,
    filt_switch,
    source,
    NN_model,
    filter_coeffs,
    upwind_matrices,
):

    num_x = params["num_x"]
    num_y = params["num_y"]
    num_features = params["num_features"]
    batch_size = params["batch_size"]
    tt_flag = params["tt_flag"]
    IC_idx = params["IC_idx"]
    device = params["device"]
    filter_type = params["filter_type"]

    fluxes, A_dxpsi, A_dypsi = upwind_flux(
        N, num_basis, psi_prev, params, upwind_matrices
    )

    if tt_flag == 0:
        sigt_psi = sigt[:, None, None, None] * psi_prev
        scattering = sigs[:, None, None] * psi_prev[:, :, :, 0]
    elif tt_flag == 1:
        sigt_psi = sigt[:, :, :, None] * psi_prev
        scattering = sigs * psi_prev[:, :, :, 0]

    sigf = torch.zeros([batch_size, num_y, num_x], device=device)
    psi_update = torch.zeros([batch_size, num_y, num_x, num_basis], device=device)
    if filt_switch == 1:
        if filter_type == 0:
            inputs = preprocess_features(
                N,
                sigt_psi,
                A_dxpsi,
                A_dypsi,
                scattering,
                source,
                params,
                sigs=sigs,
                sigt=sigt,
            )
            sigf = NN_model(inputs).squeeze(-1)
        if filter_type == 1:
            sigf0 = NN_model()
            sigf = sigf0 * torch.ones(batch_size, num_y, num_x, device=sigf0.device)
        if filter_type == 2:
            sigf0 = NN_model
            sigf = sigf0 * torch.ones(batch_size, num_y, num_x, device=device)

    psi_update = -fluxes - sigt_psi
    psi_update[:, :, :, 0] = psi_update[:, :, :, 0] + scattering + source

    if filt_switch == 1:
        sigf_psi = sigf[:, :, :, None] * psi_prev * filter_coeffs
        psi_update = psi_update - sigf_psi

    if IC_idx != 5:
        psi_update[:, :, 0, :] = psi_update[:, :, 1, :]

    psi_update[:, 0, :, :] = psi_update[:, 1, :, :]
    psi_update[:, -1, :, :] = psi_update[:, -2, :, :]
    psi_update[:, :, -1, :] = psi_update[:, :, -2, :]
    psi_update[:, 0, 0, :] = 0.5 * (psi_update[:, 1, 0, :] + psi_update[:, 0, 1, :])
    psi_update[:, 0, -1, :] = 0.5 * (psi_update[:, 0, -2, :] + psi_update[:, 1, -1, :])
    psi_update[:, -1, 0, :] = 0.5 * (psi_update[:, -2, 0, :] + psi_update[:, -1, 1, :])
    psi_update[:, -1, -1, :] = 0.5 * (
        psi_update[:, -2, -1, :] + psi_update[:, -1, -2, :]
    )

    return psi_update, sigf


def compute_cell_average(f, num_x, num_y, num_funcs):
    average = torch.zeros(num_funcs, num_y, num_x, dtype=f.dtype, device=f.device)
    for l in range(0, num_y):
        for m in range(0, num_x):
            average[:, l, m] = 0.25 * (
                f[:, l, m] + f[:, l, m + 1] + f[:, l + 1, m] + f[:, l + 1, m + 1]
            )
    return average


def rotation_test(psi):
    rot_error = np.zeros(2)
    psi_rot = np.rot90(psi)
    rot_error[0] = np.max(np.abs(psi_rot - psi))
    psi_rot = np.rot90(psi)
    rot_error[1] = np.max(np.abs(psi_rot - psi))
    return rot_error
