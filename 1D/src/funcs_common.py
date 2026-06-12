import torch
import numpy as np
import torch.nn as nn

_TIMESTEPPING_TENSOR_CACHE = {}
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


def _timestepping_tensors(N, filter_order, device):
    key = (int(N), int(filter_order), _device_cache_key(device))
    cached = _TIMESTEPPING_TENSOR_CACHE.get(key)
    if cached is not None:
        return cached

    n = torch.arange(1, N + 1, dtype=torch.float32, device=device)
    a = n / torch.sqrt((2 * n - 1) * (2 * n + 1))
    A = torch.diag(a, 1) + torch.diag(a, -1)

    eigA, V = torch.linalg.eigh(A)
    absA = torch.matmul(torch.matmul(V, torch.diag(torch.abs(eigA))), V.T)

    filt_input = torch.arange(0, N + 1, 1, device=device) / (N + 1)
    filter_coeffs = -torch.log(filter_func(filt_input, filter_order))

    cached = A, absA, filter_coeffs
    _TIMESTEPPING_TENSOR_CACHE[key] = cached
    return cached


class SimpleNN_const(nn.Module):
    def __init__(self):
        super().__init__()
        self.const = nn.Parameter(0.01 * torch.rand(()))

    def forward(self):
        return self.const


class SimpleNN(nn.Module):
    def __init__(self, num_features, num_hidden, N):
        super(SimpleNN, self).__init__()
        self.input_norm = nn.LayerNorm(num_features)
        self.hidden = nn.Linear(num_features, num_hidden)  # (inputs,hidden)
        self.hidden2 = nn.Linear(num_hidden, num_hidden)
        self.hidden_norm = nn.LayerNorm(num_hidden)
        self.hidden_activation = nn.Softplus()
        self.output = nn.Linear(num_hidden, 1)  # (hidden,output)
        self.output_activation = nn.Softplus()
        self.N = N

    def forward(self, x):
        original_shape = x.shape
        x = torch.flatten(x, start_dim=0, end_dim=1)
        x = self.input_norm(x)
        x = self.hidden_activation(
            self.hidden_norm(self.hidden(x))
        )  # Activation hidden layer
        x = (
            self.hidden_activation(self.hidden_norm(self.hidden2(x))) + x
        )  # Activation hidden layer
        x = self.output_activation(self.output(x))  # Positive filter strength
        output_shape = [original_shape[0], original_shape[1], 1]
        return x.reshape(output_shape)  # torch.ones(output_shape)  #


def timestepping(
    y0,
    filter_type,
    NN_model,
    params,
    sigs,
    sigt,
    N,
    source,
    batch_size,
    device,
    return_filter_stats=False,
):
    dt = params["dt"]
    dx = params["dx"]
    tt_flag = params["tt_flag"]
    IC_idx = params["IC_idx"]

    num_x = params["num_x"]
    num_t = params["num_t"]
    method_order = params["method_order"]
    obj_idx = params["obj_idx"]

    A, absA, filter_coeffs = _timestepping_tensors(N, params["filter_order"], device)

    source = source.to(device)

    sigt = sigt.to(device)
    sigs = sigs.to(device)

    y_prev = torch.zeros([batch_size, num_x, N + 1], device=device)
    y_prev[:, :, 0] = y0
    y = y_prev
    source_in = source[:, :, None]

    y_out = torch.zeros(batch_size, num_t, num_x, N + 1, device=device)

    if tt_flag == 0:
        sigt_in = sigt[:, None, None]
        sigs_in = sigs[:, None, None]
    if tt_flag == 1:
        sigt_in = sigt[:, :, None]
        sigs_in = sigs[:, :, None]

    store_time_history = obj_idx in (2, 3)
    filter_strength_rollout_max = None

    def update_filter_stats(sigf):
        nonlocal filter_strength_rollout_max
        if not return_filter_stats or filter_type not in (1, 2, 3):
            return
        current_max = sigf.detach().max()
        if filter_strength_rollout_max is None:
            filter_strength_rollout_max = current_max
        else:
            filter_strength_rollout_max = torch.maximum(
                filter_strength_rollout_max, current_max
            )

    for k in range(num_t):
        y1_update, sigf = PN_update(
            params,
            y_prev,
            A,
            absA,
            N,
            source,
            filter_type,
            NN_model,
            source_in,
            sigt_in,
            sigs_in,
            filter_coeffs,
        )
        update_filter_stats(sigf)
        y1 = y_prev + dt * y1_update

        # boundary conditions for Reeds problem: reflecting at x = 0 and vacauum at x = 8
        if IC_idx == 6:
            y1 = reeds_BC(y1, N)
        if method_order == 1:
            y = y1
        elif method_order == 2:
            y2_update, sigf = PN_update(
                params,
                y1,
                A,
                absA,
                N,
                source,
                filter_type,
                NN_model,
                source_in,
                sigt_in,
                sigs_in,
                filter_coeffs,
            )
            update_filter_stats(sigf)
            y = y_prev + 0.5 * dt * (y1_update + y2_update)

            if not store_time_history:
                y_out = y
            if store_time_history:
                y_out[:, k, :, :] = y
            # boundary conditions for Reeds problem: reflecting at x = 0 and vacauum at x = 8
            if IC_idx == 6:
                y = reeds_BC(y, N)
        y_prev = y

    if return_filter_stats:
        if filter_strength_rollout_max is None:
            filter_strength_rollout_max = torch.zeros((), device=device)
        return y_out, sigf, {
            "filter_strength_rollout_max": filter_strength_rollout_max,
        }
    return y_out, sigf


def PN_update(
    params,
    y_prev,
    A,
    absA,
    N,
    source,
    filter_type,
    NN_model,
    source_in,
    sigt,
    sigs,
    filter_coeffs,
):
    batch_size = params["batch_size"]
    device = params["device"]
    IC_idx = params["IC_idx"]
    num_x = params["num_x"]
    dx = params["dx"]
    method_order = params["method_order"]

    slope = torch.zeros([batch_size, num_x + 2, N + 1], device=device)
    y_expand = torch.zeros([batch_size, num_x + 2, N + 1], device=device)

    y_expand[:, 1 : num_x + 1, :] = y_prev
    if IC_idx != 6:
        y_expand[:, 0, :] = y_prev[:, num_x - 1, :]
        y_expand[:, num_x + 1, :] = y_prev[:, 0, :]

    if method_order == 2:
        slope[:, 1 : num_x + 1, :] = minmod(
            y_expand[:, 2 : num_x + 2, :] - y_expand[:, 1 : num_x + 1, :],
            y_expand[:, 1 : num_x + 1, :] - y_expand[:, 0:num_x, :],
        )

        if IC_idx != 6:
            slope[:, 0, :] = slope[:, num_x, :]
            slope[:, num_x + 1, :] = slope[:, 1, :]

    yL_plus = y_expand[:, 1 : num_x + 1, :] + 0.5 * slope[:, 1 : num_x + 1, :]

    yR_plus = y_expand[:, 2 : num_x + 2, :] - 0.5 * slope[:, 2 : num_x + 2, :]

    Ay_plus = 0.5 * torch.matmul(yL_plus + yR_plus, A.T) - 0.5 * torch.matmul(
        yR_plus - yL_plus, absA.T
    )

    yL_minus = y_expand[:, 0:num_x, :] + 0.5 * slope[:, 0:num_x, :]

    yR_minus = y_expand[:, 1 : num_x + 1, :] - 0.5 * slope[:, 1 : num_x + 1, :]

    Ay_minus = 0.5 * torch.matmul(yL_minus + yR_minus, A.T) - 0.5 * torch.matmul(
        yR_minus - yL_minus, absA.T
    )
    A_Dy = (Ay_plus - Ay_minus) / dx

    sigf = torch.zeros([batch_size, num_x], device=device)

    if filter_type in (1, 2):
        yflux = y_prev[:, :, 0]
        yflux = yflux[:, :, None]
        inputs = preprocess_features(
            A_Dy,
            sigt * y_prev,
            sigs * yflux,
            source_in,
            filter_type,
            params,
            sigs=sigs,
            sigt=sigt,
        )
        sigf = NN_model(inputs).squeeze(-1)

    if filter_type == 3:
        sigf0 = NN_model()
        # sigf0 = torch.max(NN_model(),0)
        sigf = sigf0 * torch.ones([batch_size, num_x], device=device)
        # print(sigf0)

    if IC_idx == 6:
        sigf[:, 0] = sigf[:, 1]
        sigf[:, num_x - 1] = sigf[:, num_x - 2]

    sigt_y = sigt * y_expand[:, 1 : num_x + 1, :]
    y_update = -A_Dy - sigt_y

    if filter_type in (1, 2, 3):
        y_update = y_update - sigf[:, :, None] * y_prev * filter_coeffs
        # print('sigf_max = ',torch.max(sigf[:, :, None]), ' sigf_min = ',torch.min(sigf[:, :, None]))

    y_update[:, :, 0] = (
        y_update[:, :, 0] + sigs[:, :, 0] * y_expand[:, 1 : num_x + 1, 0] + source
    )

    return y_update, sigf


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
        mean = _stat_tensor(params.get("feature_global_mean"), feature, "feature_global_mean")
        std = _stat_tensor(params.get("feature_global_std"), feature, "feature_global_std")
        return (feature - mean) / (std + _feature_eps(params))
    raise ValueError(
        f"Unsupported feature_normalization={mode!r}. "
        "Expected sample, none, or global."
    )


def _log_feature_tensor(feature, params, mode=None):
    return _normalize_feature_tensor(_log_magnitude(feature, params), params, mode=mode)


def _expand_1d_material_field(value, params):
    if value.ndim == 1:
        value = value[:, None, None]
    elif value.ndim == 2:
        value = value[:, :, None]
    if value.ndim != 3 or value.shape[-1] != 1:
        raise ValueError(f"Expected 1D material field with trailing singleton dim; got {value.shape}")
    if value.shape[1] == 1:
        value = value.expand(-1, int(params["num_x"]), -1)
    return value


def _material_feature_tensor(sigs, sigt, params, mode=None):
    sigs = _expand_1d_material_field(sigs, params)
    sigt = _expand_1d_material_field(sigt, params)
    siga = torch.clamp(sigt - sigs, min=0.0)

    features = [_log_magnitude(sigs, params), _log_magnitude(sigt, params), _log_magnitude(siga, params)]
    if params.get("include_material_ratios", False):
        denom = torch.clamp(torch.abs(sigt), min=_feature_eps(params))
        features.extend((sigs / denom, siga / denom))

    material_features = torch.cat(features, dim=-1)
    mode = params.get("material_feature_normalization", mode or "none")
    return _normalize_feature_tensor(material_features, params, mode=mode)


def _legacy_feature_groups_1d(A_Dy, sigt_y, scattering, source, filter_type):
    scattering_NN = NN_normalization(torch.abs(scattering))
    source_NN = NN_normalization(torch.abs(source))

    if filter_type == 1:
        A_Dy_NN = NN_normalization(torch.abs(A_Dy))
        sigt_y_NN = NN_normalization(torch.abs(sigt_y))
    elif filter_type == 2:
        A_Dy_mixed = A_Dy.clone()
        sigt_y_mixed = sigt_y.clone()
        A_Dy_mixed[:, :, 1::2] = torch.abs(A_Dy[:, :, 1::2])
        sigt_y_mixed[:, :, 1::2] = torch.abs(sigt_y[:, :, 1::2])
        A_Dy_NN = NN_normalization(A_Dy_mixed)
        sigt_y_NN = NN_normalization(sigt_y_mixed)
    else:
        raise ValueError(f"Unsupported NN filter_type={filter_type}")

    return A_Dy_NN, sigt_y_NN, scattering_NN, source_NN


def _apply_1d_ablation(groups, params):
    ablation_idx = int(params.get("ablation_idx", 0))
    selections = {
        0: {0, 1, 2, 3},
        1: {0},
        2: {1},
        3: {2},
        4: {3},
        5: set(),
        6: {1, 2, 3},
        7: {0, 2, 3},
        8: {0, 1, 3},
        9: {0, 1, 2},
    }
    if ablation_idx not in selections:
        raise ValueError(f"Unsupported ablation_idx={ablation_idx}")
    selected = selections[ablation_idx]
    return tuple(group if idx in selected else torch.zeros_like(group) for idx, group in enumerate(groups))


def preprocess_features(A_Dy, sigt_y, scattering, source, filter_type, params, sigs=None, sigt=None):
    variant = feature_variant(params)
    base_groups = _apply_1d_ablation(
        _legacy_feature_groups_1d(A_Dy, sigt_y, scattering, source, filter_type), params
    )

    log_mode = "none" if variant == "no_norm_log" else None
    raw_groups = (A_Dy, sigt_y, scattering, source)
    log_groups = _apply_1d_ablation(
        tuple(_log_feature_tensor(group, params, mode=log_mode) for group in raw_groups),
        params,
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


def filter_func(z, p):
    return torch.exp(-(z**p))


def NN_normalization(f):
    f_mean = torch.mean(f, dim=[1], keepdim=True)
    f_std = torch.std(f, dim=[1], keepdim=True)
    f_normalized = (f - f_mean) / (f_std + 1e-10)
    return f_normalized
    # return f


def minmod(a, b):
    mm = torch.zeros_like(a)
    mm = torch.where((torch.abs(a) <= torch.abs(b)) & (a * b > 0), a, mm)
    mm = torch.where((torch.abs(b) < torch.abs(a)) & (a * b > 0), b, mm)
    return mm


def obj_func(z):
    return torch.mean(z**2)
    # return (dx * z.pow(2).sum(dim=1)).mean()


def obj_func_time(z):
    dims = tuple(i for i in range(z.ndim) if i != 1)
    return torch.sum(torch.mean(z**2, dim=dims))


def obj_func_time_average(z):
    dims = tuple(i for i in range(z.ndim) if i != 1)
    return torch.mean(torch.mean(z**2, dim=dims))


def compute_cell_average(f, batch_size, num_x):
    f_average = torch.zeros(batch_size, num_x, dtype=f.dtype, device=f.device)
    for m in range(0, num_x):
        f_average[:, m] = 0.5 * (f[:, m] + f[:, m + 1])

    return f_average


def reeds_BC(z, N):
    for n in range(0, N, 2):
        z[:, 0, n] = z[:, 1, n]
    for n in range(1, N + 1, 2):
        z[:, 0, n] = -z[:, 1, n]
    return z
