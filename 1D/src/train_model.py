import math
import torch
import torch.optim as optim
from tqdm.auto import tqdm
from funcs_common import (
    SimpleNN,
    SimpleNN_const,
    nn_feature_count,
    timestepping,
    compute_cell_average,
)
from IC import gaussian_training, heaviside, bump, disc_source
from wandb_utils import finish_run, init_wandb, log_metrics
from wandb_utils import should_log_metrics


def _filter_label(filter_type):
    if filter_type in (1, 2):
        return "nn"
    if filter_type == 3:
        return "constant"
    return f"filter={filter_type}"


def _progress_description(params, filter_type):
    context = (params.get("wandb", {}) or {}).get("context", {}) or {}
    parts = ["1D", str(context.get("ansatz") or _filter_label(filter_type))]
    parts.append(f"N={params.get('N')}")
    if context.get("replicate") is not None:
        parts.append(f"run={context['replicate']}")
    return " ".join(parts)


def _sqrt_loss(loss_value):
    if loss_value >= 0 and math.isfinite(loss_value):
        return math.sqrt(loss_value)
    return float("nan")


def _tensor_p95(value):
    return torch.quantile(value.reshape(-1), 0.95)


def _set_progress_postfix(progress, loss_value, best_loss):
    progress.set_postfix(
        {
            "loss": f"{loss_value:.3e}",
            "sqrt_loss": f"{_sqrt_loss(loss_value):.3e}",
            "best": f"{best_loss:.3e}",
        },
        refresh=True,
    )


def _set_epoch_learning_rate(opt, base_lr, epoch, num_epochs, params):
    scheduler = str(
        params.get("lr_scheduler", params.get("learning_rate_scheduler", "none"))
    ).lower()
    if scheduler in {"", "none", "constant"}:
        return base_lr
    if scheduler not in {"cosine", "cosine_warmup", "warmup_cosine"}:
        raise ValueError(
            f"Unsupported lr_scheduler={scheduler!r}. "
            "Expected one of: none, constant, cosine."
        )

    warmup_fraction = float(
        params.get("lr_warmup_fraction", params.get("warmup_fraction", 0.0))
    )
    min_factor = float(params.get("lr_min_factor", 0.0))
    if not 0.0 <= warmup_fraction <= 1.0:
        raise ValueError("lr_warmup_fraction must be in [0, 1]")
    if not 0.0 <= min_factor <= 1.0:
        raise ValueError("lr_min_factor must be in [0, 1]")

    if num_epochs <= 1:
        factor = 1.0
    else:
        warmup_epochs = math.ceil(num_epochs * warmup_fraction)
        warmup_epochs = min(warmup_epochs, max(num_epochs - 1, 1))
        if warmup_epochs > 0 and epoch < warmup_epochs:
            factor = (epoch + 1) / warmup_epochs
        else:
            decay_steps = max(1, num_epochs - warmup_epochs - 1)
            decay_epoch = min(max(epoch - warmup_epochs, 0), decay_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * decay_epoch / decay_steps))
            factor = min_factor + (1.0 - min_factor) * cosine

    lr = base_lr * factor
    for group in opt.param_groups:
        group["lr"] = lr
    return lr


def _training_data_mode(params):
    mode = str(params.get("training_data", params.get("training_setup", "paper"))).lower()
    aliases = {
        "paper": "paper",
        "current": "paper",
        "default": "paper",
        "augmented": "augmented",
        "expanded": "augmented",
        "mixed": "augmented",
    }
    if mode not in aliases:
        raise ValueError(
            f"Unsupported 1D training_data={mode!r}. "
            "Expected one of: paper, current, augmented, expanded, mixed."
        )
    return aliases[mode]


def _rand_uniform(low, high, device):
    return low + (high - low) * torch.rand((), device=device)


def _interval_mask(x, center, half_width):
    return torch.abs(x - center) <= half_width


def _choose_time_horizon(params):
    horizons = params.get("training_time_horizons") or params.get(
        "augmented_time_horizons"
    )
    if not horizons:
        return float(params["T"])
    idx = int(torch.randint(0, len(horizons), ()).item())
    return float(horizons[idx])


def _make_step_params(params, final_time):
    step_params = dict(params)
    step_params["tt_flag"] = 1
    step_params["T"] = final_time
    step_params["num_t"] = int((final_time + step_params["dt"]) // step_params["dt"])
    return step_params


def _fill_initial_condition(psi0_row, source_row, x_edges, family, device):
    if family == 0:
        center = _rand_uniform(-0.4, 0.4, device)
        width = _rand_uniform(0.03, 0.15, device)
        amplitude = _rand_uniform(0.5, 2.0, device)
        scale = amplitude / torch.sqrt(torch.tensor(2 * math.pi, device=device)) / width
        psi0_row[:] = scale * torch.exp(-((x_edges - center) ** 2) / (2 * width**2))
    elif family == 1:
        center = _rand_uniform(-0.35, 0.35, device)
        half_width = _rand_uniform(0.08, 0.35, device)
        amplitude = _rand_uniform(0.5, 5.0, device)
        psi0_row[_interval_mask(x_edges, center, half_width)] = amplitude
    elif family == 2:
        center = _rand_uniform(-0.35, 0.35, device)
        width = _rand_uniform(0.2, 0.55, device)
        amplitude = _rand_uniform(0.5, 3.0, device)
        local = torch.abs((x_edges - center) / width)
        mask = local < 1
        psi0_row[mask] = amplitude * torch.cos(0.5 * math.pi * local[mask])
    elif family == 3:
        center = _rand_uniform(-0.25, 0.25, device)
        half_width = _rand_uniform(0.05, 0.25, device)
        psi0_row[_interval_mask(x_edges, center, half_width)] = _rand_uniform(
            0.5, 2.0, device
        )
        source_row[torch.abs(x_edges) > _rand_uniform(0.65, 0.85, device)] = _rand_uniform(
            0.5, 5.0, device
        )
        source_row[_interval_mask(x_edges, center, half_width)] += _rand_uniform(
            0.25, 2.0, device
        )
    elif family == 4:
        half_width = _rand_uniform(0.08, 0.3, device)
        psi0_row[torch.abs(x_edges) < half_width] = _rand_uniform(0.5, 5.0, device)
    elif family == 5:
        for _ in range(2):
            center = _rand_uniform(-0.55, 0.55, device)
            half_width = _rand_uniform(0.04, 0.16, device)
            source_row[_interval_mask(x_edges, center, half_width)] += _rand_uniform(
                1.0, 20.0, device
            )
    else:
        # Reeds-like source-driven case on the paper 1D domain.
        source_row[x_edges < -0.5] = _rand_uniform(5.0, 50.0, device)
        source_row[(x_edges > 0.25) & (x_edges < 0.55)] = _rand_uniform(
            0.5, 3.0, device
        )


def _fill_cross_sections(sigs_row, sigt_row, source_row, x_edges, case, sigs_max, device):
    if case == 0:
        sigs0 = torch.rand((), device=device) * sigs_max
        siga0 = (sigs_max - sigs0) * torch.rand((), device=device)
        sigs_row[:] = sigs0
        sigt_row[:] = sigs0 + siga0
    elif case == 1:
        center = _rand_uniform(-0.2, 0.2, device)
        power = _rand_uniform(2.0, 6.0, device)
        scale = _rand_uniform(1.0, 100.0, device)
        floor = _rand_uniform(0.0, 1e-3, device)
        sigs_row[:] = floor + scale * torch.abs(x_edges - center) ** power
        sigt_row[:] = sigs_row
    elif case == 2:
        high = _rand_uniform(0.5, 10.0, device)
        low = _rand_uniform(0.0, 0.25, device)
        sigs_row[:] = high
        for center in (-_rand_uniform(0.3, 0.7, device), _rand_uniform(0.3, 0.7, device)):
            half_width = _rand_uniform(0.05, 0.2, device)
            sigs_row[_interval_mask(x_edges, center, half_width)] = low
        sigt_row[:] = sigs_row + _rand_uniform(0.0, 1.0, device)
    elif case == 3:
        sigs_row[:] = _rand_uniform(0.2, 3.0, device)
        sigt_row[:] = sigs_row + _rand_uniform(0.0, 2.0, device)
        center = _rand_uniform(-0.3, 0.3, device)
        half_width = _rand_uniform(0.08, 0.35, device)
        vacuum = _interval_mask(x_edges, center, half_width)
        sigs_row[vacuum] = 0
        sigt_row[vacuum] = 0
    elif case == 4:
        background = _rand_uniform(0.05, 1.0, device)
        high_t = _rand_uniform(5.0, 50.0, device)
        scatter_fraction = _rand_uniform(0.0, 0.95, device)
        sigt_row[:] = background
        sigs_row[:] = background * scatter_fraction
        center = _rand_uniform(-0.55, 0.55, device)
        half_width = _rand_uniform(0.05, 0.25, device)
        material = _interval_mask(x_edges, center, half_width)
        sigt_row[material] = high_t
        sigs_row[material] = high_t * scatter_fraction
    else:
        sigs_row[:] = 0
        sigt_row[:] = 0
        regions = [
            (x_edges < -0.5, 0.0, 50.0),
            ((x_edges >= -0.5) & (x_edges < -0.25), 0.0, 5.0),
            ((x_edges >= -0.25) & (x_edges < 0.25), 0.0, 0.0),
            ((x_edges >= 0.25) & (x_edges < 0.5), 0.9, 1.0),
            (x_edges >= 0.5, 0.9, 1.0),
        ]
        scale = _rand_uniform(0.5, 1.5, device)
        for mask, sigs_value, sigt_value in regions:
            sigs_row[mask] = sigs_value * scale
            sigt_row[mask] = sigt_value * scale
        source_row[x_edges < -0.5] += _rand_uniform(5.0, 50.0, device)
        source_row[(x_edges >= 0.25) & (x_edges < 0.5)] += _rand_uniform(
            0.5, 2.0, device
        )


def _sample_augmented_training_batch(params, device):
    batch_size = params["batch_size"]
    num_x = params["num_x"]
    sigs_max = params["sigs_max"]
    x_edges = params["x_edges"].to(device)

    psi0_edges = torch.zeros([batch_size, num_x + 1], device=device)
    source_edges = torch.zeros([batch_size, num_x + 1], device=device)
    sigs_edges = torch.zeros([batch_size, num_x + 1], device=device)
    sigt_edges = torch.zeros([batch_size, num_x + 1], device=device)

    for b in range(batch_size):
        family = int(torch.randint(0, 7, ()).item())
        material_case = int(torch.randint(0, 6, ()).item())
        _fill_initial_condition(
            psi0_edges[b], source_edges[b], x_edges, family, device
        )
        _fill_cross_sections(
            sigs_edges[b],
            sigt_edges[b],
            source_edges[b],
            x_edges,
            material_case,
            sigs_max,
            device,
        )

    psi0 = compute_cell_average(psi0_edges, batch_size, num_x)
    source = compute_cell_average(source_edges, batch_size, num_x)
    sigs = compute_cell_average(sigs_edges, batch_size, num_x)
    sigt = compute_cell_average(sigt_edges, batch_size, num_x)

    final_time = _choose_time_horizon(params)
    step_params = _make_step_params(params, final_time)
    return psi0, source, sigs, sigt, step_params, final_time


def _mean_square_per_sample(value):
    dims = tuple(range(1, value.ndim))
    return torch.mean(value**2, dim=dims)


def _relative_mse_per_sample(error, reference, eps=1e-12):
    numerator = _mean_square_per_sample(error)
    denominator = _mean_square_per_sample(reference).clamp_min(eps)
    return torch.mean(numerator / denominator)


def _training_loss(psi, exact, N, obj_idx, params):
    eps = float(params.get("relative_loss_eps", 1e-12))
    if obj_idx == 0:
        loss = _relative_mse_per_sample(
            psi[:, :, 0] - exact[:, :, 0], exact[:, :, 0], eps
        )
    elif obj_idx == 1:
        loss = _relative_mse_per_sample(
            psi - exact[:, :, : N + 1], exact[:, :, : N + 1], eps
        )
    elif obj_idx == 2:
        loss = _relative_mse_per_sample(
            psi[:, :, :, 0] - exact[:, :, :, 0], exact[:, :, :, 0], eps
        )
    elif obj_idx == 3:
        loss = _relative_mse_per_sample(
            psi[:, :, :, 0] - exact[:, :, :, 0], exact[:, :, :, 0], eps
        )
    else:
        raise ValueError(f"Unsupported obj_idx={obj_idx}")

    aux_weight = float(params.get("aux_moment_loss_weight", 0.0))
    if aux_weight > 0 and obj_idx == 0:
        loss = loss + aux_weight * _relative_mse_per_sample(
            psi - exact[:, :, : N + 1], exact[:, :, : N + 1], eps
        )
    return loss


def training(params):

    num_x = params["num_x"]
    N_exact = params["N_exact"]
    N = params["N"]
    L = params["L"]
    dx = params["dx"]
    num_IC = params["num_IC"]
    sigs_max = params["sigs_max"]
    num_epochs = params["num_epochs"]
    batch_size = params["batch_size"]
    learning_rate = params["learning_rate"]
    momentum_factor = params["momentum_factor"]
    num_features = params["num_features"]
    num_hidden = params["num_hidden"]
    GD_optimizer = params["GD_optimizer"]
    weight_decay = params["weight_decay"]
    x_edges = params["x_edges"]
    device = params["device"]
    filter_type = params["filter_type"]
    obj_idx = params["obj_idx"]
    if filter_type in (1, 2):
        params["num_features"] = nn_feature_count(N, params)
        num_features = params["num_features"]

    if filter_type in (1, 2):
        NN_model = SimpleNN(num_features, num_hidden, N)
    elif filter_type == 3:
        NN_model = SimpleNN_const()

    NN_model = NN_model.to(device)
    if GD_optimizer == "SGD":
        opt = optim.SGD(
            NN_model.parameters(), lr=learning_rate, momentum=momentum_factor
        )
    elif GD_optimizer == "Adam":
        opt = optim.Adam(
            NN_model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    wandb_run = init_wandb(params)

    training_data = _training_data_mode(params)
    if training_data == "paper":
        psi0_edges = torch.zeros([batch_size, num_x + 1], device=device)
        source_edges = torch.zeros([batch_size, num_x + 1], device=device)
        reg1 = torch.arange(0, round(batch_size / num_IC), device=device)
        reg2 = torch.arange(
            round(batch_size / num_IC), round(2 * batch_size / num_IC), device=device
        )
        reg3 = torch.arange(
            round(2 * batch_size / num_IC), round(3 * batch_size / num_IC), device=device
        )
        reg4 = torch.arange(
            round(3 * batch_size / num_IC), round(4 * batch_size / num_IC), device=device
        )

        psi0_edges[reg1, :] = gaussian_training(num_x, x_edges)[0].to(device)
        psi0_edges[reg2, :] = heaviside(num_x, x_edges)[0].to(device)
        psi0_edges[reg3, :] = bump(num_x, x_edges)[0].to(device)
        disc_source_output = disc_source(num_x, x_edges)
        psi0_edges[reg4, :] = disc_source_output[0].to(device)
        source_edges[reg4, :] = disc_source_output[3].to(device)

        psi0 = compute_cell_average(psi0_edges, batch_size, num_x)
        source = compute_cell_average(source_edges, batch_size, num_x)
    else:
        psi0 = None
        source = None

    best_loss = math.inf
    progress = tqdm(
        range(num_epochs),
        desc=_progress_description(params, filter_type),
        unit="epoch",
        dynamic_ncols=True,
    )

    for l in progress:
        current_lr = _set_epoch_learning_rate(opt, learning_rate, l, num_epochs, params)
        opt.zero_grad()
        if training_data == "paper":
            sigs = torch.rand(batch_size, device=device) * sigs_max
            siga = (sigs_max - sigs) * torch.rand(batch_size, device=device)
            sigt = sigs + siga
            batch_psi0 = psi0
            batch_source = source
            step_params = params
            final_time = float(params["T"])
        else:
            (
                batch_psi0,
                batch_source,
                sigs,
                sigt,
                step_params,
                final_time,
            ) = _sample_augmented_training_batch(params, device)

        exact = timestepping(
            batch_psi0,
            0,
            0,
            step_params,
            sigs,
            sigt,
            N_exact,
            batch_source,
            batch_size,
            device,
        )[0]
        psi, sigf, filter_stats = timestepping(
            batch_psi0,
            filter_type,
            NN_model,
            step_params,
            sigs,
            sigt,
            N,
            batch_source,
            batch_size,
            device,
            return_filter_stats=True,
        )

        loss = _training_loss(psi, exact, N, obj_idx, params)

        loss.backward()
        opt.step()
        if filter_type == 3:
            with torch.no_grad():
                NN_model.const.clamp_(min=0.0)

        final_epoch = l == num_epochs - 1
        should_report = should_log_metrics(params, l, final=final_epoch)
        if should_report:
            loss_value = loss.detach().item()
            if math.isfinite(loss_value):
                best_loss = min(best_loss, loss_value)
            _set_progress_postfix(progress, loss_value, best_loss)
        else:
            loss_value = None

        if wandb_run is not None and should_report:
            sigs_detached = sigs.detach()
            sigt_detached = sigt.detach()
            sigma_a_detached = sigt_detached - sigs_detached
            metrics = {
                "train/loss": loss_value,
                "train/sqrt_loss": _sqrt_loss(loss_value),
                "train/sigma_s_mean": sigs_detached.mean().item(),
                "train/sigma_t_mean": sigt_detached.mean().item(),
                "train/sigma_t_max": sigt_detached.max().item(),
                "train/sigma_t_p95": _tensor_p95(sigt_detached).item(),
                "train/sigma_a_mean": sigma_a_detached.mean().item(),
                "train/time_horizon": final_time,
                "train/learning_rate": current_lr,
                "train/epoch": l,
            }
            if filter_type in (1, 2, 3):
                sigf_detached = sigf.detach()
                metrics.update(
                    {
                        "train/filter_strength_mean": sigf_detached.mean().item(),
                        "train/filter_strength_min": sigf_detached.min().item(),
                        "train/filter_strength_max": sigf_detached.max().item(),
                        "train/filter_strength_p95": _tensor_p95(sigf_detached).item(),
                        "train/filter_strength_rollout_max": filter_stats[
                            "filter_strength_rollout_max"
                        ].item(),
                    }
                )
            log_metrics(wandb_run, metrics, l, params, final=final_epoch)

        if should_report:
            if not math.isfinite(loss_value):
                tqdm.write(
                    f"{_progress_description(params, filter_type)}: "
                    f"non-finite loss at epoch {l}; stopping training."
                )
                break

    progress.close()
    finish_run(wandb_run)
    return NN_model
