import math
import torch.optim as optim
import torch
import numpy as np
from tqdm.auto import tqdm
from funcs_common import (
    SimpleNN,
    SimpleNN_const,
    obj_func,
    obj_func_time,
    timestepping,
    compute_cell_average,
)
from IC import gaussian_training, step, disc_source, bump, hat
from wandb_utils import finish_run, init_wandb, log_metrics
from wandb_utils import should_log_metrics
from training_sources import (
    frame_source,
    two_rect_source,
    gaussian_source,
    pulse_source,
)


def _filter_label(filter_type):
    if filter_type == 0:
        return "nn"
    if filter_type == 1:
        return "constant"
    return f"filter={filter_type}"


def _progress_description(params, filter_type):
    context = (params.get("wandb", {}) or {}).get("context", {}) or {}
    parts = ["2D", str(context.get("ansatz") or _filter_label(filter_type))]
    parts.append(f"N={params.get('N')}")
    if context.get("replicate") is not None:
        parts.append(f"run={context['replicate']}")
    return " ".join(parts)


def _sqrt_loss(loss_value):
    if loss_value >= 0 and math.isfinite(loss_value):
        return math.sqrt(loss_value)
    return float("nan")


def _set_progress_postfix(progress, loss_value, best_loss):
    progress.set_postfix(
        {
            "loss": f"{loss_value:.3e}",
            "sqrt_loss": f"{_sqrt_loss(loss_value):.3e}",
            "best": f"{best_loss:.3e}",
        },
        refresh=True,
    )


def training(params):

    num_x = params["num_x"]
    num_y = params["num_y"]
    N_exact = params["N_exact"]
    N = params["N"]
    num_basis = params["num_basis"]
    num_basis_exact = params["num_basis_exact"]
    num_epochs = params["num_epochs"]
    batch_size = params["batch_size"]
    learning_rate = params["learning_rate"]
    momentum_factor = params["momentum_factor"]
    num_features = params["num_features"]
    num_hidden = params["num_hidden"]
    x_edges = params["x_edges"]
    y_edges = params["y_edges"]
    sigs_max = params["sigs_max"]
    GD_optimizer = params["GD_optimizer"]
    num_IC = params["num_IC"]
    device = params["device"]
    obj_idx = params["obj_idx"]
    init_bias = params["init_bias"]
    filter_type = params["filter_type"]

    if filter_type == 0:
        NN_model = SimpleNN(num_features, num_hidden)
        with torch.no_grad():
            NN_model.output.bias.fill_(init_bias)
    elif filter_type == 1:
        NN_model = SimpleNN_const()
        with torch.no_grad():
            NN_model.const.fill_(init_bias)

    NN_model = NN_model.to(device)

    if GD_optimizer == "SGD":
        opt = optim.SGD(
            NN_model.parameters(), lr=learning_rate, momentum=momentum_factor
        )
    elif GD_optimizer == "Adam":
        opt = optim.Adam(NN_model.parameters(), lr=learning_rate)

    wandb_run = init_wandb(params)

    psi0_nodes = torch.zeros([num_IC, num_y + 1, num_x + 1])
    source_nodes = torch.zeros([num_IC, num_y + 1, num_x + 1])

    psi0_nodes[0, :, :] = gaussian_training(num_x, num_y, x_edges, y_edges)[0]
    psi0_nodes[1, :, :] = step(num_x, num_y, x_edges, y_edges)[0]
    psi0_nodes[2, :, :] = disc_source(num_x, num_y, x_edges, y_edges)[0]
    psi0_nodes[3, :, :] = bump(num_x, num_y, x_edges, y_edges)[0]
    psi0_nodes[4, :, :] = hat(num_x, num_y, x_edges, y_edges)[0]

    # assemble sources
    source_nodes[0, :, :] = 0
    source_nodes[1, :, :] = frame_source(num_x, num_y, x_edges, y_edges)
    source_nodes[2, :, :] = two_rect_source(num_x, num_y, x_edges, y_edges)
    source_nodes[3, :, :] = gaussian_source(num_x, num_y, x_edges, y_edges)
    source_nodes[4, :, :] = pulse_source(num_x, num_y, x_edges, y_edges)

    psi0 = compute_cell_average(psi0_nodes, num_x, num_y, num_IC)
    source = compute_cell_average(source_nodes, num_x, num_y, num_IC)

    psi0 = psi0.to(device)
    source = source.to(device)

    best_loss = math.inf
    progress = tqdm(
        range(num_epochs),
        desc=_progress_description(params, filter_type),
        unit="epoch",
        dynamic_ncols=True,
    )

    for l in progress:
        opt.zero_grad()
        sigs = torch.rand(batch_size, device=device) * sigs_max
        siga = (1 - sigs) * torch.rand(batch_size, device=device)
        sigt = sigs + siga

        training_indices = torch.randint(0, num_IC, (batch_size, 2), device=device)
        psi0_training = psi0[training_indices[:, 0], :, :]
        source_training = source[training_indices[:, 1], :, :]

        exact = timestepping(
            psi0_training,
            0,
            0,
            params,
            sigs,
            sigt,
            N_exact,
            num_basis_exact,
            source_training,
        )[0]

        FPN, sigf = timestepping(
            psi0_training,
            1,
            NN_model,
            params,
            sigs,
            sigt,
            N,
            num_basis,
            source_training,
        )
        if obj_idx == 0:
            loss = obj_func(FPN[:, :, :, 0] - exact[:, :, :, 0])
        elif obj_idx == 1:
            loss = obj_func_time(FPN[:, :, :, :, 0] - exact[:, :, :, :, 0])
        loss.backward()
        opt.step()
        if filter_type == 1:
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
            sigf_detached = sigf.detach()
            log_metrics(
                wandb_run,
                {
                    "train/loss": loss_value,
                    "train/sqrt_loss": _sqrt_loss(loss_value),
                    "train/sigma_s_mean": sigs.detach().mean().item(),
                    "train/sigma_t_mean": sigt.detach().mean().item(),
                    "train/sigma_a_mean": (sigt - sigs).detach().mean().item(),
                    "train/filter_strength_mean": sigf_detached.mean().item(),
                    "train/filter_strength_min": sigf_detached.min().item(),
                    "train/filter_strength_max": sigf_detached.max().item(),
                    "train/epoch": l,
                },
                l,
                params,
                final=final_epoch,
            )
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
