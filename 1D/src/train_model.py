import math
import torch
import torch.optim as optim
from tqdm.auto import tqdm
from funcs_common import (
    SimpleNN,
    SimpleNN_const,
    obj_func,
    obj_func_time,
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
        siga = (sigs_max - sigs) * torch.rand(batch_size, device=device)
        sigt = sigs + siga

        exact = timestepping(
            psi0, 0, 0, params, sigs, sigt, N_exact, source, batch_size, device
        )[0]
        psi, sigf = timestepping(
            psi0,
            filter_type,
            NN_model,
            params,
            sigs,
            sigt,
            N,
            source,
            batch_size,
            device,
        )


        if obj_idx == 0:
            loss = obj_func(psi[:, :, 0] - exact[:, :, 0])
        elif obj_idx == 1:
            loss = obj_func(psi - exact[:, :, : N + 1])
        elif obj_idx == 2:
            loss = obj_func_time(psi[:, :, :, 0] - exact[:, :, :, 0])

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
            metrics = {
                "train/loss": loss_value,
                "train/sqrt_loss": _sqrt_loss(loss_value),
                "train/sigma_s_mean": sigs.detach().mean().item(),
                "train/sigma_t_mean": sigt.detach().mean().item(),
                "train/sigma_a_mean": (sigt - sigs).detach().mean().item(),
                "train/epoch": l,
            }
            if filter_type in (1, 2, 3):
                sigf_detached = sigf.detach()
                metrics.update(
                    {
                        "train/filter_strength_mean": sigf_detached.mean().item(),
                        "train/filter_strength_min": sigf_detached.min().item(),
                        "train/filter_strength_max": sigf_detached.max().item(),
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
