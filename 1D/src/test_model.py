import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from funcs_common import SimpleNN, obj_func, timestepping, compute_cell_average
from params_common import model_tag_from_params, tagged_model_path
from IC import (
    gaussian_testing,
    heaviside,
    bump,
    disc_source,
    vanishing_cs,
    disc_cs,
    reeds,
)


def _ensure_finite(name, value):
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains non-finite values")


def _ensure_positive_finite(name, value):
    _ensure_finite(name, value)
    if torch.any(value <= 0):
        value_text = value.detach().cpu().numpy()
        raise FloatingPointError(f"{name} must be positive; got {value_text}")


def _as_float(value):
    tensor = torch.as_tensor(value).detach().cpu()
    if tensor.numel() != 1:
        raise ValueError(f"Expected a scalar value; got shape {tuple(tensor.shape)}")
    return tensor.item()


def _print_error_summary(
    ic_type,
    T,
    N,
    error0,
    errorf,
    total_error_reduction,
    flux_err0,
    flux_errf,
    flux_error_reduction,
):
    print(f"{ic_type} | T={float(T):.3g} | N={N}")
    print(f"  {'metric':<7} {f'P{N} err':>10} {f'FP{N} err':>10} {'FP/P':>10}")
    print(
        f"  {'total':<7} "
        f"{_as_float(error0):>10.4f} "
        f"{_as_float(errorf):>10.4f} "
        f"{_as_float(total_error_reduction):>10.4f}"
    )
    print(
        f"  {'flux':<7} "
        f"{_as_float(flux_err0):>10.4f} "
        f"{_as_float(flux_errf):>10.4f} "
        f"{_as_float(flux_error_reduction):>10.4f}"
    )


def testing(params, model_idx=0):

    num_x = params["num_x"]
    num_t = params["num_t"]
    N_exact = params["N_exact"]
    N = params["N"]
    T = params["T"]
    batch_size = params["batch_size"]
    x = params["x"]
    xl = params["xl"]
    xr = params["xr"]
    x_edges = params["x_edges"]
    dx = params["dx"]
    device = torch.device(params["device"])
    IC_idx = params["IC_idx"]
    filter_type = params["filter_type"]
    show_plot = params["show_plot"]
    model_tag = model_tag_from_params(params)

    model_filename = load_model(N, model_idx, filter_type, model_tag)
    if not os.path.exists(model_filename):
        raise FileNotFoundError(
            f"Missing checkpoint for N={N}, model_idx={model_idx}, "
            f"filter_type={filter_type}, model_tag={model_tag}: {model_filename}"
        )
    NN_model = torch.load(
        model_filename, map_location=torch.device(device), weights_only=False
    )
    NN_model.to(device)
    NN_model.eval()

    # for name, param in NN_model.named_parameters():
    #     if 'weight' in name and param.requires_grad:
    #         norm = torch.norm(param).item()
    #         print(f"Layer: {name} | Weight norm: {norm:.4f}")

    # elif filter_type == 3:
    #     if N == 3:
    #         sigf = 27.1199
    #     elif N == 7:
    #         sigf = 16.1425
    #     elif N == 9:
    #         sigf = 10.2298
    #     else:
    #         sigf = 10
    #     NN_model = sigf

    with torch.no_grad():
        if IC_idx == 0:
            ic_type = "Gaussian"
            psi0_out, sigs_out, sigt_out, source_out = gaussian_testing(num_x, x_edges)
        elif IC_idx == 1:
            ic_type = "Vanishing_cross_section"
            psi0_out, sigs_out, sigt_out, source_out = vanishing_cs(num_x, x_edges)
        elif IC_idx == 2:
            ic_type = "Discontinuous_cross_section"
            psi0_out, sigs_out, sigt_out, source_out = disc_cs(num_x, x_edges)
        elif IC_idx == 3:
            ic_type = "Step"
            psi0_out, sigs_out, sigt_out, source_out = heaviside(num_x, x_edges)
        elif IC_idx == 4:
            ic_type = "Bump"
            psi0_out, sigs_out, sigt_out, source_out = bump(num_x, x_edges)
        elif IC_idx == 5:
            ic_type = "Discontinuous source"
            psi0_out, sigs_out, sigt_out, source_out = disc_source(num_x, x_edges)
        elif IC_idx == 6:
            ic_type = "Reeds"
            psi0_out, sigs_out, sigt_out, source_out, params = reeds(params)
            x = params["x"]
            xl = params["xl"]
            xr = params["xr"]
            x_edges = params["x_edges"]
            dx = params["dx"]
            num_x = params["num_x"]
            num_t = params["num_t"]

        psi0_edges = torch.zeros(batch_size, num_x + 1, device=device)
        sigs_edges = torch.zeros(batch_size, num_x + 1, device=device)
        sigt_edges = torch.zeros(batch_size, num_x + 1, device=device)
        source_edges = torch.zeros(batch_size, num_x + 1, device=device)
        psi0_edges[0, :] = psi0_out.to(device)
        sigs_edges[0, :] = sigs_out.to(device)
        sigt_edges[0, :] = sigt_out.to(device)
        source_edges[0, :] = source_out.to(device)

        psi0 = compute_cell_average(psi0_edges, batch_size, num_x)
        sigs = compute_cell_average(
            sigs_edges,
            batch_size,
            num_x,
        )
        sigt = compute_cell_average(sigt_edges, batch_size, num_x)
        source = compute_cell_average(source_edges, batch_size, num_x)

        exact = timestepping(
            psi0, 0, 0, params, sigs, sigt, N_exact, source, batch_size, device
        )[0]

        PN = timestepping(
            psi0, 0, 0, params, sigs, sigt, N, source, batch_size, device
        )[0]

        FPN, sigf = timestepping(
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

        _ensure_finite("exact solution", exact)
        _ensure_finite(f"P{N} solution", PN)
        _ensure_finite(f"FP{N} solution", FPN)
        _ensure_finite("filter strength", sigf)

        total_den = obj_func(exact[:, :, 0 : N + 1])
        flux_den = obj_func(exact[:, :, 0])
        _ensure_positive_finite("total exact norm", total_den)
        _ensure_positive_finite("flux exact norm", flux_den)

        error0 = torch.sqrt(obj_func(PN - exact[:, :, 0 : N + 1]) / total_den)
        errorf = torch.sqrt(obj_func(FPN - exact[:, :, 0 : N + 1]) / total_den)
        flux_err0 = torch.sqrt(obj_func(PN[:, :, 0] - exact[:, :, 0]) / flux_den)
        flux_errf = torch.sqrt(obj_func(FPN[:, :, 0] - exact[:, :, 0]) / flux_den)

        _ensure_positive_finite(f"P{N} total error", error0)
        _ensure_finite(f"FP{N} total error", errorf)
        _ensure_positive_finite(f"P{N} flux error", flux_err0)
        _ensure_finite(f"FP{N} flux error", flux_errf)

    total_error_reduction = errorf / error0
    flux_error_reduction = flux_errf / flux_err0
    _ensure_finite("total error reduction", total_error_reduction)
    _ensure_finite("flux error reduction", flux_error_reduction)
    if params.get("print_results", True):
        _print_error_summary(
            ic_type,
            T,
            N,
            error0,
            errorf,
            total_error_reduction,
            flux_err0,
            flux_errf,
            flux_error_reduction,
        )

    sigf = sigf.detach().cpu().numpy()
    sigf = sigf[0, :]
    exact = exact[0, :, :].detach().cpu().numpy()
    PN = PN[0, :, :].detach().cpu().numpy()
    FPN = FPN[0, :, :].detach().cpu().numpy()

    exact_flux = np.sqrt(2) * exact[:, 0]
    PN_flux = np.sqrt(2) * PN[:, 0]
    FPN_flux = np.sqrt(2) * FPN[:, 0]

    output_dir = os.path.join("results", ic_type)
    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams.update({"font.size": 16})

    fig, ax1 = plt.subplots(figsize=(6, 5), constrained_layout=True)

    # Plot on the first y-axis (left side)
    (line1,) = ax1.plot(x, exact_flux, label="Exact", color="r")
    (line2,) = ax1.plot(x, PN_flux, linestyle="--", color="b", label="y_PN")
    (line3,) = ax1.plot(x, FPN_flux, linestyle="-.", color="g", label="NN Filter")

    # Set labels and limits
    ax1.set_xlim([xl, xr])
    # ax1.set_ylabel('Scalar Flux')
    ax1.set_xlabel("z", fontsize=18)

    # Create a second y-axis that shares the same x-axis
    ax2 = ax1.twinx()

    (line4,) = ax2.plot(x, sigf, linestyle=":", color="m", label=r"$\sigma_f$")
    # ax2.set_ylabel(r'$\sigma_f$')
    lines = [line1, line2, line3, line4]  # Combine line objects
    labels = [line.get_label() for line in lines]  # Get labels for the lines
    # ax1.legend(lines, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.15))

    if show_plot == 1:
        plt.show()

    if IC_idx == 6:
        T_int = int(T)
    else:
        T_int = int(10 * T)
    # format T with a fixed number of decimals (say 3)
    filename = os.path.join(output_dir, f"P{N}_t{T_int}.png")

    plt.savefig(filename, bbox_inches="tight", dpi=300)

    # fig, ax = plt.subplots()  # Create figure and axes

    # for j in range(N+1):
    #     ax.plot(x, FPN[:, j], label=f'{j}')

    # ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))  # Puts legend outside top-right
    # # ax.legend()
    # ax.set_xlim([xl, xr])
    # ax.set_title('FPN Moments')
    # ax.set_xlabel('x')
    # plt.show()
    plt.close()
    return flux_error_reduction


def load_model(N, model_idx, filter_type, model_tag=None):
    valid_N = {3, 7, 9}
    if N not in valid_N:
        raise ValueError(f"Invalid value for N: {N}. Expected one of {valid_N}.")
    return tagged_model_path(N, model_idx, filter_type, model_tag)
