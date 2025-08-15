import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from funcs_common import SimpleNN, obj_func, timestepping, compute_cell_average
from IC import gaussian_testing, heaviside, bump, disc_source, vanishing_cs, disc_cs, reeds


def testing(params):

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
    device = params["device"]
    IC_idx = params["IC_idx"]
    filter_type = params["filter_type"]

    if filter_type == 0:
        model_filename = load_model(N)
        NN_model = torch.load(model_filename, map_location=torch.device(device))
        NN_model.to(device)
        NN_model.eval()

        for name, param in NN_model.named_parameters():
            if 'weight' in name and param.requires_grad:
                norm = torch.norm(param).item()
                print(f"Layer: {name} | Weight norm: {norm:.4f}")

    elif filter_type == 1:
        if N == 3:
            sigf = 27.1199
        elif N == 7:
            sigf = 16.1425
        elif N == 9:
            sigf = 10.2298
        else:
            sigf = 10
        NN_model = sigf

    with torch.no_grad():
        if IC_idx == 0:
            ic_type = "Gaussian"
            psi0_out, sigs_out, sigt_out, source_out = gaussian_testing(num_x, x_edges)
        elif IC_idx == 1:
            ic_type = "Vanishing cross-section"
            psi0_out, sigs_out, sigt_out, source_out = vanishing_cs(num_x, x_edges)
        elif IC_idx == 2:
            ic_type = "Discontinuous cross-section"
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
            ic_type = "Reed's problem"
            psi0_out, sigs_out, sigt_out, source_out, params = reeds(params)
            x = params["x"]
            xl = params["xl"]
            xr = params["xr"]
            x_edges = params["x_edges"]
            dx = params["dx"]
            num_x = params["num_x"]
            num_t = params["num_t"]

        psi0_edges = torch.zeros(batch_size, num_x + 1)
        sigs_edges = torch.zeros(batch_size, num_x + 1)
        sigt_edges = torch.zeros(batch_size, num_x + 1)
        source_edges = torch.zeros(batch_size, num_x + 1)
        psi0_edges[0, :] = psi0_out
        sigs_edges[0, :] = sigs_out
        sigt_edges[0, :] = sigt_out
        source_edges[0, :] = source_out

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
            psi0, 1, NN_model, params, sigs, sigt, N, source, batch_size, device
        )

        error0 = torch.sqrt(
            obj_func(PN - exact[:, :, 0 : N + 1]) / obj_func(exact[:, :, 0 : N + 1])
        )
        errorf = torch.sqrt(
            obj_func(FPN - exact[:, :, 0 : N + 1]) / obj_func(exact[:, :, 0 : N + 1])
        )
        flux_err0 = torch.sqrt(
            obj_func(PN[:, :, 0] - exact[:, :, 0]) / obj_func(exact[:, :, 0])
        )
        flux_errf = torch.sqrt(
            obj_func(FPN[:, :, 0] - exact[:, :, 0]) / obj_func(exact[:, :, 0])
        )

    print(torch.max(FPN[:,:,-1]))
    total_error_reduction = errorf / error0
    flux_error_reduction = flux_errf / flux_err0
    print(ic_type, "errors T =", T)

    print(
        f"TOTAL:  P{N} error = ",
        error0,
        f"FP{N} error = ",
        errorf,
        "total error_reduction = ",
        total_error_reduction,
    )
    print(
        f"FLUX:   P{N} error = ",
        flux_err0,
        f"FP{N} error = ",
        flux_errf,
        "flux error_reduction  = ",
        flux_error_reduction,
    )

    sigf  = sigf[0, :].detach().numpy()
    exact = exact[0, :, :].detach().numpy()
    PN    = PN[0, :, :].detach().numpy()
    FPN   = FPN[0, :, :].detach().numpy()


    exact_flux = np.sqrt(2) * exact[:, 0]
    PN_flux    = np.sqrt(2) * PN[:, 0]
    FPN_flux   = np.sqrt(2) * FPN[:, 0]
    


    plt.rcParams.update({"font.size": 16})
    fig, ax1 = plt.subplots()

    # Plot on the first y-axis (left side)
    (line1,) = ax1.plot(x, exact_flux, label="Exact", color="r")
    (line2,) = ax1.plot(x, PN_flux, linestyle="--", color="b", label="y_PN")
    (line3,) = ax1.plot(x, FPN_flux, linestyle="-.", color="g", label="NN Filter")

    # Set labels and limits
    ax1.set_xlim([xl, xr])
    # ax1.set_ylabel('Scalar Flux')
    # ax1.set_xlabel('x')

    # Create a second y-axis that shares the same x-axis
    ax2 = ax1.twinx()

    (line4,) = ax2.plot(x, sigf, linestyle=":", color="m", label=r"$\sigma_f$")
    # ax2.set_ylabel(r'$\sigma_f$')
    lines = [line1, line2, line3, line4]  # Combine line objects
    labels = [line.get_label() for line in lines]  # Get labels for the lines
    # ax1.legend(lines, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.15))
    plt.show()
    #plt.savefig("trained_models/plot.png")

    # fig, ax = plt.subplots()  # Create figure and axes

    # for j in range(N+1):
    #     ax.plot(x, FPN[:, j], label=f'{j}')

    # ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))  # Puts legend outside top-right
    # # ax.legend()
    # ax.set_xlim([xl, xr])
    # ax.set_title('FPN Moments')
    # ax.set_xlabel('x')
    # plt.show()

    return 0


def load_model(N):
    valid_N = {3, 7, 9}
    if N not in valid_N:
        raise ValueError(f"Invalid value for N: {N}. Expected one of {valid_N}.")

    filename = f"trained_models/model_N{N}.pth"

    return filename
