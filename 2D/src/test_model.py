import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from IC import gaussian_testing, step, disc_source, bump, hat, holhraum, lattice
from funcs_common import obj_func, timestepping, compute_cell_average, rotation_test

# from training_sources import pulse_source, two_rect_source, frame_source


def testing(params):

    num_x = params["num_x"]
    num_y = params["num_y"]
    N_exact = params["N_exact"]
    N = params["N"]
    num_basis = params["num_basis"]
    num_basis_exact = params["num_basis_exact"]
    IC_idx = params["IC_idx"]
    x = params["x"]
    y = params["y"]
    xl = params["xl"]
    xr = params["xr"]
    x_edges = params["x_edges"]
    y_edges = params["y_edges"]
    batch_size = params["batch_size"]
    plot_idx = params["plot_idx"]
    device = params["device"]
    T = params["T"]
    filter_type = params["filter_type"]
    sigf_const = params["sigf_const"]
    show_plots = params["show_plots"]
    show_sym_errors = params["show_sym_errors"]
    show_slices = params["show_slices"]
    if filter_type == 0:
        model_filename = load_model(N)
        NN_model = torch.load(model_filename, map_location=torch.device(device), weights_only= False)
        NN_model.to(device)
        NN_model.eval()

    elif filter_type == 1:
        if N == 3:
            sigf = 31.75
        elif N == 5:
            sigf = 27.27
        elif N == 7:
            sigf = 16.52
        elif N == 9:
            sigf = 13.62
        else:
            sigf = 10
        NN_model = sigf

    elif filter_type == 2:
        sigf = sigf_const
        NN_model = sigf

    with torch.no_grad():
        if IC_idx == 0:
            ic_type = "Linesource"
            psi0_out, source_out, sigs_out, sigt_out = gaussian_testing(
                num_x, num_y, x_edges, y_edges
            )
        elif IC_idx == 1:
            ic_type = "Step"
            psi0_out, source_out, sigs_out, sigt_out = step(
                num_x, num_y, x_edges, y_edges
            )
        elif IC_idx == 2:
            ic_type = "Disc  Source"
            psi0_out, source_out, sigs_out, sigt_out = disc_source(
                num_x, num_y, x_edges, y_edges
            )
        elif IC_idx == 3:
            ic_type = "Bump"
            psi0_out, source_out, sigs_out, sigt_out = bump(
                num_x, num_y, x_edges, y_edges
            )
        elif IC_idx == 4:
            ic_type = "Hat"
            psi0_out, source_out, sigs_out, sigt_out = hat(
                num_x, num_y, x_edges, y_edges
            )
        elif IC_idx == 5:
            ic_type = "Holhraum"
            psi0_out, source_out, sigs_out, sigt_out, params = holhraum(params)
            # NOTE: holhraum overwrites parameters defined in parameter file
            num_x = params["num_x"]
            num_y = params["num_y"]
            x = params["x"]
            xl = params["xl"]
            xr = params["xr"]
            y = params["y"]
            T = params["T"]
        elif IC_idx == 6:
            ic_type = "Lattice"
            psi0_out, source_out, sigs_out, sigt_out, params = lattice(params)
            # NOTE: lattice overwrites parameters defined in parameter file
            num_x = params["num_x"]
            num_y = params["num_y"]
            x = params["x"]
            xl = params["xl"]
            xr = params["xr"]
            y = params["y"]
            T = params["T"]

        psi0_edges = torch.zeros(batch_size, num_y + 1, num_x + 1, dtype=torch.float32)
        sigs_edges = torch.zeros(batch_size, num_y + 1, num_x + 1, dtype=torch.float32)
        sigt_edges = torch.zeros(batch_size, num_y + 1, num_x + 1, dtype=torch.float32)
        source_edges = torch.zeros(
            batch_size, num_y + 1, num_x + 1, dtype=torch.float32
        )

        psi0_edges[0, :, :] = psi0_out
        sigs_edges[0, :, :] = sigs_out
        sigt_edges[0, :, :] = sigt_out
        source_edges[0, :, :] = source_out

        psi0 = compute_cell_average(psi0_edges, num_x, num_y, batch_size)
        sigs = compute_cell_average(sigs_edges, num_x, num_y, batch_size)
        sigt = compute_cell_average(sigt_edges, num_x, num_y, batch_size)
        source = compute_cell_average(source_edges, num_x, num_y, batch_size)

        # if IC_idx == 0:
        #     exact_np = np.load("linesource_37.npy")
        #     exact = torch.zeros(batch_size, num_y, num_x, num_basis_exact)
        #     exact[0, :, :, :] = torch.from_numpy(exact_np)

        # elif IC_idx == 6:
        #     if T == 1.6:
        #         exact_np = np.load("lattice_37_T16.npy")
        #     elif T == 3.2:
        #         exact_np = np.load("lattice_37_T32.npy")
        #     exact = torch.zeros(batch_size, num_y, num_x, num_basis_exact)
        #     exact[0, :, :, :] = torch.from_numpy(exact_np)

        # else:
        exact = timestepping(
            psi0, 0, 0, params, sigs, sigt, N_exact, num_basis_exact, source
        )[0]
        # np.save("P37",  exact)

        PN = timestepping(psi0, 0, 0, params, sigs, sigt, N, num_basis, source)[0]
        FPN, sigf = timestepping(
            psi0, 1, NN_model, params, sigs, sigt, N, num_basis, source
        )

        error0 = torch.sqrt(
            obj_func(PN - exact[:, :, :, :num_basis])
            / obj_func(exact[:, :, :, :num_basis])
        )
        errorf = torch.sqrt(
            obj_func(FPN - exact[:, :, :, :num_basis])
            / obj_func(exact[:, :, :, :num_basis])
        )
        flux_err0 = torch.sqrt(
            obj_func(PN[:, :, :, 0] - exact[:, :, :, 0]) / obj_func(exact[:, :, :, 0])
        )
        flux_errf = torch.sqrt(
            obj_func(FPN[:, :, :, 0] - exact[:, :, :, 0]) / obj_func(exact[:, :, :, 0])
        )

    total_error_reduction = errorf / error0
    flux_error_reduction = flux_errf / flux_err0
    print(ic_type, "errors T = ", T)
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

    sigf = sigf.detach().numpy()
    exact = exact[0, :, :, 0].detach().numpy()
    PN = PN[0, :, :, 0].detach().numpy()
    FPN = FPN[0, :, :, 0].detach().numpy()

    if IC_idx == 6:
        floor_value = 1e-7
        exact = np.where(exact < floor_value, floor_value, exact)
        PN = np.where(PN < floor_value, floor_value, PN)
        FPN = np.where(FPN < floor_value, floor_value, FPN)
        exact = np.log10(exact)
        PN = np.log10(PN)
        FPN = np.log10(FPN)

    cmap = mpl.cm.inferno
    cmap_sigf = mpl.cm.plasma
    levels = 100

    if show_sym_errors == 1:
        rot_error = rotation_test(FPN)
        print("Rotation error =", rot_error)

        PN_sym = PN - np.flip(PN, axis=0)
        print("Max symmetry error =", np.max(PN_sym))

    fig_PN, ax_PN = plt.subplots(constrained_layout=True)
    plt.rcParams.update({"font.size": 16})
    contour_PN = ax_PN.contourf(y, x, PN, levels, cmap=cmap)
    # ax_PN.set_title(f"$P_{{{N}}}$")
    ax_PN.tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False
    )
    ax_PN.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
    fig_PN.colorbar(contour_PN, ax=ax_PN, orientation="vertical", shrink=0.8)

    fig_FPN, ax_FPN = plt.subplots(constrained_layout=True)
    contour_FPN = ax_FPN.contourf(y, x, FPN, levels, cmap=cmap)
    # ax_FPN.set_title(f"$FP_{{{N}}}$")
    ax_FPN.tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False
    )
    ax_FPN.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
    fig_FPN.colorbar(contour_FPN, ax=ax_FPN, orientation="vertical", shrink=0.8)

    fig_exact, ax_exact = plt.subplots(constrained_layout=True)
    plt.rcParams.update({"font.size": 16})
    contour_exact = ax_exact.contourf(y, x, exact, levels, cmap=cmap)
    # ax_exact.set_title(f"$P_{{{N_exact}}}$")
    ax_exact.tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False
    )
    ax_exact.tick_params(
        axis="y", which="both", left=False, right=False, labelleft=False
    )
    fig_exact.colorbar(contour_exact, ax=ax_exact, orientation="vertical", shrink=0.8)

    if show_slices == 1:
        fig_slice0, ax_slice0 = plt.subplots()
        plt.xlim(xl, xr)
        (line1,) = ax_slice0.plot(
            x, exact[plot_idx, :], color="r", label=f"$P_{{{N_exact}}}$"
        )
        (line2,) = ax_slice0.plot(
            x, PN[plot_idx, :], linestyle="--", color="b", label=f"$P_{{{N}}}$"
        )
        (line3,) = ax_slice0.plot(
            x, FPN[plot_idx, :], linestyle="-.", color="g", label=f"$FP_{{{N}}}$"
        )
        ax_slice0_sigf = ax_slice0.twinx()
        (line4,) = ax_slice0_sigf.plot(
            x, sigf[plot_idx, :], linestyle=":", color="m", label=r"$\sigma_f$"
        )
        lines = [line1, line2, line3, line4]
        # lines = [line1, line2, line3]
        labels = [line.get_label() for line in lines]
        ax_slice0.tick_params(axis="x", bottom=True, labelbottom=True)
        #ax_slice0.legend(lines, labels, bbox_to_anchor=(1.08, 1.15), ncol=4, frameon=False)

        fig_slice45, ax_slice45 = plt.subplots()
        xl45 = -np.sqrt(2)
        xr45 =  np.sqrt(2)
        plt.xlim(xl45, xr45)
        exact45 = exact[np.arange(exact.shape[0]), np.arange(exact.shape[0])]
        PN45   = PN[np.arange(PN.shape[0]), np.arange(PN.shape[0])]
        FPN45  = FPN[np.arange(FPN.shape[0]), np.arange(FPN.shape[0])]
        sigf45 = sigf[np.arange(FPN.shape[0]), np.arange(FPN.shape[0])]
        x45      = torch.linspace(xl45, xr45, num_x)

        (line1,) = ax_slice45.plot(x45, exact45, color="r", label=f"$P_{{{N_exact}}}$")
        (line2,) = ax_slice45.plot(
            x45, PN45, linestyle="--", color="b", label=f"$P_{{{N}}}$"
        )
        (line3,) = ax_slice45.plot(
            x45, FPN45, linestyle="-.", color="g", label=f"$FP_{{{N}}}$"
        )
        ax_slice45_sigf = ax_slice45.twinx()
        (line4,) = ax_slice45_sigf.plot(
            x45, sigf45, linestyle=":", color="m", label=r"$\sigma_f$"
        )
        lines = [line1, line2, line3, line4]
        # lines = [line1, line2, line3]
        labels = [line.get_label() for line in lines]
        ax_slice45.tick_params(axis="x", bottom=True, labelbottom=True)
        #ax_slice45.legend(lines, labels, bbox_to_anchor=(1.08, 1.15), ncol=4, frameon=False)

    fig_sig, ax_sig = plt.subplots(constrained_layout=True)
    contour_sig = ax_sig.contourf(y, x, sigf, levels, cmap=cmap_sigf)
    ax_sig.set_title(fr"$\sigma_f$, $t = {T}$")
    ax_sig.tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False
    )
    ax_sig.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
    fig_sig.colorbar(contour_sig, ax=ax_sig, orientation="vertical", shrink=0.8)

    if show_plots == 1:
        plt.show()


def load_model(N):
    valid_N = {3, 5, 7, 9}
    if N not in valid_N:
        raise ValueError(f"Invalid value for N: {N}. Expected one of {valid_N}.")

    filename = f"trained_models/model_N{N}.pth"

    return filename
