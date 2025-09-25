import torch
import torch.optim as optim
from funcs_common import (
    SimpleNN,
    obj_func,
    timestepping,
    compute_cell_average,
    SimpleNN_const,
)
from IC import gaussian_training, heaviside, bump, disc_source
import wandb
import matplotlib.pyplot as plt


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

    if params["const_net"]:
        NN_model = SimpleNN_const(num_features, num_hidden, N)
    else:
        NN_model = SimpleNN(num_features, num_hidden, N)
    NN_model = NN_model.to(device)
    if GD_optimizer == "SGD":
        opt = optim.SGD(
            NN_model.parameters(), lr=learning_rate, momentum=momentum_factor
        )
    elif GD_optimizer == "Adam":
        opt = optim.Adam(
            NN_model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, num_epochs, eta_min=1e-5)
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

    psi0 = compute_cell_average(psi0_edges, batch_size, num_x).to(device)
    source = compute_cell_average(source_edges, batch_size, num_x).to(device)

    # Initialize wandb
    wandb_init = True
    if wandb_init:
        wandb.init(
            project=f"1D_training_T_half",
            config=params,
        )
        wandb.watch(NN_model, log_freq=1)

    for l in range(num_epochs):
        opt.zero_grad()

        # sample random ic and sources
        perm = torch.randperm(source.shape[0], device=device)

        sigs = torch.rand(batch_size, device=device) * sigs_max
        sigt = (sigs_max - sigs) * torch.rand(batch_size, device=device)

        exact = timestepping(
            psi0[perm, :],
            0,
            0,
            params,
            sigs,
            sigt,
            N_exact,
            source[perm, :],
            batch_size,
            device,
        )[0]
        psi = timestepping(
            psi0[perm, :],
            filter_type,
            NN_model,
            params,
            sigs,
            sigt,
            N,
            source[perm, :],
            batch_size,
            device,
        )[0]

        # for k in range(4):
        #     plt.plot(psi[k, :, 0].detach().cpu().numpy())
        #     plt.plot(exact[k, :, 0].detach().cpu().numpy(), linestyle="dashed")
        # plt.xlabel("x")
        # plt.ylabel("psi")
        # plt.title("psi at epoch {}".format(l))
        # plt.savefig("tmp_{}.png".format(l))
        # plt.clf()
        loss = (dx * (psi[:, :, 0] - exact[:, :, 0]).pow(2).sum(dim=1)).mean()
        # err = psi[:, :, 0].squeeze(-1) - exact[:, :, 0].squeeze(-1)  # shape (B,N)
        # num = (dx * err.pow(2).sum(dim=1)).mean()  # scalar
        # den = (dx * exact[:, :, 0].squeeze(-1).pow(2).sum(dim=1)).mean()
        # loss = num  # / (den + 1e-12)

        loss.backward()

        total_norm = 0.0
        for p in NN_model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm**0.5

        opt.step()
        scheduler.step()

        print("Loss at epoch", l, ":", loss.item(), "grad norm:", total_norm)
        if torch.isnan(loss):
            print("NaN loss detected. Stopping training.")
            break

        if wandb_init:
            wandb.log({"loss": loss.item()})

        # total_norm = 0.0
        # for p in NN_model.parameters():
        #     if p.grad is not None:
        #         param_norm = p.grad.data.norm(2)  # L2 norm of this parameter's gradient
        #         total_norm += param_norm.item() ** 2
        # total_norm = total_norm ** 0.5
        # print(f"Gradient norm: {total_norm:.4e}")

    wandb.finish()
    return NN_model
