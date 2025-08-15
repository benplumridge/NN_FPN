import torch
import torch.optim as optim
from funcs_common import SimpleNN, obj_func, timestepping, compute_cell_average
from IC import gaussian_training, heaviside, bump, disc_source


def training(params):

    num_x = params["num_x"]
    N_exact = params["N_exact"]
    N = params["N"]
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


    NN_model = SimpleNN(num_features, num_hidden)
    NN_model = NN_model.to(device)
    if GD_optimizer == "SGD":
        opt = optim.SGD(
            NN_model.parameters(), lr=learning_rate, momentum=momentum_factor
        )
    elif GD_optimizer == "Adam":
        opt = optim.Adam(NN_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    psi0_edges = torch.zeros([batch_size, num_x + 1], device=device)
    source_edges = torch.zeros([batch_size, num_x + 1], device=device)
    reg1 = torch.arange(0, round(batch_size / num_IC))
    reg2 = torch.arange(round(batch_size / num_IC), round(2 * batch_size / num_IC))
    reg3 = torch.arange(round(2 * batch_size / num_IC), round(3 * batch_size / num_IC))
    reg4 = torch.arange(round(3 * batch_size / num_IC), round(4 * batch_size / num_IC))

    psi0_edges[reg1, :] = gaussian_training(num_x, x_edges)[0].to(device)
    psi0_edges[reg2, :] = heaviside(num_x, x_edges)[0].to(device)
    psi0_edges[reg3, :] = bump(num_x, x_edges)[0].to(device)
    disc_source_output = disc_source(num_x, x_edges)
    psi0_edges[reg4, :] = disc_source_output[0].to(device)
    source_edges[reg4, :] = disc_source_output[3].to(device)

    psi0 = compute_cell_average(psi0_edges, batch_size, num_x)
    source = compute_cell_average(source_edges, batch_size, num_x)

    for l in range(num_epochs):
        opt.zero_grad()
        sigs = torch.rand(batch_size) * sigs_max
        sigt = (1 - sigs) * torch.rand(batch_size)

        exact = timestepping(
            psi0, 0, 0, params, sigs, sigt, N_exact, source, batch_size, device
        )[0]
        psi = timestepping(
            psi0, 1, NN_model, params, sigs, sigt, N, source, batch_size, device
        )[0]

        psi = psi.to("cpu")
        exact = exact.to("cpu")
        loss = obj_func(psi[:, :, 0] - exact[:, :, 0])
        loss.backward()
        opt.step()

        print("Loss at epoch", l, ":", loss.item())
        if torch.isnan(loss):
            print("NaN loss detected. Stopping training.")
            break

    return NN_model
