import torch
import numpy as np
import torch.nn as nn


class SimpleNN_const(nn.Module):

    def __init__(self, num_features, num_hidden, N):
        super(SimpleNN_const, self).__init__()

        self.const = nn.Parameter(torch.ones(1, 1), requires_grad=True)

    def forward(self, x):
        original_shape = x.shape

        output_shape = [original_shape[0], original_shape[1], 1]
        const_broadcasted = self.const.expand(
            output_shape[0], output_shape[1], output_shape[2]
        ).contiguous()
        return const_broadcasted


class SimpleNN(nn.Module):
    def __init__(self, num_features, num_hidden, N):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(num_features, num_hidden)  # (inputs,hidden)
        self.hidden2 = nn.Linear(num_hidden, num_hidden)  # (inputs,hidden)
        self.output = nn.Linear(num_hidden, 1)  # (hidden,output)
        self.N = N

    def forward(self, x):
        original_shape = x.shape
        x = torch.flatten(x, start_dim=0, end_dim=1)
        x = torch.relu(self.hidden(x))  # Activation hidden layer
        # if self.N > 3:
        #    x = torch.relu(self.hidden2(x)) + x  # Activation hidden layer
        x = torch.relu(self.output(x))  # Activation output layer
        output_shape = [original_shape[0], original_shape[1], 1]
        return x.reshape(output_shape)  # torch.ones(output_shape)  #


# class SimpleNN(nn.Module):
#     def __init__(self, num_features, num_hidden):
#         super(SimpleNN, self).__init__()
#         self.hidden1 = nn.Linear(num_features, num_hidden)  # (inputs,hidden)
#         # self.hidden2 = nn.Linear(num_hidden, num_hidden)  # (inputs,hidden)
#         # self.hidden3 = nn.Linear(num_hidden, num_hidden)  # (inputs,hidden)
#         # self.hidden4 = nn.Linear(num_hidden, num_hidden)  # (inputs,hidden)

#         #self.bn1 = nn.LayerNorm(num_features)
#         self.bn2 = nn.LayerNorm(num_hidden)
#         # self.bn3 = nn.LayerNorm(num_hidden)
#         # self.bn4 = nn.LayerNorm(num_hidden)
#         # self.bn5 = nn.LayerNorm(num_hidden)
#         self.output = nn.Linear(num_hidden, 1)  # (hidden,output)
#         print(self)

#     def forward(self, x):
#         # print("Input shape:", x.shape)  # Debugging line
#         original_shape = x.shape
#         x = torch.flatten(x, start_dim=0, end_dim=1)
#         # print("Flattened input shape:", x.shape)  # Debugging line
#         #x = self.bn1(x)
#         x = torch.tanh(self.hidden1(x))      # Activation hidden layer
#         x = self.bn2(x)
#         #x = torch.tanh(self.hidden2(x)) + x  # Activation hidden layer
#         # x = self.bn3(x)
#         # x = torch.tanh(self.hidden3(x)) + x  # Activation hidden layer
#         # x = self.bn4(x)
#         # x = torch.tanh(self.hidden4(x)) + x  # Activation hidden layer
#         # x = self.bn5(x)
#         x = torch.relu(self.output(x))  # Activation output layer
#         output_shape = [original_shape[0], original_shape[1], 1]
#         return x.reshape(output_shape)


def timestepping(
    y0, filter_type, NN_model, params, sigs, sigt, N, source, batch_size, device
):
    dt = params["dt"]
    dx = params["dx"]
    tt_flag = params["tt_flag"]
    IC_idx = params["IC_idx"]

    num_x = params["num_x"]
    num_t = params["num_t"]
    filter_func = params["filter"]

    # CONSTRUCT A vector: does not need updating
    a = torch.zeros(N)

    for n in range(1, N + 1):
        a[n - 1] = n / np.sqrt((2 * n - 1) * (2 * n + 1))
    A = torch.diag(a, 1) + torch.diag(a, -1)

    eigA, V = torch.linalg.eig(A)
    eigA = torch.real(eigA)
    V = torch.real(V)
    absA = torch.matmul(torch.matmul(V, torch.diag(torch.abs(eigA))), V.T)

    B = -absA / dx
    C = 0.5 * (absA - A) / dx
    D = 0.5 * (absA + A) / dx

    B = B.to(device)
    C = C.to(device)
    D = D.to(device)
    source = source.to(device)
    filter_func = filter_func.to(device)
    A = A.to(device)
    absA = absA.to(device)
    sigt = sigt.to(device)
    sigs = sigs.to(device)

    y_prev = torch.zeros([batch_size, num_x, N + 1], device=device)
    y_prev[:, :, 0] = y0
    y = y_prev
    source_in = source[:, :, None]

    if tt_flag == 0:
        sigt_in = sigt[:, None, None]
        sigs_in = sigs[:, None, None]
    if tt_flag == 1:
        sigt_in = sigt[:, :, None]
        sigs_in = sigs[:, :, None]

    for k in range(1, num_t + 1):

        y1_update = PN_update(
            params,
            y_prev,
            A,
            absA,
            B,
            C,
            D,
            N,
            source,
            filter_type,
            NN_model,
            source_in,
            sigt_in,
            sigs_in,
        )[0]
        y1 = y_prev + dt * y1_update

        # boundary conditions for Reeds problem: reflecting at x = 0 and vacauum at x = 8
        if IC_idx == 6:
            y1 = reeds_BC(y1, N)

        y2_update, sigf = PN_update(
            params,
            y1,
            A,
            absA,
            B,
            C,
            D,
            N,
            source,
            filter_type,
            NN_model,
            source_in,
            sigt_in,
            sigs_in,
        )
        y = y + 0.5 * dt * (y1_update + y2_update)

        # boundary conditions for Reeds problem: reflecting at x = 0 and vacauum at x = 8
        if IC_idx == 6:
            y = reeds_BC(y, N)
        y_prev = y

    return y, sigf


def PN_update(
    params,
    y_prev,
    A,
    absA,
    B,
    C,
    D,
    N,
    source,
    filter_type,
    NN_model,
    source_in,
    sigt,
    sigs,
):
    batch_size = params["batch_size"]
    device = params["device"]
    IC_idx = params["IC_idx"]
    num_x = params["num_x"]
    filter_func = params["filter"]

    flux_limiter = torch.zeros([batch_size, num_x + 2, N + 1], device=device)
    y_expand = torch.zeros([batch_size, num_x + 2, N + 1], device=device)

    y_expand[:, 1 : num_x + 1, :] = y_prev
    if IC_idx != 6:
        y_expand[:, 0, :] = y_prev[:, num_x - 1, :]
        y_expand[:, num_x + 1, :] = y_prev[:, 0, :]

    flux_limiter[:, 1 : num_x + 1, :] = minmod(
        y_expand[:, 2 : num_x + 2, :] - y_expand[:, 1 : num_x + 1, :],
        y_expand[:, 1 : num_x + 1, :] - y_expand[:, 0:num_x, :],
    )

    flux_limiter[:, 0, :] = flux_limiter[:, num_x + 1, :]
    flux_limiter[:, num_x + 1, :] = flux_limiter[:, 1, :]

    B_y = torch.matmul(y_expand[:, 1 : num_x + 1, :], B.T)
    C_y = torch.matmul(y_expand[:, 2 : num_x + 2, :], C.T)
    D_y = torch.matmul(y_expand[:, :num_x, :], D.T)

    flux_diff1 = (
        flux_limiter[:, :num_x, :]
        - 2 * flux_limiter[:, 1 : num_x + 1, :]
        + flux_limiter[:, 2 : num_x + 2, :]
    )
    flux_diff2 = flux_limiter[:, 2 : num_x + 2, :] - flux_limiter[:, :num_x, :]
    flux1 = 0.25 * torch.matmul(flux_diff1, A.T)
    flux2 = 0.25 * torch.matmul(flux_diff2, absA.T)
    A_Dy = B_y + C_y + D_y - flux1 + flux2

    sigf = torch.zeros([batch_size, num_x], device=device)

    if filter_type in (1, 2):
        yflux = y_prev[:, :, 0]
        yflux = yflux[:, :, None]
        inputs = preprocess_features(
            A_Dy, sigt * y_prev, sigs * yflux, source_in, filter_type
        )
        sigf = NN_model(inputs).squeeze(-1)

    if filter_type == 3:
        sigf0 = NN_model()
        sigf = sigf0 * torch.ones(batch_size, num_x, device=device)

    if IC_idx == 6:
        sigf[:, 0] = sigf[:, 1]
        sigf[:, num_x - 1] = sigf[:, num_x - 2]

    y_update = torch.zeros([batch_size, num_x, N + 1], device=device)

    sigt_y = sigt * y_expand[:, 1 : num_x + 1, :]
    y_update = A_Dy - sigt_y

    if filter_type in (1, 2, 3):

        y_update = y_update - sigf[:, :, None] * y_prev * filter_func

    y_update[:, :, 0] = (
        y_update[:, :, 0] + sigs[:, :, 0] * y_expand[:, 1 : num_x + 1, 0] + source
    )

    return y_update, sigf


def preprocess_features(A_Dy, sigt_y, scattering, source, filter_type):
    scattering_NN = NN_normalization(torch.abs(scattering))
    source_NN = NN_normalization(torch.abs(source))

    if filter_type == 1:
        A_Dy_NN = NN_normalization(torch.abs(A_Dy))
        sigt_y_NN = NN_normalization(torch.abs(sigt_y))

    elif filter_type == 2:
        A_Dy_temp = A_Dy.clone()
        y_temp = sigt_y.clone()

        A_Dy[:, :, 1::2] = torch.abs(A_Dy_temp[:, :, 1::2])
        sigt_y[:, :, 1::2] = torch.abs(y_temp[:, :, 1::2])

        sigt_y_NN = NN_normalization(sigt_y)
        A_Dy_NN = NN_normalization(A_Dy)

    inputs = torch.cat((A_Dy_NN, sigt_y_NN, scattering_NN, source_NN), dim=-1)
    return inputs


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


def obj_func(z, dx):
    return torch.mean(z**2)
    # return (dx * z.pow(2).sum(dim=1)).mean()


def compute_cell_average(f, batch_size, num_x):
    f_average = torch.zeros(batch_size, num_x)
    for m in range(0, num_x):
        f_average[:, m] = 0.5 * (f[:, m] + f[:, m + 1])

    return f_average


def reeds_BC(z, N):
    for n in range(0, N, 2):
        z[:, 0, n] = z[:, 1, n]
    for n in range(1, N + 1, 2):
        z[:, 0, n] = -z[:, 1, n]
    return z
