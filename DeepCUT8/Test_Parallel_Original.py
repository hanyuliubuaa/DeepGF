import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from Lorenz.Dynamics import f
from Lorenz.Measurements import h
import time

# torch.set_default_dtype(torch.float64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
B = 32 # Batch size
L = 200 # Length
D_z = 2 # Measurement dimension
D_x = 3 # State dimension
lr = 0.001 # Learning rate
num_epochs = 2 # Number of total epochs
Q = torch.eye(3).to(device) # Process noise covariance
R = torch.diag(torch.tensor([10000., 400.])).to(device) # Measurement noise covariance

X_TRUE = torch.tensor(np.loadtxt("./data/Test_gt.txt", dtype='float32'), device=device).view(-1, D_x, L+1).transpose(2, 1)
Z_TRUE = torch.tensor(np.loadtxt("./data/Test_data.txt", dtype='float32'), device=device).view(-1, D_z, L).transpose(2, 1)


x_true = X_TRUE
z = Z_TRUE
with torch.no_grad():
    data = z #[B, L, D_z]
    B, _, _ = data.shape
    x = torch.zeros([B, L+1, D_x], device=device)
    P = torch.zeros([B, L+1, D_x, D_x], device=device)
    x[:, 0, :] = torch.tensor([1, 1, 1]).repeat(B, 1)
    P[:, 0, :, :] = (torch.eye(D_x) * 100).repeat(B, 1, 1)
    loss = 0
    for e in range(L):
        # Time Update Step
        n = 3
        N = 59
        sqrt_P = torch.linalg.cholesky(P[:, e, :, :], upper=False) #[B, D_x, D_x]
        r1 = 2.2551372655
        r2 = 0.7174531274
        r3 = 1.8430194370
        r4 = 1.5584810327
        r6 = 1.3055615004
        w1 = 0.0246319934
        w2 = 0.081510094
        w3 = 0.00976723555
        w4 = 0.0057724893
        w6 = 0.0002794729
        w7 = 1 - w1 * 6 - w2 * 8 - w3 * 12 - w4 * 8 - w6 * 24
        hh = 2.74
        x_CUT8 = torch.zeros([B, n, N], dtype=torch.float32, device=device)
        w_CUT8 = torch.zeros([B, N], dtype=torch.float32, device=device)
        x_CUT8[:, :, 0:3] = torch.eye(3).unsqueeze(0).repeat(B, 1, 1)
        x_CUT8[:, :, 3:6] = -torch.eye(3).unsqueeze(0).repeat(B, 1, 1)
        x_CUT8[:, :, 6:14] = torch.tensor([[
            [1, -1, 1, 1, -1, 1, -1, -1],
            [1, 1, -1, 1, -1, -1, 1, -1],
            [1, 1, 1, -1, 1, -1, -1, -1]
        ]]).repeat(B, 1, 1)
        x_CUT8[:, :, 14:26] = torch.tensor([[
            [1, 1, 0, -1, -1, 0, 1, 1, 0, -1, -1, 0],
            [1, 0, 1, 1, 0, -1, -1, 0, 1, -1, 0, -1],
            [0, 1, 1, 0, 1, 1, 0, -1, -1, 0, -1, -1]
        ]]).repeat(B, 1, 1)
        x_CUT8[:, :, 26:34] = torch.tensor([[
            [1, -1, 1, 1, -1, 1, -1, -1],
            [1, 1, -1, 1, -1, -1, 1, -1],
            [1, 1, 1, -1, 1, -1, -1, -1]
        ]]).repeat(B, 1, 1)
        x_CUT8[:, :, 34:58] = torch.tensor([[
            [hh, -hh, hh, hh, -hh, hh, -hh, -hh, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1],
            [1, 1, -1, 1, -1, -1, 1, -1, hh, hh, -hh, hh, -hh, -hh, hh, -hh, 1, 1, -1, 1, -1, -1, 1, -1],
            [1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, hh, hh, hh, -hh, hh, -hh, -hh, -hh]
        ]]).repeat(B, 1, 1)
        x_CUT8[:, :, 0:6] = x_CUT8[:, :, 0:6] * r1
        x_CUT8[:, :, 6:14] = x_CUT8[:, :, 6:14] * r2
        x_CUT8[:, :, 14:26] = x_CUT8[:, :, 14:26] * r3
        x_CUT8[:, :, 26:34] = x_CUT8[:, :, 26:34] * r4
        x_CUT8[:, :, 34:58] = x_CUT8[:, :, 34:58] * r6
        w_CUT8[:, 0:6] = w_CUT8[:, 0:6] + w1
        w_CUT8[:, 6:14] = w_CUT8[:, 6:14] + w2
        w_CUT8[:, 14:26] = w_CUT8[:, 14:26] + w3
        w_CUT8[:, 26:34] = w_CUT8[:, 26:34] + w4
        w_CUT8[:, 34:58] = w_CUT8[:, 34:58] + w6
        w_CUT8[:, 58] = w7
        x_CUT8 = x_CUT8.transpose(1, 2)
        x_CUT8_1 = torch.zeros_like(x_CUT8, device=device)
        for i in range(N):
            x_CUT8_1[:, i, :] = (sqrt_P @ x_CUT8[:, i, :].unsqueeze(-1)).squeeze(-1) + x[:, e, :]
        x_pred_sigma = f(x_CUT8_1.reshape([-1, D_x])).reshape([B, -1, D_x]) #[B, N, D_x]
        x_pred = (w_CUT8.unsqueeze(-1) * x_pred_sigma).sum(dim=1) #[B, D_x]
        x_centered = x_pred_sigma - x_pred.unsqueeze(1)
        P_pred = (w_CUT8.unsqueeze(-1) * x_centered).transpose(1, 2).matmul(x_centered) #[B, D_x, D_x]
        P_pred = P_pred + Q.repeat(B, 1, 1)

        # Measurement Update Step
        sqrt_P = torch.linalg.cholesky(P_pred, upper=False) #[B, D_x, D_x]
        x_CUT8_2 = torch.zeros_like(x_CUT8, device=device)
        for i in range(N):
            x_CUT8_2[:, i, :] = (sqrt_P @ x_CUT8[:, i, :].unsqueeze(-1)).squeeze(-1) + x_pred
        z_pred_sigma = h(x_CUT8_2.reshape([-1, D_x])).reshape([B, -1, D_z]) #[B, N, D_z]
        z_pred = (w_CUT8.unsqueeze(-1) * z_pred_sigma).sum(dim=1) #[B, D_z]
        z_centered = z_pred_sigma - z_pred.unsqueeze(1)
        P_zz = (w_CUT8.unsqueeze(-1) * z_centered).transpose(1, 2).matmul(z_centered) #[B, D_z, D_z]
        P_zz = P_zz + R.repeat(B, 1, 1)
        x_centered = x_CUT8_2 - x_pred.unsqueeze(1)
        P_xz = (w_CUT8.unsqueeze(-1) * x_centered).transpose(1, 2).matmul(z_centered) #[B, D_x, D_z]
        K = torch.bmm(P_xz, torch.linalg.inv(P_zz)) #[B, D_x, D_z]
        x[:, e+1, :] = x_pred + (K @ (data[:, e, :] - z_pred).unsqueeze(-1)).squeeze(-1)
        P[:, e+1, :, :] = P_pred - K @ P_zz @ K.transpose(1, 2)

residual = x[:, 1:, :] - x_true[:, 1:, :]

RMSE = torch.sqrt(torch.mean(torch.mean(residual*residual, dim=0), dim=1))

print(torch.sqrt(torch.mean(RMSE**2)))

np.savetxt('RMSE.txt', RMSE.cpu().numpy())