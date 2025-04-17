import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from Lorenz.Dynamics_simple import f
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

start_time = time.time()

with torch.no_grad():
    data = z #[B, L, D_z]
    B, _, _ = data.shape
    x = torch.zeros([B, L+1, D_x], device=device)
    P = torch.zeros([B, L+1, D_x, D_x], device=device)
    x[:, 0, :] = torch.tensor([1, 1, 1]).repeat(B, 1)
    P[:, 0, :, :] = (torch.eye(D_x) * 100).repeat(B, 1, 1)
    B = 1
    n = 3
    N = 14
    r1 = torch.sqrt(torch.tensor((n + 2) / 2, dtype=torch.float32, device=device))
    r2 = torch.sqrt(torch.tensor((n + 2) / (n - 2), dtype=torch.float32, device=device))
    w1 = 4 / (n + 2)**2
    w2 = (n - 2)**2 / (2**n * (n + 2)**2)
    x_CUT4 = torch.zeros(B, n, N, dtype=torch.float32, device=device)
    w_CUT4 = torch.zeros(B, N, dtype=torch.float32, device=device)
    x_CUT4[:, :, 0:3] = torch.eye(3).unsqueeze(0).repeat(B, 1, 1)
    x_CUT4[:, :, 3:6] = -torch.eye(3).unsqueeze(0).repeat(B, 1, 1)
    x_CUT4[:, :, 6:] = torch.tensor([[
        [1, -1, 1, 1, -1, 1, -1, -1],
        [1, 1, -1, 1, -1, -1, 1, -1],
        [1, 1, 1, -1, 1, -1, -1, -1]
    ]]).repeat(B, 1, 1)
    x_CUT4[:, :, 0:6] = x_CUT4[:, :, 0:6] * r1
    x_CUT4[:, :, 6:] = x_CUT4[:, :, 6:] * r2
    w_CUT4[:, 0:6] = w_CUT4[:, 0:6] + w1
    w_CUT4[:, 6:] = w_CUT4[:, 6:] + w2
    x_CUT4 = x_CUT4.transpose(1, 2)
    for num in range(1000):
        print(num)
        for e in range(L):
            # Time Update Step
            sqrt_P = torch.linalg.cholesky(P[num, e, :, :].unsqueeze(0), upper=False) #[B, D_x, D_x]
            x_CUT4_new = (sqrt_P.unsqueeze(1).repeat(1, N, 1, 1) @ x_CUT4.unsqueeze(-1)).squeeze(-1) + x[num, e, :].unsqueeze(0).unsqueeze(1).repeat(1, N, 1)
            x_pred_sigma = f(x_CUT4_new.reshape([-1, D_x])).reshape([B, -1, D_x]) #[B, N, D_x]
            x_pred = (w_CUT4.unsqueeze(-1) * x_pred_sigma).sum(dim=1) #[B, D_x]
            x_centered = x_pred_sigma - x_pred.unsqueeze(1)
            P_pred = (w_CUT4.unsqueeze(-1) * x_centered).transpose(1, 2).matmul(x_centered) #[B, D_x, D_x]
            P_pred = P_pred + Q.repeat(B, 1, 1)

            # Measurement Update Step
            sqrt_P = torch.linalg.cholesky(P_pred, upper=False) #[B, D_x, D_x]
            x_CUT4_new = (sqrt_P.unsqueeze(1).repeat(1, N, 1, 1) @ x_CUT4.unsqueeze(-1)).squeeze(-1) + x_pred.unsqueeze(1).repeat(1, N, 1)
            z_pred_sigma = h(x_CUT4_new.reshape([-1, D_x])).reshape([B, -1, D_z]) #[B, N, D_z]
            z_pred = (w_CUT4.unsqueeze(-1) * z_pred_sigma).sum(dim=1) #[B, D_z]
            z_centered = z_pred_sigma - z_pred.unsqueeze(1)
            P_zz = (w_CUT4.unsqueeze(-1) * z_centered).transpose(1, 2).matmul(z_centered) #[B, D_z, D_z]
            P_zz = P_zz + R.repeat(B, 1, 1)
            x_centered = x_CUT4_new - x_pred.unsqueeze(1)
            P_xz = (w_CUT4.unsqueeze(-1) * x_centered).transpose(1, 2).matmul(z_centered) #[B, D_x, D_z]
            K = torch.bmm(P_xz, torch.linalg.inv(P_zz)) #[B, D_x, D_z]
            x[num, e+1, :] = x_pred + (K @ (data[num, e, :] - z_pred).unsqueeze(-1)).squeeze(-1)
            P[num, e+1, :, :] = P_pred - K @ P_zz @ K.transpose(1, 2)

end_time = time.time()
print(end_time - start_time)