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
import scipy

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
mat_file = scipy.io.loadmat('samples.mat')
samples = mat_file['samples']

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
        N = 43
        sqrt_P = torch.linalg.cholesky(P[:, e, :, :], upper=False) #[B, D_x, D_x]
        x_S2KF = torch.tensor(samples, dtype=torch.float32, device=device).unsqueeze(0).transpose(1, 2).repeat(B, 1, 1)
        w_S2KF = torch.zeros(B, N, dtype=torch.float32, device=device) + 1/N
        x_S2KF_1 = torch.zeros_like(x_S2KF, device=device)
        for i in range(N):
            x_S2KF_1[:, i, :] = (sqrt_P @ x_S2KF[:, i, :].unsqueeze(-1)).squeeze(-1) + x[:, e, :]
        x_pred_sigma = f(x_S2KF_1.reshape([-1, D_x])).reshape([B, -1, D_x]) #[B, N, D_x]
        x_pred = (w_S2KF.unsqueeze(-1) * x_pred_sigma).sum(dim=1) #[B, D_x]
        x_centered = x_pred_sigma - x_pred.unsqueeze(1)
        P_pred = (w_S2KF.unsqueeze(-1) * x_centered).transpose(1, 2).matmul(x_centered) #[B, D_x, D_x]
        P_pred = P_pred + Q.repeat(B, 1, 1)

        # Measurement Update Step
        sqrt_P = torch.linalg.cholesky(P_pred, upper=False) #[B, D_x, D_x]
        x_S2KF_2 = torch.zeros_like(x_S2KF, device=device)
        for i in range(N):
            x_S2KF_2[:, i, :] = (sqrt_P @ x_S2KF[:, i, :].unsqueeze(-1)).squeeze(-1) + x_pred
        z_pred_sigma = h(x_S2KF_2.reshape([-1, D_x])).reshape([B, -1, D_z]) #[B, N, D_z]
        z_pred = (w_S2KF.unsqueeze(-1) * z_pred_sigma).sum(dim=1) #[B, D_z]
        z_centered = z_pred_sigma - z_pred.unsqueeze(1)
        P_zz = (w_S2KF.unsqueeze(-1) * z_centered).transpose(1, 2).matmul(z_centered) #[B, D_z, D_z]
        P_zz = P_zz + R.repeat(B, 1, 1)
        x_centered = x_S2KF_2 - x_pred.unsqueeze(1)
        P_xz = (w_S2KF.unsqueeze(-1) * x_centered).transpose(1, 2).matmul(z_centered) #[B, D_x, D_z]
        K = torch.bmm(P_xz, torch.linalg.inv(P_zz)) #[B, D_x, D_z]
        x[:, e+1, :] = x_pred + (K @ (data[:, e, :] - z_pred).unsqueeze(-1)).squeeze(-1)
        P[:, e+1, :, :] = P_pred - K @ P_zz @ K.transpose(1, 2)

residual = x[:, 1:, :] - x_true[:, 1:, :]

RMSE = torch.sqrt(torch.mean(torch.mean(residual*residual, dim=0), dim=1))

print(torch.sqrt(torch.mean(RMSE**2)))

np.savetxt('RMSE.txt', RMSE.cpu().numpy())