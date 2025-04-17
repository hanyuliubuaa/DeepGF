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
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

torch.set_default_dtype(torch.float64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
L = 200 # Length
D_z = 2 # Measurement dimension
D_x = 3 # State dimension
Q = torch.eye(3).to(device) # Process noise covariance
R = torch.diag(torch.tensor([10000., 400.])).to(device) # Measurement noise covariance

X_TRUE = torch.tensor(np.loadtxt("./data/Test_gt.txt", dtype='float32'), device=device).view(-1, D_x, L+1).transpose(2, 1)
Z_TRUE = torch.tensor(np.loadtxt("./data/Test_data.txt", dtype='float32'), device=device).view(-1, D_z, L).transpose(2, 1)

x_true = X_TRUE
z = Z_TRUE

alpha = 0.9918
beta = 0.0011
kappa = 0.0034
with torch.no_grad():
    data = z #[B, L, D_z]
    B, _, _ = data.shape
    lamda = alpha**2 * (D_x + kappa) - D_x
    x = torch.zeros([B, L+1, D_x], device=device)
    P = torch.zeros([B, L+1, D_x, D_x], device=device)
    x[:, 0, :] = torch.tensor([1, 1, 1]).repeat(B, 1)
    P[:, 0, :, :] = (torch.eye(D_x) * 100).repeat(B, 1, 1)
    w_UKF = torch.zeros([B, 2*D_x + 1], device=device)
    w_UKF[:, 0] = w_UKF[:, 0] + lamda / (D_x+lamda)
    w_UKF[:, 1:] = w_UKF[:, 1:] + 1 / 2 / (D_x+lamda)
    for k in range(L):
        # Time Update Step
        x_UKF = torch.zeros([B, 2*D_x + 1, D_x], device=device) #[B, N, D_x]
        x_UKF[:, 0, :] = x[:, k, :]
        # U, S, V = torch.svd(P[:, k, :, :])
        # sqrt_P = torch.linalg.cholesky(torch.bmm(U, S.unsqueeze(-1) * U.transpose(1, 2)), upper=False) #[B, D_x, D_x]
        sqrt_P = torch.linalg.cholesky(P[:, k, :, :], upper=False) #[B, D_x, D_x]
        for i in range(D_x):
            x_UKF[:, i+1, :] = x[:, k, :] + np.sqrt(D_x+lamda) * sqrt_P[:, :, i]
            x_UKF[:, i+D_x+1, :] = x[:, k, :] - np.sqrt(D_x+lamda) * sqrt_P[:, :, i]
        x_pred_sigma = f(x_UKF.reshape([-1, D_x])).reshape([B, -1, D_x]) #[B, N, D_x]
        x_pred = (w_UKF.unsqueeze(-1) * x_pred_sigma).sum(dim=1) #[B, D_x]
        x_centered = x_pred_sigma - x_pred.unsqueeze(1)
        P_pred = (w_UKF.unsqueeze(-1) * x_centered).transpose(1, 2).matmul(x_centered) #[B, D_x, D_x]
        P_pred = P_pred + (1 - alpha**2 + beta) * (x_pred_sigma[:, 0, :] - x_pred).unsqueeze(2) @ (x_pred_sigma[:, 0, :] - x_pred).unsqueeze(1)
        P_pred = P_pred + Q.repeat(B, 1, 1)

        # Measurement Update Step
        x_UKF = torch.zeros([B, 2*D_x + 1, D_x], device=device) #[B, N, D_x]
        x_UKF[:, 0, :] = x_pred
        sqrt_P = torch.linalg.cholesky(P_pred, upper=False) #[B, D_x, D_x]
        for i in range(D_x):
            x_UKF[:, i+1, :] = x_pred + np.sqrt(D_x+lamda) * sqrt_P[:, :, i]
            x_UKF[:, i+D_x+1, :] = x_pred - np.sqrt(D_x+lamda) * sqrt_P[:, :, i]
        z_pred_sigma = h(x_UKF.reshape([-1, D_x])).reshape([B, -1, D_z]) #[B, N, D_z]
        z_pred = (w_UKF.unsqueeze(-1) * z_pred_sigma).sum(dim=1) #[B, D_z]
        z_centered = z_pred_sigma - z_pred.unsqueeze(1)
        P_zz = (w_UKF.unsqueeze(-1) * z_centered).transpose(1, 2).matmul(z_centered) #[B, D_z, D_z]
        P_zz = P_zz + (1 - alpha**2 + beta) * (z_pred_sigma[:, 0, :] - z_pred).unsqueeze(2) @ (z_pred_sigma[:, 0, :] - z_pred).unsqueeze(1)
        P_zz = P_zz + R.repeat(B, 1, 1)
        x_centered = x_UKF - x_pred.unsqueeze(1)
        P_xz = (w_UKF.unsqueeze(-1) * x_centered).transpose(1, 2).matmul(z_centered) #[B, D_x, D_z]
        K = torch.bmm(P_xz, torch.linalg.inv(P_zz)) #[B, D_x, D_z]
        x[:, k+1, :] = x_pred + (K @ (data[:, k, :] - z_pred).unsqueeze(-1)).squeeze(-1)
        P[:, k+1, :, :] = P_pred - K @ P_zz @ K.transpose(1, 2)
    residual = x[:, 1:, :] - x_true[:, 1:, :]
    RMSE = torch.sqrt(torch.mean(torch.mean(residual*residual, dim=0), dim=1))
    RMSE = torch.sqrt(torch.mean(RMSE**2))
    print(f"Evaluating: alpha={alpha:.4f}, beta={beta:.4f}, kappa={kappa:.4f} -> RMSE={RMSE.item():.4f}")

