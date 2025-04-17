import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from Lorenz.Dynamics import f
from Lorenz.Measurements import h
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

seed = 2026
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.set_num_threads(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
B = 32 # Batch size
L = 200 # Length
D_z = 2 # Measurement dimension
D_x = 3 # State dimension
lr = 0.0001 # Learning rate
num_epochs = 10  # Number of total epochs
Q = torch.eye(3).to(device) # Process noise covariance
R = torch.diag(torch.tensor([10000., 400.])).to(device) # Measurement noise covariance
model_weights_path = './model/DeepCUT6_300_N_neuron.pt'

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.N = 27
        self.fc1 = nn.Linear(9, 300)
        self.fc2 = nn.Linear(300, 300)
        self.fc3 = nn.Linear(300, self.N * 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        # w_sigma = x[:, :self.N]
        x_sigma = x[:, :].reshape(-1, self.N, 3)
        return x_sigma
model = MLP().to(device)
model.load_state_dict(torch.load('./model/UKF_20_300_N_neuron.pt'))
total = sum([param.nelement() for param in model.parameters()])
print('Number of parameter: % .4f' % (total))

# Load dataset
def str2float(list):
    strlist = []
    for i in list:
        strlist.append(float(i))
    return strlist

class CustomDataset(Dataset):
    def __init__(self, data_list, label_list):
        self.data_list = data_list
        self.label_list = label_list
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        text = self.data_list[idx]
        label = self.label_list[idx]
        return text, label


# Train data and label
# When creating a dataset, each row represents a set of data and each column is listed as a dimension of the data
Train_data_list = np.loadtxt("./data/Train_data.txt", dtype='float32')[:1000, :]
Train_label_list = np.loadtxt("./data/Train_gt.txt", dtype='float32')[:1000, :]
custom_dataset = CustomDataset(Train_data_list, Train_label_list)
Train_loader = DataLoader(custom_dataset, batch_size=B, shuffle=True)

# Test data and label
Test_data_list = np.loadtxt("./data/Test_data.txt", dtype='float32')
Test_label_list = np.loadtxt("./data/Test_gt.txt", dtype='float32')
custom_dataset = CustomDataset(Test_data_list, Test_label_list)
Test_loader = DataLoader(custom_dataset, batch_size=len(Test_data_list), shuffle=False)


# Loss and Update
# LossF = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Train
model.train()
for epoch in range(num_epochs):
    for batch_index, (data, labels) in enumerate(Train_loader):
        data = data.view(-1, D_z, L).transpose(2, 1).to(device) #[B, L, D_z]
        B, _, _ = data.shape
        x = torch.zeros([B, L+1, D_x], device=device)
        P = torch.zeros([B, L+1, D_x, D_x], device=device)
        x[:, 0, :] = torch.tensor([1, 1, 1]).repeat(B, 1)
        P[:, 0, :, :] = (torch.eye(D_x) * 100).repeat(B, 1, 1)
        loss = 0
        for k in range(L):
            # Time Update Step
            U, S, V = torch.svd(P[:, k, :, :])
            sqrt_P = torch.linalg.cholesky(torch.bmm(U, S.unsqueeze(-1) * U.transpose(1, 2)), upper=False) #[B, D_x, D_x]
            sqrt_P_lower = torch.cat((sqrt_P[:, :, 0], sqrt_P[:, 1:, 1], sqrt_P[:, 2:, 2]), dim=1)
            x_sigma = model(torch.cat((x[:, k, :], sqrt_P_lower), dim=1))
            n = 3
            N = 27
            w_sigma = torch.zeros([B, N], device=device)
            w1 = 0.0290351301
            w2 = 0.0633844605
            w3 = 0.0005195469
            w4 = 0.0610193182
            w_sigma[:, 0:6] = w_sigma[:, 0:6] + w1
            w_sigma[:, 6:18] = w_sigma[:, 6:18] + w2
            w_sigma[:, 18:26] = w_sigma[:, 18:26] + w3
            w_sigma[:, 26] = w4
            x_pred_sigma = f(x_sigma.reshape([-1, D_x])).reshape([B, -1, D_x]) #[B, N, D_x]
            x_pred = (w_sigma.unsqueeze(-1) * x_pred_sigma).sum(dim=1) #[B, D_x]
            x_centered = x_pred_sigma - x_pred.unsqueeze(1)
            P_pred = (w_sigma.unsqueeze(-1) * x_centered).transpose(1, 2).matmul(x_centered) #[B, D_x, D_x]
            P_pred = P_pred + Q.repeat(B, 1, 1)

            # Measurement Update Step
            sqrt_P = torch.linalg.cholesky(P_pred, upper=False) #[B, D_x, D_x]
            sqrt_P_lower = torch.cat((sqrt_P[:, :, 0], sqrt_P[:, 1:, 1], sqrt_P[:, 2:, 2]), dim=1)
            x_sigma = model(torch.cat((x_pred, sqrt_P_lower), dim=1))
            z_pred_sigma = h(x_sigma.reshape([-1, D_x])).reshape([B, -1, D_z]) #[B, N, D_z]
            z_pred = (w_sigma.unsqueeze(-1) * z_pred_sigma).sum(dim=1) #[B, D_z]
            z_centered = z_pred_sigma - z_pred.unsqueeze(1)
            P_zz = (w_sigma.unsqueeze(-1) * z_centered).transpose(1, 2).matmul(z_centered) #[B, D_z, D_z]
            P_zz = P_zz + R.repeat(B, 1, 1)
            x_centered = x_sigma - x_pred.unsqueeze(1)
            P_xz = (w_sigma.unsqueeze(-1) * x_centered).transpose(1, 2).matmul(z_centered) #[B, D_x, D_z]
            K = torch.bmm(P_xz, torch.linalg.inv(P_zz)) #[B, D_x, D_z]
            x[:, k+1, :] = x_pred + (K @ (data[:, k, :] - z_pred).unsqueeze(-1)).squeeze(-1)
            P[:, k+1, :, :] = P_pred - K @ P_zz @ K.transpose(1, 2)
            # Compute unsupervised loss
            residual = z_pred - data[:, k, :] # [B, D_z]
            residual = residual.unsqueeze(2)  # [B, D_z, 1]
            # loss = loss + torch.mean((residual.transpose(1, 2) @ torch.linalg.inv(R.repeat(B, 1, 1)) @ residual).squeeze()) / L
            loss = loss + torch.mean(0.5*(residual.transpose(1, 2) @ torch.linalg.inv(P_zz) @ residual).squeeze() + 0.5*torch.log(torch.linalg.det(P_zz))) / L
        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_index % 1 == 0:
            print('[{}/{}],[{}/{}],loss={:.4f}'.format(epoch, num_epochs, batch_index, len(Train_loader), loss))
torch.save(model.state_dict(), model_weights_path)
