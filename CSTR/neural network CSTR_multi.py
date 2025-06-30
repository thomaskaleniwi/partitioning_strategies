# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 19:06:17 2025

@author: TKale
"""

import torch
import os
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

class CSTRDynamicsNN(nn.Module):
    def __init__(self, dim_x=4, dim_u=2, dim_p=11, dropout_p = 0.0005):
        """
        Neural network for point‐prediction of CSTR dynamics:
            x_next = A⁺ x + A⁻ x + E⁺ p + E⁻ p + B u
        with A⁺≥0, A⁻≤0, E⁺≥0, E⁻≤0 for monotonic parts.
        """
        super().__init__()
        self.dim_x = dim_x
        self.dim_u = dim_u
        self.dim_p = dim_p
        # k = Gesamtzahl der Inputs x,p,u
        self.k = dim_x + dim_p + dim_u
        # k_u = Anzahl der nicht-monotonen Inputs = dim_u
        self.k_u = dim_u
        # k_mp = Anzahl der monotonen Inputs = k - k_u = dim_x + dim_p
        self.k_mp = dim_x + dim_p
        
        hidden = 128

        # --- MONOTONE Teil 1: nur x und p, Ausgabe k_mp - k_u Neuronen ---
        self.mon1a = nn.Linear(self.k_mp, hidden, bias=True)
        self.mon1_dropout1 = nn.Dropout(dropout_p)
        self.mon1b = nn.Linear(hidden, self.k_mp, bias=True)
        # self.mon1_dropout2 = nn.Dropout(dropout_p)
        # self.mon1c = nn.Linear(hidden, self.k_mp, bias=True)
        # self.mon1_dropout3 = nn.Dropout(dropout_p)
        # self.mon1d = nn.Linear(hidden, self.k_mp, bias=True)
        # self.mon1_dropout4 = nn.Dropout(dropout_p)
        # # self.mon1e = nn.Linear(hidden, hidden, bias=True)
        # self.mon1_dropout5 = nn.Dropout(dropout_p)
        # self.mon1f = nn.Linear(hidden, k_mp, bias=True)
        
        
        # --- concatenation: h1 and u, output k neurons  ---
        self.mon2a = nn.Linear(self.k, hidden, bias=True)
        self.mon2_dropout1 = nn.Dropout(dropout_p)
        self.mon2b = nn.Linear(hidden, hidden, bias=True)
        self.mon2_dropout2 = nn.Dropout(dropout_p)
        self.mon2c = nn.Linear(hidden, self.dim_x, bias=False)
        # self.mon2_dropout3 = nn.Dropout(dropout_p)
        # self.mon2d = nn.Linear(hidden, hidden, bias=True)
        # self.mon2_dropout4 = nn.Dropout(dropout_p)
        # self.mon2e = nn.Linear(hidden, hidden, bias=True)
        # self.mon2_dropout5 = nn.Dropout(dropout_p)
        # self.mon2f = nn.Linear(hidden, self.dim_x, bias=False)
        # self.mon2_dropout6 = nn.Dropout(dropout_p)
        # self.mon2g = nn.Linear(hidden, hidden, bias=True)
        # self.mon2_dropout7 = nn.Dropout(dropout_p)
        # self.mon2h = nn.Linear(hidden, self.dim_x, bias=False)
        
    
        
        
    def forward(self, x, u, p):
        B = x.size(0)
        # 1) PARTIELLE MONOTONIE LAYER 1: nur x & p
        xp = torch.cat([x, p], dim=1)           # (batch, dim_x+dim_p)
        # positive Gewichte erzwingen:
        w1a = F.relu(self.mon1a.weight)           # (out, in)
        b1a = self.mon1a.bias                     # (out,)
        w1b = F.relu(self.mon1b.weight)
        b1b = self.mon1b.bias
        # w1c = F.relu(self.mon1c.weight)
        # b1c = self.mon1c.bias
        # w1d = F.relu(self.mon1d.weight)
        # b1d = self.mon1d.bias
        # w1e = F.relu(self.mon1e.weight)
        # b1e = self.mon1e.bias
        # w1f = F.relu(self.mon1f.weight)
        # b1f = self.mon1f.bias
        
        
        h1a = F.tanh(F.linear(xp, w1a, b1a )) # (batch, k_mp - k_u)
        h1a_dropout = self.mon1_dropout1(h1a)
        h1b = F.tanh(F.linear(h1a_dropout, w1b, b1b))
        # h1b_dropout = self.mon1_dropout2(h1b)
        # h1c = F.tanh(F.linear(h1b_dropout, w1c, b1c))
        # h1c_dropout = self.mon1_dropout3(h1c)
        # h1d = F.tanh(F.linear(h1c_dropout, w1d, b1d))
        # h1d_dropout = self.mon1_dropout4(h1d)
        # h1e = F.sigmoid(F.linear(h1d_dropout, w1e, b1e))
        # h1e_dropout = self.mon1_dropout5(h1e)
        # h1f = F.sigmoid(F.linear(h1e_dropout, w1f, b1f))

        # 2) Ausgabe LAYER 2: h1 & u
        hu = torch.cat([h1b, u], dim=1)          # (batch, (k_mp - k_u)+k_u)
        
        w2b = F.relu(self.mon2b.weight)
        b2b = self.mon2b.bias
        w2c = self.mon2c.weight
        # b2c = self.mon2c.bias
        # w2d = F.relu(self.mon2d.weight)
        # b2d = self.mon2d.bias
        # w2e = F.relu(self.mon2e.weight)
        # b2e = self.mon2e.bias
        # w2f = self.mon2f.weight
        # b2f = self.mon2f.bias
        # w2g = F.relu(self.mon2g.weight)
        # b2g = self.mon2g.bias
        # w2h = self.mon2h.weight
        
        

#-------------------------------------------------------------------------------
        W_raw_2a = self.mon2a.weight       # shape (k, k_mp + k_u)
        b2a = self.mon2a.bias
        W_raw_2b = self.mon2b.weight
        # b2b = self.mon2b.bias
        # W_raw_2c = self.mon2c.weight
        # b2c = self.mon2c.bias
        # W_raw_2d = self.mon2d.weight
        # b2d = self.mon2d.bias
        # W_raw_2e = self.mon2e.weight
        # b2e = self.mon2e.bias
        # W_raw_2f = self.mon2f.weight
        # b2f = self.mon2f.bias
        # W_raw_2g = self.mon2g.weight
        
        
        # Splitting of weights in positive and negative parts
        W_xp_2a, W_u_2a = W_raw_2a[:, :self.k_mp], W_raw_2a[:, self.k_mp:]    # xp: dim (k,k_mp), u: dim (k,k_u)
        # W_xp_2b, W_u_2b = W_raw_2b[:, :self.k_mp], W_raw_2b[:, self.k_mp:]
        # W_xp_2c, W_u_2c = W_raw_2c[:, :self.k_mp], W_raw_2c[:, self.k_mp:] 
        # W_xp_2d, W_u_2d = W_raw_2d[:, :self.k_mp], W_raw_2d[:, self.k_mp:]
        # W_xp_2e, W_u_2e = W_raw_2e[:, :self.k_mp], W_raw_2e[:, self.k_mp:]
        
        # Forcing the xp matrices to be positive
        W_xp_2a_pos = F.relu(W_xp_2a)
        # W_xp_2b_pos = F.relu(W_xp_2b)
        # W_xp_2c_pos = F.relu(W_xp_2c)
        # W_xp_2d_pos = F.relu(W_xp_2d)
        # W_xp_2e_pos = F.relu(W_xp_2e)

        # u matrix remains free 
        W_u_2a_free = W_u_2a
        # W_u_2b_free = W_u_2b
        # W_u_2c_free = W_u_2c
        # W_u_2d_free = W_u_2d
        # W_u_2e_free = W_u_2e

        # concatenate both matrices
        W_mon2a = torch.cat((W_xp_2a_pos, W_u_2a_free), dim=1)  # shape (k, k_mp+k_u)
        # W_mon2b = torch.cat((W_xp_2b_pos, W_u_2b_free), dim=1)
        # W_mon2c = torch.cat((W_xp_2c_pos, W_u_2c_free), dim=1)
        # W_mon2d = torch.cat((W_xp_2d_pos, W_u_2d_free), dim=1)
        # W_mon2e = torch.cat((W_xp_2e_pos, W_u_2e_free), dim=1)
        
       
        
#-------------------------------------------------------------------------------      
        
        h2a = F.tanh(F.linear(hu, W_mon2a, b2a ))
        h2a_dropout = self.mon2_dropout1(h2a)
        h2b = F.tanh(F.linear(h2a, w2b, b2b ))
        # h2b_dropout = self.mon2_dropout2(h2b)
        h2c = F.linear(h2b, w2c)
        # # h2c_dropout = self.mon2_dropout3(h2c)
        # h2d = F.tanh(F.linear(h2c, w2d, b2d ))
        # h2d_dropout = self.mon2_dropout4(h2d)
        # h2e = F.tanh(F.linear(h2d_dropout, w2e, b2e ))
        # h2e_dropout = self.mon2_dropout5(h2e)
        # h2f = F.linear(h2e_dropout, w2f)
        # h2f_dropout = self.mon2_dropout6(h2f)
        # h2g = F.tanh(F.linear(h2f_dropout, w2g, b2g ))
        # h2g_dropout = self.mon2_dropout7(h2g)
        # h2h = F.linear(h2g_dropout, w2h )
        
        z  = h2c

        x_next = z
        
        return x_next #cA and cB must be positive otherwise there is no physical sense


# creation of training-,validation- and test-setup
    
# 1) Hyperparameter
batch_size   = 128
lr           = 1e-3
n_epochs     = 1500
patience     = 59   # for Early Stopping
device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2) Dataset / DataLoader

# Path to data
save_dir = 'D:/partitioning approaches_CSTR/CSTR_NN/Multi-step'
inpath = os.path.join(save_dir, 'CSTR_data_multi.pt')

# extract all data
data = torch.load(inpath)

# unpack and declare data
X_train = data['X_train'].to(device)   # Tensor shape (n_train, 17)
Y_train = data['Y_train'].to(device)   # Tensor shape (n_train,  4)
X_val   = data['X_val'].to(device)     # Tensor shape (n_val,   17)
Y_val   = data['Y_val'].to(device)     # Tensor shape (n_val,    4)
X_test  = data['X_test'].to(device)    # Tensor shape (n_test,  17)
Y_test  = data['Y_test'].to(device)    # Tensor shape (n_test,   4)

H = X_train.size(1)  # Sequenzlänge (Simulationsteps (20))

# normalize training-data-set
# normalize training-data-set overr **samples** und **timesteps**
x_mean = X_train.mean(dim=(0,1), keepdim=True)  # shape (1,1,17)
x_std  = X_train.std(dim=(0,1),  keepdim=True)
y_mean = Y_train.mean(dim=(0,1), keepdim=True)  # shape (1,1, 4)
y_std  = Y_train.std(dim=(0,1),  keepdim=True)

# no division by zero
x_std[x_std == 0] = 1.0
y_std[y_std == 0] = 1.0

# in place scaling
X_train_norm = (X_train - x_mean) / x_std
X_val_norm   = (X_val   - x_mean) / x_std
X_test_norm  = (X_test  - x_mean) / x_std

Y_train_norm = (Y_train - y_mean) / y_std
Y_val_norm  = (Y_val  - y_mean) / y_std
Y_test_norm = (Y_test - y_mean) / y_std

x_mean = x_mean.to(device)
x_std  = x_std.to(device)
y_mean = y_mean.to(device)
y_std  = y_std.to(device)


torch.save({
    "X_train":    X_train,
    "Y_train":    Y_train,
    "X_val":      X_val,
    "Y_val":      Y_val,
    "X_test":     X_test,
    "Y_test":     Y_test,
    # new:
    "x_mean":     x_mean.cpu(),
    "x_std":      x_std.cpu(),
    "y_mean":     y_mean.cpu(),
    "y_std":      y_std.cpu(),
}, inpath)
# Dataloader 

def make_loader(X, Y, batch_size, shuffle=True):
    ds = TensorDataset(X,Y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

train_loader = make_loader(X_train_norm, Y_train_norm, batch_size, shuffle=True)
val_loader   = make_loader(X_val_norm,   Y_val_norm,   batch_size, shuffle=False)
test_loader  = make_loader(X_test_norm,  Y_test_norm,  batch_size, shuffle=False)

# 3) Modell, Loss, Optimizer
model     = CSTRDynamicsNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay= 1e-5)

# 4) Early‐Stopping Setup
best_val_loss = float('inf')
epochs_no_improve = 0
BEST_MODEL_PATH   = os.path.join(save_dir, 'best_model.pth')
os.makedirs(os.path.dirname(BEST_MODEL_PATH), exist_ok=True)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
optimizer,
mode='min',           # wir wollen, dass der Loss abnimmt
factor=0.7,           # LR ← LR * 0.5
patience=10,           # Warte 5 Epochen ohne Verbesserung
threshold=1e-5,       # minimale relative Verbesserung
threshold_mode='rel',
cooldown=5,           # nach Reduktion 5 Epochen Pause
min_lr=1e-5,
verbose=False
)

# 5) Training‐Loop
eps_start = 1.0    # zu Beginn: immer Teacher Forcing
eps_end   = 0.0    # am Ende: nie Teacher Forcing
decay_epochs = n_epochs

for epoch in range(1, n_epochs+1):
    # ↓ scheduled sampling ratio
    # eps = eps_end + (eps_start - eps_end) * max(0, (decay_epochs-epoch)/decay_epochs)
    # — Training
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        # xb: (bs, H, 17), yb: (bs, H, 4)
        xb, yb = xb.to(device), yb.to(device)
        bs = xb.size(0)
        
        # Unroll multi-step
        x_prev = xb[:,0,:4]                  # initial state
        loss_batch = 0.0
        preds = []
        
        for t in range(H):
            u_t = xb[:,t,4:6]
            p_t = xb[:,t,6:]
            y_pred = model(x_prev, u_t, p_t)       # (bs,4)
            preds.append(y_pred.unsqueeze(1))      # (bs,1,4)
            x_prev = y_pred                        # teacher-forcing = 0: always Pred
            
            # === scheduled sampling: mask für jede Probe im Batch ===
            # mask[i]=True  → teacher forcing: x_prev ← true yb
            # mask[i]=False → model prediction
            # mask = (torch.rand(bs, device=device) < eps).float().unsqueeze(1)  # (bs,1)
            y_true_t = yb[:, t, :]                                          # (bs,4)
            # x_prev = mask * y_true_t + (1-mask) * y_pred


        preds_seq = torch.cat(preds, dim=1)        # (bs,H,4)
        # loss1 = criterion(preds_seq[:,0,:], yb[:,0,:])
        #loss = criterion(preds_seq, yb)
        # alpha = 0.3
        # loss   = alpha * loss1 + (1 - alpha) * lossM
        loss = criterion(preds_seq, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * bs

    train_loss /= len(train_loader.dataset)

    # — Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            x_prev = xb[:,0,:4]
            preds = []
            for t in range(H):
                u_t = xb[:,t,4:6]
                p_t = xb[:,t,6:]
                y_pred = model(x_prev, u_t, p_t)
                preds.append(y_pred.unsqueeze(1))
                x_prev = y_pred
            preds_seq = torch.cat(preds, dim=1)
            val_loss += criterion(preds_seq, yb).item() * xb.size(0)
    val_loss /= len(val_loader.dataset)
    scheduler.step(val_loss)
    print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    # Early‐Stopping Check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping!")
            break

# 6) Load the best model
state = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True)
model.load_state_dict(state)

# 7) Test‐Evaluation
model.eval()
test_loss = 0.0
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        x_prev = xb[:,0,:4]
        preds = []
        for t in range(H):
            u_t = xb[:,t,4:6]
            p_t = xb[:,t,6:]
            y_pred = model(x_prev, u_t, p_t)
            preds.append(y_pred.unsqueeze(1))
            x_prev = y_pred
        preds_seq = torch.cat(preds, dim=1)
        
        test_loss += criterion(preds_seq, yb).item() * xb.size(0)
    test_loss /= len(test_loader.dataset)

print(f"Test Loss: {test_loss:.6f}")
