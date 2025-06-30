# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 11:42:10 2025

@author: TKale
"""

import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F


# Load data and model
save_dir = 'D:/partitioning approaches_CSTR/CSTR_NN'
inpath = os.path.join(save_dir, 'CSTR_data.pt')
data = torch.load(inpath)
X_train = data['X_train']
Y_train = data['Y_train']
X_test = data['X_test']
Y_test = data['Y_test']

device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Recompute normalization parameters
x_mean = X_train.mean(dim=0, keepdim=True)
x_std  = X_train.std(dim=0,  keepdim=True)
x_std[x_std == 0] = 1.0

y_mean = Y_train.mean(dim=0, keepdim=True)
y_std  = Y_train.std(dim=0, keepdim=True)
y_std[y_std == 0] = 1.0

# Normalize test inputs
X_test_norm = (X_test - x_mean) / x_std

# Prepare model
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
        
        hidden = 32

        # --- MONOTONE Teil 1: nur x und p, Ausgabe k_mp - k_u Neuronen ---
        self.mon1a = nn.Linear(self.k_mp, hidden, bias=True)
        self.mon1_dropout1 = nn.Dropout(dropout_p)
        self.mon1b = nn.Linear(hidden, self.k_mp, bias=True)
        # self.mon1_dropout2 = nn.Dropout(dropout_p)
        # self.mon1c = nn.Linear(hidden, hidden, bias=True)
        # self.mon1_dropout3 = nn.Dropout(dropout_p)
        # self.mon1d = nn.Linear(hidden, self.k_mp, bias=True)
        # self.mon1_dropout4 = nn.Dropout(dropout_p)
        # # self.mon1e = nn.Linear(hidden, hidden, bias=True)
        # self.mon1_dropout5 = nn.Dropout(dropout_p)
        # self.mon1f = nn.Linear(hidden, k_mp, bias=True)
        
        
        # --- concatenation: h1 and u, output k neurons  ---
        self.mon2a = nn.Linear(self.k, hidden, bias=True)
        self.mon2_dropout1 = nn.Dropout(dropout_p)
        self.mon2b = nn.Linear(hidden, self.dim_x, bias=False)
        # self.mon2_dropout2 = nn.Dropout(dropout_p)
        # self.mon2c = nn.Linear(hidden, hidden, bias=True)
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
        
        w2b = self.mon2b.weight
        # b2b = self.mon2b.bias
        # w2c = F.relu(self.mon2c.weight)
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
        h2b = F.linear(h2a, W_raw_2b )
        # # h2b_dropout = self.mon2_dropout2(h2b)
        # h2c = F.tanh(F.linear(h2b, w2c, b2c ))
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
        
        z  = h2b

        x_next = z
        
        return x_next #cA and cB must be positive otherwise there is no physical sense




model = CSTRDynamicsNN()
best_model_path = os.path.join(save_dir, 'best_model.pth')
state = torch.load(best_model_path)
model.load_state_dict(state)
model.eval()

# Compute predictions on test set
with torch.no_grad():
    x = X_test_norm[:, :4].to(device)
    u = X_test_norm[:, 4:6].to(device)
    p = X_test_norm[:, 6:].to(device)

    y_norm_pred = model(x, u, p).detach().cpu().numpy()
    y_true     = Y_test.numpy()
    
y_std_np  = y_std.cpu().numpy()  # shape (1,4)
y_mean_np = y_mean.cpu().numpy() # shape (1,4)
y_pred = y_norm_pred * y_std_np + y_mean_np  # shape (batch,4)


residuals = y_pred - y_true

# Diagnostic Plot: Scatter y_true vs y_pred for each state
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
state_names = ['cA', 'cB', 'teta', 'teta_K']
for i, ax in enumerate(axes.flat):
    ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.5)
    ax.plot([y_true[:, i].min(), y_true[:, i].max()],
            [y_true[:, i].min(), y_true[:, i].max()], linestyle='--')
    ax.set_xlabel(f"True {state_names[i]}")
    ax.set_ylabel(f"Pred {state_names[i]}")
    ax.grid(False)
fig.tight_layout()
plt.savefig("pairplot_onestep_NN_CSTR.pdf", format="pdf")
plt.savefig("pairplot_onestep_NN_CSTR.svg", format="svg")
plt.show()



# Residual Analysis: Histogram of residuals for each state
fig2, axes2 = plt.subplots(2, 2, figsize=(8, 8))
for i, ax in enumerate(axes2.flat):
    ax.hist(residuals[:, i], bins=50, density=True)
    ax.set_title(f"Residuals for {state_names[i]}")
    ax.set_xlabel("Prediction Error")
    ax.set_ylabel("Density")
    ax.grid(False)
fig2.tight_layout()
plt.savefig("residualplot_onestep_NN_CSTR.pdf", format="pdf")
plt.savefig("residualplot_onestep_NN_CSTR.svg", format="svg")
plt.show()

