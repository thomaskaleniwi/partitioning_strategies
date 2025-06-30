# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 17:09:41 2025

@author: TKale
"""

import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import do_mpc
import torch#
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1) directory with .pkl-data

plan_nominal_max_yieldB = np.load('D:/partitioning approaches_CSTR/CSTR_NN/nominal_max_yieldB/sampling_plan_nominal_max_yieldB.pkl',allow_pickle=True)  
dh_nominal_max_yieldB = do_mpc.sampling.DataHandler(plan_nominal_max_yieldB)   
dh_nominal_max_yieldB.data_dir = 'D:/partitioning approaches_CSTR/CSTR_NN/nominal_max_yieldB/'
dh_nominal_max_yieldB.set_param(sample_name = 'nominal_max_yieldB')

dh_nominal_max_yieldB.set_post_processing('CL_data',lambda data: data['CL'][0])

# plan_nominal_gain = np.load('D:/partitioning approaches_CSTR/nominal_gain/sampling_plan_nominal_gain.pkl',allow_pickle=True)
# dh_nominal_gain = do_mpc.sampling.DataHandler(plan_nominal_gain)
# dh_nominal_gain.data_dir = 'D:/partitioning approaches_CSTR/nominal_gain/'
# dh_nominal_gain.set_param(sample_name = 'nominal_gain')

# dh_nominal_gain.set_post_processing('CL_data',lambda data: data['CL'][0])

# 2) extract sequences
X_seq_list = []
Y_seq_list = []

# X_list2 = []
# y_list2 = []

res_nominal_max_yieldB = dh_nominal_max_yieldB[:]
#res_nominal_gain = dh_nominal_gain[:]

for case in res_nominal_max_yieldB:
    cl = case['CL_data']
    x_traj = np.column_stack([ cl['_x','cA'], cl['_x','cB'], cl['_x','teta'], cl['_x','teta_K'] ])  # (T,4)
    u_traj = np.column_stack([ cl['_u','u_F'], cl['_u','u_Qk'] ])                                # (T,2)
    p_traj = np.column_stack([ cl['_tvp','cA0'], cl['_tvp','k1'], cl['_tvp','k2'], cl['_tvp','k3'],
                               cl['_tvp','deltaH_AB'], cl['_tvp','deltaH_BC'], cl['_tvp','deltaH_AD'],
                               cl['_tvp','rho'], cl['_tvp','Cp'], cl['_tvp','Cpk'], cl['_tvp','kw'] ])  # (T,11)
    T = x_traj.shape[0]
    H = T-1
    # für jeden möglichen Startindex eine Länge-H Sequenz bauen:
    for k in range(T - H):
        # Input: [x_k, u_k, p_k] für k..k+H-1
        seq_in = np.hstack([
            x_traj[k:k+H],       # (H,4)
            u_traj[k:k+H],       # (H,2)
            p_traj[k:k+H]        # (H,11)
        ])                        # => (H,17)
        # Target: x_{k+1..k+H}
        seq_out = x_traj[k+1:k+H+1]  # (H,4)
        X_seq_list.append(seq_in)
        Y_seq_list.append(seq_out)

X_seq = np.stack(X_seq_list, axis=0)  # (n_seq, H, 17)
Y_seq = np.stack(Y_seq_list, axis=0)
    
# --- Train/Val/Test Split ---
X_tr, X_tmp, Y_tr, Y_tmp = train_test_split(
    X_seq, Y_seq, test_size=0.30, random_state=42, shuffle=True
)
X_val, X_test, Y_val, Y_test = train_test_split(
    X_tmp, Y_tmp, test_size=0.50, random_state=42, shuffle=True
)

# 1) Flatten über die Sequenzdimension:
#    Aus (N_seq, H, input_dim) → (N_seq*H, input_dim)
N, H, inp = X_tr.shape
_, _, out = Y_tr.shape
Xf = X_tr.reshape(N*H, inp)
Yf = Y_tr.reshape(N*H, out)

# --- Coverage‐Plots direkt nach dem train/test‐split ---
# 1) In ein DataFrame packen
cols_x = [f"x{i+1}" for i in range(4)]
cols_u = [f"u{i+1}" for i in range(2)]
cols_p = [f"p{i+1}" for i in range(11)]
cols_y = ['cA_next','cB_next','teta_next','tetaK_next']

# X_train.shape == (N,17), Y_train.shape == (N,4)
df_tr = pd.DataFrame(
    np.hstack([Xf, Yf]),
    columns=cols_x + cols_u + cols_p + cols_y
)

# 2) Pairplot für Inputs → cA_next und cB_next
sns.set(style="whitegrid", context="talk")
g = sns.pairplot(
    df_tr,
    x_vars=cols_x+cols_u+cols_p,
    y_vars=['cA_next','cB_next','teta_next','tetaK_next'],
    kind='scatter',
    plot_kws=dict(alpha=0.3, s=20)
)
g.fig.suptitle("Trainings–Coverage: Inputs vs cA_next/cB_next", y=1.02)
plt.show()

#input plots
# 1) Spalten-Namen bauen
cols = [f"X{i+1}" for i in range(inp)]

# 2) Aus Deinem NumPy-Array ein pandas-DataFrame machen
df = pd.DataFrame(Xf, columns=cols)

n = df.shape[1]
fig, axes = plt.subplots(n, n, figsize=(3*n, 3*n))

for i, xi in enumerate(cols):
    for j, xj in enumerate(cols):
        ax = axes[i, j]
        if i == j:
            # Diagonale: xi gegen xi
            ax.scatter(df[xi], df[xi], s=20, alpha=0.3)
            ax.plot(df[xi], df[xi], color='gray', lw=1)  # 45°-Linie
        else:
            ax.scatter(df[xj], df[xi], s=20, alpha=0.3)
        if i == n-1:
            ax.set_xlabel(xj)
        else:
            ax.set_xticks([])
        if j == 0:
            ax.set_ylabel(xi)
        else:
            ax.set_yticks([])

fig.tight_layout()
plt.show()

save_dir = 'D:/partitioning approaches_CSTR/CSTR_NN/Multi-step'
os.makedirs(save_dir, exist_ok=True)

outpath = os.path.join(save_dir, 'CSTR_data_multi.pt')
torch.save(
    {
      "X_train": torch.from_numpy(X_tr).float(),
      "Y_train": torch.from_numpy(Y_tr).float(),
      "X_val":   torch.from_numpy(X_val).float(),
      "Y_val":   torch.from_numpy(Y_val).float(),
      "X_test":  torch.from_numpy(X_test).float(),
      "Y_test":  torch.from_numpy(Y_test).float(),
    },
    outpath
)

print("Train:",      X_tr.shape, Y_tr.shape)
print("Validation:", X_val.shape,   Y_val.shape)
print("Test:",       X_test.shape,  Y_test.shape)

