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

# 2) lists for input and output for both the max_yieldB and the gain-scheduling trajectory
X_list1 = []
y_list1 = []

X_list2 = []
y_list2 = []

res_nominal_max_yieldB = dh_nominal_max_yieldB[:]
#res_nominal_gain = dh_nominal_gain[:]

for i in range(len(res_nominal_max_yieldB)):
# a) extract closed-loop data from the dict
# both res-objects have the same length such that index i still can be used
# here: i represents the current sample i
    mpc_data1 = res_nominal_max_yieldB[i]['CL_data']
    #mpc_data2 = res_nominal_gain[i]['CL_data']

# b) read out trajectories
# states x: (N_sim)×4
    x_traj1 = np.column_stack([
        mpc_data1['_x','cA'],
        mpc_data1['_x','cB'],
        mpc_data1['_x','teta'],
        mpc_data1['_x','teta_K']
    ])
    
    # x_traj2 = np.column_stack([
    #     mpc_data2['_x','cA'],
    #     mpc_data2['_x','cB'],
    #     mpc_data2['_x','teta'],
    #     mpc_data2['_x','teta_K']
    # ])
# control inputs u: (N_sim)×2
    u_traj1 = np.column_stack([
        mpc_data1['_u','u_F'],
        mpc_data1['_u','u_Qk']
    ])
    
    # u_traj2 = np.column_stack([
    #     mpc_data2['_u','u_F'],
    #     mpc_data2['_u','u_Qk']
    # ])
# Parameter p: (N_sim)×11
    p_traj1 = np.column_stack([
        mpc_data1['_tvp','cA0'],
        mpc_data1['_tvp','k1'],
        mpc_data1['_tvp','k2'],
        mpc_data1['_tvp','k3'],
        mpc_data1['_tvp','deltaH_AB'],
        mpc_data1['_tvp','deltaH_BC'],
        mpc_data1['_tvp','deltaH_AD'],
        mpc_data1['_tvp','rho'],
        mpc_data1['_tvp','Cp'],
        mpc_data1['_tvp','Cpk'],
        mpc_data1['_tvp','kw']
    ])
    
    # p_traj2 = np.column_stack([
    #     mpc_data2['_tvp','cA0'],
    #     mpc_data2['_tvp','k1'],
    #     mpc_data2['_tvp','k2'],
    #     mpc_data2['_tvp','k3'],
    #     mpc_data2['_tvp','deltaH_AB'],
    #     mpc_data2['_tvp','deltaH_BC'],
    #     mpc_data2['_tvp','deltaH_AD'],
    #     mpc_data2['_tvp','rho'],
    #     mpc_data2['_tvp','Cp'],
    #     mpc_data2['_tvp','Cpk'],
    #     mpc_data2['_tvp','kw']
    # ])

# c) for each time-step k: Input = [x_k, u_k, p_k], Label = x_{k+1}
    for k in range(x_traj1.shape[0] - 1):
        X_list1.append(np.concatenate([ x_traj1[k], u_traj1[k], p_traj1[k] ]))
        y_list1.append(       x_traj1[k+1]                          )
        
        # X_list2.append(np.concatenate([ x_traj2[k], u_traj2[k], p_traj2[k] ]))
        # y_list2.append(       x_traj2[k+1]                          )

# 3) concatenate lists to large arrays
X1 = np.vstack(X_list1)  # shape = (n_samples, 4+2+11)
Y1 = np.vstack(y_list1)  # shape = (n_samples, 4)
    
# X2 = np.vstack(X_list2)  # shape = (n_samples, 4+2+11)
# Y2 = np.vstack(y_list2)  # shape = (n_samples, 4)
    
# merge the two arrays from the two different trajectory runs into one for inputs and labels

# X = np.vstack((X1,X2)) # shape = (2*n_samples, 4+2+11)
# Y = np.vstack((Y1,Y2)) # shape = (2*n_samples, 4)
    

# 4) Split in train/val/test (70/15/15)
X_train, X_tmp, Y_train, Y_tmp = train_test_split(
    X1, Y1,
    test_size=0.30,
    random_state=42,  # ensures reproducability
    shuffle=True     
)
X_val, X_test, Y_val, Y_test = train_test_split(
    X_tmp, Y_tmp,
    test_size=0.50,
    random_state=42,
    shuffle=True
)

# --- Coverage‐Plots direkt nach dem train/test‐split ---
# 1) In ein DataFrame packen
cols_x = [f"x{i+1}" for i in range(4)]
cols_u = [f"u{i+1}" for i in range(2)]
cols_p = [f"p{i+1}" for i in range(11)]
cols_y = ['cA_next','cB_next','teta_next','tetaK_next']

# X_train.shape == (N,17), Y_train.shape == (N,4)
df_tr = pd.DataFrame(
    np.hstack([X_train, Y_train]),
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
g.fig.suptitle("Trainings–Coverage: Inputs vs cA_next_cB_next", y=1.02)
plt.savefig("Trainings_Coverage_Inputs_vs_cA_next_cB_next.pdf", format="pdf")
plt.savefig("Trainings_Coverage_Inputs_vs_cA_next/cB_next", format="svg")
plt.show()


#input plots
# 1) Spalten-Namen bauen
cols = [f"X{i+1}" for i in range(X_train.shape[1])]

# 2) Aus Deinem NumPy-Array ein pandas-DataFrame machen
df = pd.DataFrame(X_train, columns=cols)

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

save_dir = 'D:/partitioning approaches_CSTR/CSTR_NN/One-step'
os.makedirs(save_dir, exist_ok=True)

outpath = os.path.join(save_dir, 'CSTR_data.pt')
torch.save(
    {
      "X_train": torch.from_numpy(X_train).float(),
      "Y_train": torch.from_numpy(Y_train).float(),
      "X_val":   torch.from_numpy(X_val).float(),
      "Y_val":   torch.from_numpy(Y_val).float(),
      "X_test":  torch.from_numpy(X_test).float(),
      "Y_test":  torch.from_numpy(Y_test).float(),
    },
    outpath
)

print("Train:",      X_train.shape, Y_train.shape)
print("Validation:", X_val.shape,   Y_val.shape)
print("Test:",       X_test.shape,  Y_test.shape)

