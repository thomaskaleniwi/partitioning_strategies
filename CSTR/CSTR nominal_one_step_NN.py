# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 22:25:07 2025

@author: TKale
"""

# %% Import basic stuff

# Essentials
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pdb as pdb
import math
import subprocess
# Tools
from IPython.display import clear_output
import copy
import sys
# Specialized packages
from casadi import *
from casadi.tools import *
import control
import time as time
import os.path
import time as time
import multiprocessing as mp
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

# Custom packages
import do_mpc


# Customizing Matplotlib:
mpl.rcParams['font.size'] = 15
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['axes.grid'] = True
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['axes.unicode_minus'] = 'true'
mpl.rcParams['axes.labelsize'] = 'large'
mpl.rcParams['legend.fontsize'] = 'large'
mpl.rcParams['xtick.labelsize'] = 'large'
mpl.rcParams['ytick.labelsize'] = 'large'
mpl.rcParams['axes.labelpad'] = 6

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
        
        return x_next #cA and cB must be positive otherwise there is no physical senset be positive otherwise there is no physical sense

# here first load the weights and biases of the neural network before starting the full partitioning algorithm
dim_x, dim_u, dim_p = 4, 2, 11
k_mp = dim_x + dim_p
save_dir = 'D:/partitioning approaches_CSTR/CSTR_NN'
device = torch.device("cpu")
model_NN = CSTRDynamicsNN()
best_model_path = os.path.join(save_dir, 'best_model.pth')
state = torch.load(best_model_path)
model_NN.load_state_dict(state)
model_NN.eval()
# Path to data
inpath = os.path.join(save_dir, 'CSTR_data.pt')
# extract all data
data = torch.load(inpath)
xm = data['x_mean'].flatten()
xstd  = data['x_std'].flatten()
x_mean = xm[0:4]
u_mean = xm[4:6]
p_mean = xm[6:]
x_std  = xstd[0:4]
u_std  = xstd[4:6]
p_std  = xstd[6:]
y_mean = data['y_mean'].flatten()
y_std  = data['y_std'].flatten()

# helper function for normalization and denormalization
def normalize(x, mean, std):
    return (x - mean) / std

def denormalize(x_norm, mean, std):
    return x_norm * std + mean
# helper function for converting NN parameters into numpy arrays
def to_numpy(tensor):
    return tensor.detach().cpu().numpy()

# import weights and biases


W1a = to_numpy(model_NN.mon1a.weight)  # shape (hidden, k_mp)
mask1a = (W1a > 0)
W1a_pos = W1a*mask1a
b1a = to_numpy(model_NN.mon1a.bias)
W1b = to_numpy(model_NN.mon1b.weight)
mask2a = (W1b > 0)
W1b_pos = W1b*mask2a
b1b = to_numpy(model_NN.mon1b.bias)


W2a_raw = to_numpy(model_NN.mon2a.weight) # has to be split into positive and negative parts to ensure monotonicity
W_xp_2a, W_u_2a = W2a_raw[:, :k_mp], W2a_raw[:, k_mp:]
mask2a_pos = (W_xp_2a > 0)
W_xp_2a_pos = W_xp_2a*mask2a_pos
W2a = np.hstack((W_xp_2a_pos, W_u_2a))

b2a     = to_numpy(model_NN.mon2a.bias)
W2b     = to_numpy(model_NN.mon2b.weight)
#b2b     = to_numpy(model_NN.mon2b.bias)

# extract positive and negative weights from W2f to build decompositionfunction

mask2b_pos = (W2b > 0)
mask2b_neg = (W2b < 0)
W2b_pos = W2b*mask2b_pos
W2b_neg = W2b*mask2b_neg

# define casadi symbols

x = SX.sym("x", dim_x)
u = SX.sym("u", dim_u)
p = SX.sym("p", dim_p)
x1 = SX.sym("x1", dim_x)
p1 = SX.sym("p1", dim_p)
x2 = SX.sym("x2", dim_x)
p2 = SX.sym("p2", dim_p)
xp = vertcat(x, p)

# construct forward-passes for z1 and z2 each

h1a = tanh(W1a @ xp + b1a)
h1b = tanh(W1b @ h1a + b1b)

# h2a = tanh(W2a_raw @ h1b + b2a)
# h2b = W2b @ h2a + b2b
hu = vertcat(h1b, u)
z = tanh(W2a @ hu + b2a)

z_next = z # z-prediction for decomposition-fct
x_next = W2b @ z # state prediction

z_CSTR = Function("z_CSTR",[x, u, p],[z_next],["x", "u", "p"],["z_next"])
f_CSTR = Function("f_CSTR",[x, u, p],[x_next],["x", "u", "p"],["x_next"])

# forward pass for decomposition function
z1 = z_CSTR(x1,u,p1)
z2 = z_CSTR(x2,u,p2)

d_CSTR = W2b_pos @ z1 + W2b_neg @ z2

d = Function("d",[vertcat(x1), vertcat(x2), vertcat(u), vertcat(p1), vertcat(p2)], [vertcat(d_CSTR)], ["x1", "x2", "u", "p1", "p2"],["d_CSTR"] )


#constants
dt = 0.1 # [h] timestep
cA0s = 5.1 # mol/l
k1_0 = 1.287*10**12 # [h^-1]
k2_0 = 1.287*10**12 # [h^-1]
k3_0 = 9.043*10**9 # [l/molA*h]
EA_1R_0 = 9758.3 # K 1.2 as an alternative for using former setpoint
EA_2R_0 = 9758.3 # K 1.2
EA_3R_0 = 8560.0 # K 1.1
deltaH_AB_0 = 4.2 # kJ/mol A  also notice that changign this parameter makes the system exothermal such that the setpoint changes as well
deltaH_BC_0 = -11.0 # kJ/mol B
deltaH_AD_0 = -41.85 # kJ/mol A
rho_0 = 0.9342 # kg/l
Cp_0 = 3.01 # kJ/kg*K
Cpk_0 = 2.0 # kJ/(kg*K)
Ar = 0.215 # m^2
Vr = 10.0 # l
mk = 5.0 # kg
teta0 = 104.9 #130 # °C
kw_0 = 4032 # kJ/h*m^2*K

#operating point 

cAs = 2.14 #1.235 # mol/l
cBs = 1.09 #0.9 # mol/l
tetas = 114.2 #134.14  # °C
tetaKs = 112.9 #128.95 # °C
Fs = 14.19 #18.83  # h^-1
Qks = -1113.5 #-4495.7 # kJ/h


#system definitions
nx = 4 # number of states (Ca,Cb,deltaT,deltaT1)
nu = 2 # number of inputs (F,Qk)
nd = 11 # number of (uncertain) parameters (cA0s;k1;k2;k3;deltaH_AB;deltaH_BC;deltaH_AD;rho;Cp;Cpk;kw)

#Set up do-mpc model 
 
model_type = 'continuous'
model = do_mpc.model.Model(model_type)

# set the model states, inputs and parameter(s)

cA = model.set_variable(var_type='_x', var_name='cA', shape=(1,1))
cB = model.set_variable(var_type='_x', var_name='cB', shape=(1,1))
teta = model.set_variable(var_type='_x', var_name='teta', shape=(1,1))
teta_K = model.set_variable(var_type='_x', var_name='teta_K', shape=(1,1))

u_F = model.set_variable(var_type='_u', var_name='u_F',shape=(1,1))
u_Qk = model.set_variable(var_type='_u', var_name='u_Qk',shape=(1,1))

cA0 = model.set_variable(var_type='_tvp', var_name='cA0',shape=(1,1))
k1 = model.set_variable(var_type='_tvp', var_name='k1',shape=(1,1))
k2 = model.set_variable(var_type='_tvp', var_name='k2',shape=(1,1))
k3 = model.set_variable(var_type='_tvp', var_name='k3',shape=(1,1))
deltaH_AB = model.set_variable(var_type='_tvp', var_name='deltaH_AB',shape=(1,1))
deltaH_BC = model.set_variable(var_type='_tvp', var_name='deltaH_BC',shape=(1,1))
deltaH_AD = model.set_variable(var_type='_tvp', var_name='deltaH_AD',shape=(1,1))
rho = model.set_variable(var_type='_tvp', var_name='rho',shape=(1,1))
Cp = model.set_variable(var_type='_tvp', var_name='Cp',shape=(1,1))
Cpk = model.set_variable(var_type='_tvp', var_name='Cpk',shape=(1,1))
kw = model.set_variable(var_type='_tvp', var_name='kw',shape=(1,1))

# Set right-hand-side for ODE for all introduced states 
# Names are inherited from the state definition

rhs_cA = []
rhs_cB = []
rhs_teta = []
rhs_teta_K = []

rhs_cA.append(u_F*(cA0-cA)-k1*exp(-EA_1R_0/(teta+273.15))*cA-k3*exp(-EA_3R_0/(teta+273.15))*cA**2)
rhs_cB.append(-u_F*cB+k1*exp(-EA_1R_0/(teta+273.15))*cA-k2*exp(-EA_2R_0/(teta+273.15))*cB)
rhs_teta.append(u_F*(teta0-teta)+(kw*Ar*(teta_K-teta))/(rho*Cp*Vr)-(k1*exp(-EA_1R_0/(teta+273.15))*cA*deltaH_AB+k2*exp(-EA_2R_0/(teta+273.15))*cB*deltaH_BC+k3*exp(-EA_3R_0/(teta+273.15))*cA**2*deltaH_AD)/(rho*Cp))
rhs_teta_K.append(1/(mk*Cpk)*(u_Qk+kw*Ar*(teta-teta_K)))

model.set_rhs('cA',vertcat(*rhs_cA))
model.set_rhs('cB',vertcat(*rhs_cB))
model.set_rhs('teta',vertcat(*rhs_teta))
model.set_rhs('teta_K',vertcat(*rhs_teta_K))

# Setup model

model.setup()

# Get rhs-equations as Casadi-Function

system = Function('system',[model.x,model.u,model.tvp],[model._rhs])

# specifiy bounds on states and inputs 
lb_x = 0*np.ones((nx,1))
ub_x = np.ones((nx,1))

lb_u = 0*np.ones((nu,1))
ub_u = np.inf*np.ones((nu,1))

# state constraints

lb_cA = 0 # mol/l
ub_cA = 5.1 # mol/l given by the maximal inflow concentration of A
lb_cB = 0 # mol/l
ub_cB = 2 # mol/l 
lb_teta = 90 # °C
ub_teta = 120 # °C
lb_teta_K = 90 # °C
ub_teta_K = 120 # °C

lb_x[0] = lb_cA
ub_x[0] = ub_cA
lb_x[1] = lb_cB
ub_x[1] = ub_cB
lb_x[2] = lb_teta
ub_x[2] = ub_teta
lb_x[3] = lb_teta_K
ub_x[3] = ub_teta_K

# input constraints

lb_F = 3 #5 # h^-1
ub_F = 35 # h^-1
lb_Qk = -9000 # -8500  #kJ/h
ub_Qk = 0.1 #kJ/h should be set 0,but in that case there would exist a division through 0

lb_u[0] = lb_F
ub_u[0] = ub_F
lb_u[1] = lb_Qk
ub_u[1] = ub_Qk

# scaling 
scaling_x = ub_x 
scaling_u = ub_u 

# create the simulator

simulator = do_mpc.simulator.Simulator(model)
simulator.set_param(t_step = dt)
p_template = simulator.get_tvp_template()

cA0_var = 2/17 # mol/l
k1_var = 40/1287 # l/mol A * h
k2_var = 40/1287 # l/mol B * h
k3_var = 270/9043 # l/mol A * h
deltaH_AB_var = 59/105 # kJ/mol A
deltaH_BC_var = 48/275 # kJ/mol B
deltaH_AD_var = 47/1395 # kJ/mol A
rho_var = 2/4671 # kg/l
Cp_var = 4/301 # kJ/kg K
Cpk_var = 1/40 # kJ/(kg*K)
kw_var = 5/168 # kJ/h*m^2*K


def p_fun_maxhalf(t):
    p_template['cA0'] = (1+0.5*cA0_var)*cA0s
    p_template['k1'] = (1+0.5*k1_var)*k1_0
    p_template['k2'] = (1-0.5*k2_var)*k2_0
    p_template['k3'] = (1-0.5*k3_var)*k3_0
    p_template['deltaH_AB']= (1-0.5*deltaH_AB_var)*deltaH_AB_0
    p_template['deltaH_BC']= (1+0.5*deltaH_BC_var)*deltaH_BC_0
    p_template['deltaH_AD']= (1+0.5*deltaH_AD_var)*deltaH_AD_0
    p_template['rho'] = (1-0.5*rho_var)*rho_0
    p_template['Cp'] = (1-0.5*Cp_var)*Cp_0
    p_template['Cpk'] = (1-0.5*Cpk_var)*Cpk_0
    p_template['kw'] = (1+0.5*kw_var)*kw_0
    return p_template

def p_fun_minhalf(t):
    p_template['cA0'] = (1-0.5*cA0_var)*cA0s
    p_template['k1'] = (1-0.5*k1_var)*k1_0
    p_template['k2'] = (1+0.5*k2_var)*k2_0
    p_template['k3'] = (1+0.5*k3_var)*k3_0
    p_template['deltaH_AB']= (1+0.5*deltaH_AB_var)*deltaH_AB_0
    p_template['deltaH_BC']= (1-0.5*deltaH_BC_var)*deltaH_BC_0
    p_template['deltaH_AD']= (1-0.5*deltaH_AD_var)*deltaH_AD_0
    p_template['rho'] = (1+0.5*rho_var)*rho_0
    p_template['Cp'] = (1+0.5*Cp_var)*Cp_0
    p_template['Cpk'] = (1+0.5*Cpk_var)*Cpk_0
    p_template['kw'] = (1-0.5*kw_var)*kw_0
    return p_template

def p_max(t):
    p_template['cA0'] = (1+cA0_var)*cA0s
    p_template['k1'] = (1+k1_var)*k1_0
    p_template['k2'] = (1-k2_var)*k2_0
    p_template['k3'] = (1-k3_var)*k3_0
    p_template['deltaH_AB']= (1-deltaH_AB_var)*deltaH_AB_0
    p_template['deltaH_BC']= (1+deltaH_BC_var)*deltaH_BC_0
    p_template['deltaH_AD']= (1+deltaH_AD_var)*deltaH_AD_0
    p_template['rho'] = (1-rho_var)*rho_0
    p_template['Cp'] = (1-Cp_var)*Cp_0
    p_template['Cpk'] = (1-Cpk_var)*Cpk_0
    p_template['kw'] = (1+kw_var)*kw_0
    return p_template

def p_min(t):
    p_template['cA0'] = (1-cA0_var)*cA0s
    p_template['k1'] = (1-k1_var)*k1_0
    p_template['k2'] = (1+k2_var)*k2_0
    p_template['k3'] = (1+k3_var)*k3_0
    p_template['deltaH_AB']= (1+deltaH_AB_var)*deltaH_AB_0
    p_template['deltaH_BC']= (1-deltaH_BC_var)*deltaH_BC_0
    p_template['deltaH_AD']= (1-deltaH_AD_var)*deltaH_AD_0
    p_template['rho'] = (1+rho_var)*rho_0
    p_template['Cp'] = (1+Cp_var)*Cp_0
    p_template['Cpk'] = (1+Cpk_var)*Cpk_0
    p_template['kw'] = (1-kw_var)*kw_0
    return p_template
    
def p_fun_0(t):
    p_template['cA0'] = cA0s
    p_template['k1'] = k1_0
    p_template['k2'] = k2_0
    p_template['k3'] = k3_0
    p_template['deltaH_AB']= deltaH_AB_0
    p_template['deltaH_BC']= deltaH_BC_0
    p_template['deltaH_AD']= deltaH_AD_0
    p_template['rho'] = rho_0
    p_template['Cp'] = Cp_0
    p_template['Cpk'] = Cpk_0
    p_template['kw'] = kw_0
    return p_template

def interpol(a,b,fac):
    return a*fac+b*(1-fac)

def p_fun_var(t, in_seed,const=True):
    if const:
        np.random.seed(int(in_seed))
        rand=np.random.uniform(0,1,size=11)
        p_template['cA0'] = interpol(cA0s*(1+cA0_var),cA0s*(1-cA0_var),rand[0])
        p_template['k1'] = interpol(k1_0*(1+k1_var),k1_0*(1-k1_var),rand[1])
        p_template['k2'] = interpol(k2_0*(1+k2_var),k2_0*(1-k2_var),rand[2])
        p_template['k3'] = interpol(k3_0*(1+k3_var),k3_0*(1-k3_var),rand[3])
        p_template['deltaH_AB'] = interpol(deltaH_AB_0*(1+deltaH_AB_var),deltaH_AB_0*(1-deltaH_AB_var),rand[4])
        p_template['deltaH_BC'] = interpol(deltaH_BC_0*(1+deltaH_BC_var),deltaH_BC_0*(1-deltaH_BC_var),rand[5])
        p_template['deltaH_AD'] = interpol(deltaH_AD_0*(1+deltaH_AD_var),deltaH_AD_0*(1-deltaH_AD_var),rand[6])
        p_template['rho'] = interpol((1+rho_var)*rho_0,(1-rho_var)*rho_0,rand[7])
        p_template['Cp'] = interpol((1+Cp_var)*Cp_0,(1-Cp_var)*Cp_0,rand[8])
        p_template['Cpk'] = interpol((1+Cpk_var)*Cpk_0,(1-Cpk_var)*Cpk_0,rand[9])
        p_template['kw'] = interpol((1+kw_var)*kw_0,(1-kw_var)*kw_0,rand[10])
    else:
        np.random.seed(int(t//dt+in_seed))
        rand=np.random.uniform(0,1,size=11)
        p_template['cA0'] = interpol(cA0s*(1+cA0_var),cA0s*(1-cA0_var),rand[0])
        p_template['k1'] = interpol(k1_0*(1+k1_var),k1_0*(1-k1_var),rand[1])
        p_template['k2'] = interpol(k2_0*(1-k2_var),k2_0*(1+k2_var),rand[2])
        p_template['k3'] = interpol(k3_0*(1+k3_var),k3_0*(1-k3_var),rand[3])
        p_template['deltaH_AB'] = interpol(deltaH_AB_0*(1+deltaH_AB_var),deltaH_AB_0*(1-deltaH_AB_var),rand[4])
        p_template['deltaH_BC'] = interpol(deltaH_BC_0*(1+deltaH_BC_var),deltaH_BC_0*(1-deltaH_BC_var),rand[5])
        p_template['deltaH_AD'] = interpol(deltaH_AD_0*(1+deltaH_AD_var),deltaH_AD_0*(1-deltaH_AD_var),rand[6])
        p_template['rho'] = interpol((1+rho_var)*rho_0,(1-rho_var)*rho_0,rand[7])
        p_template['Cp'] = interpol((1+Cp_var)*Cp_0,(1-Cp_var)*Cp_0,rand[8])
        p_template['Cpk'] = interpol((1+Cpk_var)*Cpk_0,(1-Cpk_var)*Cpk_0,rand[9])
        p_template['kw'] = interpol((1+kw_var)*kw_0,(1-kw_var)*kw_0,rand[10])
    return p_template



# %% Set up orthogonal collocation for MPC


def L(tau_col, tau, j):
    l = 1
    for k in range(len(tau_col)):
        if k!=j:
            l *= (tau-tau_col[k])/(tau_col[j]-tau_col[k]) 
    return l


def LgrInter(tau_col, tau, xk):
    z = 0
    for j in range(len(tau_col)):
        z += L(tau_col, tau, j)*xk[j,:]

    return z



# collocation degree
K = 3

# collocation points (excluding 0)
tau_col = collocation_points(K,'radau')

# collocation points (including 0)
tau_col = [0]+tau_col


tau = SX.sym('tau')

A = np.zeros((K+1,K+1))

for j in range(K+1):
    dLj = gradient(L(tau_col, tau, j), tau)
    dLj_fcn = Function('dLj_fcn', [tau], [dLj])
    for k in range(K+1):
        A[j,k] = dLj_fcn(tau_col[k])

D = np.zeros((K+1,1))

for j in range(K+1):
    Lj = L(tau_col, tau, j)
    Lj_fcn = Function('Lj', [tau], [Lj])
    D[j] = Lj_fcn(1)
     

# seed defines the corresponding uncertainty case


def closed_loop_comp(seed,starting_point,flag):
    N = 30 #chosen same as in paper "Determinstic safety guarantees for learning-based control of monotone systems"
    N_RCIS = 1
    
    x = SX.sym('x',nx,1)
    u = SX.sym('u',nu,1)
    p = SX.sym('p',nd,1)

    #stage cost
    Q1 = 1; Q2 = 1; Q3 = 1; Q4 = 1; R1 = 1; R2 = 1; R3 = 1
    
    stage_cost = (model.x['cA']-cAs).T@Q1@(model.x['cA']-cAs)+(model.x['cB']-cBs).T@Q2@(model.x['cB']-cBs)+(model.u-u).T@R3@(model.u-u)+(model.x['teta']-tetas).T@Q3@(model.x['teta']-tetas)+(model.x['teta_K']-tetaKs).T@Q4@(model.x['teta_K']-tetaKs)+(model.u['u_F']-Fs).T@R1@(model.u['u_F']-Fs)+(model.u['u_Qk']-Qks).T@R2@(model.u['u_Qk']-Qks)
    stage_cost_fcn = Function('stage_cost',[model.x,model.u,u],[stage_cost]) #+(model.x['teta']-tetas).T@Q3@(model.x['teta']-tetas)+(model.x['teta_K']-tetaKs).T@Q4@(model.x['teta_K']-tetaKs)+(model.u['u_F']-Fs).T@R1@(model.u['u_F']-Fs)+(model.u['u_Qk']-Qks).T@R2@(model.u['u_Qk']-Qks)
    
    terminal_cost = 10*((model.x['cA']-cAs).T@Q1@(model.x['cA']-cAs)+(model.x['cB']-cBs).T@Q2@(model.x['cB']-cBs)+(model.x['teta']-tetas).T@Q3@(model.x['teta']-tetas)+(model.x['teta_K']-tetaKs).T@Q4@(model.x['teta_K']-tetaKs))
    terminal_cost_fcn = Function('terminal_cost',[model.x],[terminal_cost])
    
    
    opt_x = struct_symSX([
        entry('x', shape=nx, repeat=[N+1,K+1]),
        entry('u', shape=nu, repeat=[N])
        ])
    
    
# set the bounds on opt_x    
    lb_opt_x = opt_x(0)
    ub_opt_x = opt_x(np.inf)



    lb_opt_x['x'] = lb_x/scaling_x
    ub_opt_x['x'] = ub_x/scaling_x
    lb_opt_x['u'] = lb_u/scaling_u
    ub_opt_x['u'] = ub_u/scaling_u

###############################################################################

        
#  Set up the objective and the constraints of the problem
    J = 0 # cost fct for normal prediction horizon N
    g = []    # constraint expression g
    lb_g = []  # lower bound for constraint expression g
    ub_g = []  # upper bound for constraint expression g
    
    
    x_init = SX.sym('x_init', nx,1)
    u_bef = SX.sym('u_bef',nu,1)
    
# Set initial constraints
    
    
    g.append(opt_x['x',0,0]-x_init)
    lb_g.append(np.zeros((nx,1)))
    ub_g.append(np.zeros((nx,1)))
        
    
# objective and equality constraints can be set together as number of subregions remains same   
    
    for i in range(N):
        if i==0:
            J += stage_cost_fcn(opt_x['x',i,0]*scaling_x, opt_x['u',i]*scaling_u,u_bef)    
        else:
            J += stage_cost_fcn(opt_x['x',i,0]*scaling_x, opt_x['u',i]*scaling_u,opt_x['u',i-1]*scaling_u)

# equality constraints + inequality constraints (system equation)
        for k in range(1,K+1):
            x_next = -dt*system(opt_x['x',i,k]*scaling_x, opt_x['u',i]*scaling_u,p)/scaling_x
            for j in range(K+1):
                x_next += A[j,k]*opt_x['x',i,j]
            g.append(x_next)
            lb_g.append(np.zeros((nx,1)))
            ub_g.append(np.zeros((nx,1)))

        x_next_plus = horzcat(*opt_x['x',i])@D
        g.append(opt_x['x', i+1,0]-x_next_plus)
        lb_g.append(np.zeros((nx,1)))
        ub_g.append(np.ones((nx,1)))
            
# terminal cost                

    J += terminal_cost_fcn(opt_x['x',-1,0]*scaling_x)
        

    # Concatenate constraints
    g = vertcat(*g)
    lb_g = vertcat(*lb_g)
    ub_g = vertcat(*ub_g)

# one solver is enough for the full-partitioning case since the  partitioning remains the same in each step 
# such that the solution of on prediction aligns already with the previous solution
# long term short: for the full-partitioning case two solvers will be used but they are the same since the problem is defined equally and the switching in the even/odd simulation steps will not have an effect

    prob = {'f':J,'x':vertcat(opt_x),'g':g, 'p':vertcat(x_init,p,u_bef)}
    mpc_mon_solver_cut = nlpsol('solver','ipopt',prob,{'ipopt.max_iter':4000,'ipopt.resto_failure_feasibility_threshold':1e-9,'ipopt.required_infeasibility_reduction':0.99,'ipopt.linear_solver':'MA57','ipopt.ma86_u':1e-6,'ipopt.print_level':3, 'ipopt.sb': 'yes', 'print_time':1,'ipopt.ma57_automatic_scaling':'yes','ipopt.ma57_pre_alloc':10,'ipopt.ma27_meminc_factor':100,'ipopt.ma27_pivtol':1e-4,'ipopt.ma27_la_init_factor':100})


# run the closed-loop

    if seed>4:
        simulator.set_tvp_fun(lambda t:p_fun_var(t,seed,flag))
    elif seed==0:
        simulator.set_tvp_fun(p_fun_0)
    elif seed==1:
        simulator.set_tvp_fun(p_fun_maxhalf)
    elif seed==2:
        simulator.set_tvp_fun(p_fun_minhalf)
    elif seed==3:
        simulator.set_tvp_fun(p_max)
    elif seed==4:
        simulator.set_tvp_fun(p_min)
    simulator.setup()    
    
    x_0 = starting_point/scaling_x
    simulator.reset_history()
    simulator.x0 = x_0*ub_x
    uinitial = np.array([[14.9],[-1113.5]])/scaling_u
    opt_x_k = opt_x(1)
    
    N_sim = 20
    p = vertcat(p_fun_0(0))
    
    CL_time = time.time()
    opt = []
    sol = []
 
    for j in range(N_sim):
    # solve optimization problem 
        print(j)

        if j>0:           
            mpc_res = mpc_mon_solver_cut(p=vertcat(x_0,p,u_k), x0=opt_x_k, lbg=lb_g, ubg=ub_g, lbx = lb_opt_x, ubx = ub_opt_x)
        else: # j = 0 
            print('Run a first iteration to generate good Warmstart Values')            
            mpc_res = mpc_mon_solver_cut(p=vertcat(x_0,p,uinitial), x0=opt_x_k, lbg=lb_g, ubg=ub_g, lbx = lb_opt_x, ubx = ub_opt_x)
            opt_x_k = opt_x(mpc_res['x'])
            
            mpc_res = mpc_mon_solver_cut(p=vertcat(x_0,p,uinitial), x0=opt_x_k, lbg=lb_g, ubg=ub_g, lbx = lb_opt_x, ubx = ub_opt_x)        
       
                
       
        opt_x_k = opt_x(mpc_res['x'])
        u_k = opt_x_k['u',0]*ub_u
    # simulate the system
        x_next = simulator.make_step(u_k)

    
    # Update the initial state
        x_0 = x_next/ub_x
    

        opt.append(copy.deepcopy(mpc_res))
        sol.append(copy.deepcopy(opt_x_k))
  
    
    CL_time = time.time()-copy.deepcopy(CL_time) #time-measurement to evaluate the performance of the closed-loop simulation 
    mpc_mon_cut_res=copy.copy(simulator.data)
    
    datadict = {'CL': [mpc_mon_cut_res,CL_time], 'OL': [opt,sol]}
    
    return datadict             

dd = closed_loop_comp(4,np.array([[0],[0],[100],[100]]),False) # with "print(data.data_fields)" the full list of available fields can be inspected 

#make hundred runs with modified decomposition function in order to generate scatter plots of the closed-loop trajectory for many different uncertainty-scenarios
# sim_p = []
# t_sim = dd['CL'][0]['_time']
# for i in range(100):
#     sim_p.append(closed_loop_comp(i,np.array([[0],[0],[100],[100]]),False))

# size = 1
# fig = plt.figure(layout="constrained",figsize=(size*70,size*10)) #width and height
# ax_dict = fig.subplot_mosaic(
#     [
#         ["cA", "cB","teta","teta_K"],
#     ],empty_sentinel="X", gridspec_kw = {"wspace" : 0.2, "hspace" : 0.3}
# )

# for i in sim_p:
#     ax_dict["cA"].scatter(t_sim,i['CL'][0]['_x','cA'],label="cA_closed-loop")
#     ax_dict["cB"].scatter(t_sim,i['CL'][0]['_x','cB'],label="cB_closed-loop")
#     ax_dict["teta"].scatter(t_sim,i['CL'][0]['_x','teta'],label="teta_closed-loop")
#     ax_dict["teta_K"].scatter(t_sim,i['CL'][0]['_x','teta_K'],label="teta_K_closed-loop")

# ax_dict["cA"].grid(False)
# ax_dict["cA"].axhline(y=cAs,color = 'red',linestyle ='-',linewidth=2)
# ax_dict["cA"].axhline(y=lb_cA ,color = 'red',linestyle ='--',linewidth=2)
# ax_dict["cA"].axhline(y=ub_cA ,color = 'red',linestyle ='--',linewidth=2)
# ax_dict["cA"].set_xlabel("time [h]")
# ax_dict["cA"].xaxis.labelpad = 20
# ax_dict["cA"].set_ylabel("cA [mol/l]",rotation = 0)
# ax_dict["cA"].yaxis.labelpad = 80
# ax_dict["cB"].grid(False)
# ax_dict["cB"].axhline(y=cBs,color = 'red',linestyle ='-',linewidth=2)
# ax_dict["cB"].axhline(y=lb_cB ,color = 'red',linestyle ='--',linewidth=2)
# ax_dict["cB"].axhline(y=ub_cB ,color = 'red',linestyle ='--',linewidth=2)
# ax_dict["cB"].set_xlabel("time [h]", labelpad=1)
# ax_dict["cB"].xaxis.labelpad = 20
# ax_dict["cB"].set_ylabel("cB [mol/l]",rotation = 0, labelpad=12)
# ax_dict["cB"].yaxis.labelpad = 80
# ax_dict["teta"].grid(False)
# ax_dict["teta"].axhline(y=tetas,color = 'red',linestyle ='-',linewidth=2)
# ax_dict["teta"].axhline(y=lb_teta ,color = 'red',linestyle ='--',linewidth=2)
# ax_dict["teta"].axhline(y=ub_teta ,color = 'red',linestyle ='--',linewidth=2)
# ax_dict["teta"].set_xlabel("time [h]", labelpad=1)
# ax_dict["teta"].xaxis.labelpad = 20
# ax_dict["teta"].set_ylabel("teta [°C]",rotation = 0, labelpad=12)
# ax_dict["teta"].yaxis.labelpad = 80
# ax_dict["teta_K"].grid(False)
# ax_dict["teta_K"].axhline(y=tetaKs,color = 'red',linestyle ='-',linewidth=2)
# ax_dict["teta_K"].axhline(y=lb_teta_K ,color = 'red',linestyle ='--',linewidth=2)
# ax_dict["teta_K"].axhline(y=ub_teta_K ,color = 'red',linestyle ='--',linewidth=2)
# ax_dict["teta_K"].set_xlabel("time [h]", labelpad=1)
# ax_dict["teta_K"].xaxis.labelpad = 20
# ax_dict["teta_K"].set_ylabel("teta [°C]",rotation = 0, labelpad=12)
# ax_dict["teta_K"].yaxis.labelpad = 80

# plot the simulation-trajectories of states (cA,cB,teta and teta_K) and the inputs (F and Qk) over the time (simulation timesteps)
# also there are two more nominal trajectories plotted for the cases when the parameters reach their values with respect to the maximal or the minimal case 

for i in range(2): #create nominal trajectories with both pmax and pmin but for nominal inputs
    if i == 0:
        simulator.set_tvp_fun(p_max)
        simulator.setup()
    else: 
        simulator.set_tvp_fun(p_min)
        simulator.setup()

    x_0 = np.array([[0],[0],[30],[30]])/scaling_x
    simulator.reset_history()
    simulator.x0 = x_0*ub_x
    N_sim = 20

    for j in range(N_sim):
        u = dd['OL'][1][j]['u',0]*ub_u
        simulator.make_step(u)
    
    if i == 0:
        sim_max = copy.copy(simulator.data)
    else:
        sim_min = copy.copy(simulator.data)
            
##################################################################################
size = 1
fig = plt.figure(layout="constrained",figsize=(size*18,size*9)) #width and height
ax_dict = fig.subplot_mosaic(
    [
        ["cA", "cB"],
        ["teta","teta_K"]
    ],empty_sentinel="X", gridspec_kw = {"wspace" : 0.2, "hspace" : 0.3}
)


t_sim = dd['CL'][0]['_time']
N_sim = dd['CL'][0]['_time'].shape[0]
t_pred = np.arange(0,3.1,0.1)
N = np.array(dd['OL'][1][0]['x',:,0,0]).shape[0]



# use data from CL-run in order to generate trajectories from monotone NN and compare these trajectories with trajectories from the simulator
x_next_NN = []
x_sim = dd['CL'][0]['_x'] # same initial state as for the simulator
u_sim = dd['CL'][0]['_u']
p0 = np.concatenate((np.array([cA0s]),np.array([k1_0]),np.array([k2_0]),np.array([k3_0]),np.array([deltaH_AB_0]),np.array([deltaH_BC_0]),np.array([deltaH_AD_0]),np.array([rho_0]),np.array([Cp_0]),np.array([Cpk_0]),np.array([kw_0])),axis=0)
p_max = np.concatenate((np.array([(1+cA0_var)*cA0s]),np.array([(1+k1_var)*k1_0]),np.array([(1+k2_var)*k2_0]),np.array([(1+k3_var)*k3_0]),np.array([(1+deltaH_AB_var)*deltaH_AB_0]),np.array([(1+deltaH_BC_var)*deltaH_BC_0]),np.array([(1+deltaH_AD_var)*deltaH_AD_0]),np.array([(1+rho_var)*rho_0]),np.array([(1+Cp_var)*Cp_0]),np.array([(1+Cpk_var)*Cpk_0]),np.array([(1+kw_var)*kw_0])),axis=0)
p_min = np.concatenate((np.array([(1-cA0_var)*cA0s]),np.array([(1-k1_var)*k1_0]),np.array([(1-k2_var)*k2_0]),np.array([(1-k3_var)*k3_0]),np.array([(1-deltaH_AB_var)*deltaH_AB_0]),np.array([(1-deltaH_BC_var)*deltaH_BC_0]),np.array([(1-deltaH_AD_var)*deltaH_AD_0]),np.array([(1-rho_var)*rho_0]),np.array([(1-Cp_var)*Cp_0]),np.array([(1-Cpk_var)*Cpk_0]),np.array([(1-kw_var)*kw_0])),axis=0)

xm = data['x_mean'].flatten().numpy()
xstd  = data['x_std'].flatten().numpy()
x_mean = xm[0:4]
u_mean = xm[4:6]
p_mean = xm[6:]
x_std  = xstd[0:4]
u_std  = xstd[4:6]
p_std  = xstd[6:]
y_mean = data['y_mean'].flatten().numpy()
y_std  = data['y_std'].flatten().numpy()

# helper function for normalization and denormalization
def normalize(x, mean, std):
    return (x - mean) / std

def denormalize(x_norm, mean, std):
    return x_norm * std + mean

for q in range(N_sim):
    
    if q == 0:
        
        x0 = x_sim[0]        # erster tatsächlicher Simulator-Zustand  
        u0 = u_sim[0]  
        
        
        x_prev = normalize(x0, x_mean, x_std)
        u_prev = normalize(u0, u_mean, u_std)
        p_prev = normalize(p0, p_mean, p_std)
        p_prev_max = normalize(p_max, p_mean, p_std)
        p_prev_min = normalize(p_min, p_mean, p_std)
        
        x_preds = [x0]
        x_preds_min = [x0]
        x_preds_max = [x0]
        
        # x_nextd = x_sim[0]
        # x_next_NN.append(x_nextd)
        # x_prev = normalize(x_nextd,x_mean,x_std)
        # u_prev = normalize(u_sim[q],u_mean,u_std)
        # p_prev = normalize(p_prev,p_mean,p_std)
    else:
        
        x_norm = d(x_prev,x_prev, u_prev, p_prev,p_prev)
        x_norm_max = d(x_prev,x_prev, u_prev, p_prev_max,p_prev_min)
        x_norm_min = d(x_prev,x_prev, u_prev, p_prev_min,p_prev_max)
        
        x_norm = np.array(x_norm.full()).flatten()
        x_norm_max = np.array(x_norm_max.full()).flatten()
        x_norm_min = np.array(x_norm_min.full()).flatten()
        
        x_pred = denormalize(x_norm, y_mean, y_std)
        x_pred_max = denormalize(x_norm_max, y_mean, y_std)
        x_pred_min = denormalize(x_norm_min, y_mean, y_std)
        
        x_preds.append(x_pred)
        x_preds_max.append(x_pred_max)
        x_preds_min.append(x_pred_min)
        
        x_prev = normalize(x_pred, x_mean, x_std)
        x_prev_max = normalize(x_pred_max, x_mean, x_std)
        x_prev_min = normalize(x_pred_min, x_mean, x_std)
        u_now = u_sim[k]   # aus Closed-Loop-Datensatz
        u_prev = normalize(u_now, u_mean, u_std)
        
        # x_nextd = np.array(f_CSTR(x_prev,u_prev,p_prev)).flatten()
        # x_next_NN.append(denormalize(x_nextd,y_mean,y_std))
        # x_prev = normalize(x_nextd,x_mean,x_std)
        # u_prev = normalize(u_sim[q],u_mean,u_std)
    


cA_NN = []
cB_NN = []
teta_NN  = []
teta_K_NN = []

cA_NN_max = []
cB_NN_max = []
teta_NN_max  = []
teta_K_NN_max = []

cA_NN_min = []
cB_NN_min = []
teta_NN_min  = []
teta_K_NN_min = []

for xNN in x_preds:
    cA_NN.append(float(xNN[0]))  
    cB_NN.append(float(xNN[1]))
    teta_NN.append( float(xNN[2]))
    teta_K_NN.append(float(xNN[3]))
    
for xNN in x_preds_max:
    cA_NN_max.append(float(xNN[0]))  
    cB_NN_max.append(float(xNN[1]))
    teta_NN_max.append( float(xNN[2]))
    teta_K_NN_max.append(float(xNN[3]))
    
for xNN in x_preds_min:
    cA_NN_min.append(float(xNN[0]))  
    cB_NN_min.append(float(xNN[1]))
    teta_NN_min.append( float(xNN[2]))
    teta_K_NN_min.append(float(xNN[3]))



cA_NN     = np.array(cA_NN)
cB_NN     = np.array(cB_NN)
teta_NN   = np.array(teta_NN)
teta_K_NN = np.array(teta_K_NN)

cA_NN_max     = np.array(cA_NN_max)
cB_NN_max     = np.array(cB_NN_max)
teta_NN_max   = np.array(teta_NN_max)
teta_K_NN_max = np.array(teta_K_NN_max)

cA_NN_min     = np.array(cA_NN_min)
cB_NN_min     = np.array(cB_NN_min)
teta_NN_min   = np.array(teta_NN_min)
teta_K_NN_min = np.array(teta_K_NN_min)

# states 
ax_dict["cA"].plot(t_sim,dd['CL'][0]['_x','cA'],label = "cA_CL")
ax_dict["cA"].plot(t_sim,cA_NN,color='black',label = "cA_NN")
# ax_dict["cA"].plot(t_sim,cA_NN_max,color='red')
# ax_dict["cA"].plot(t_sim,cA_NN_min,color='green')
ax_dict["cA"].axhline(y=cAs,color = 'red',linestyle ='-',linewidth=2)
ax_dict["cA"].axhline(y=lb_cA ,color = 'red',linestyle ='--',linewidth=2)
ax_dict["cA"].axhline(y=ub_cA ,color = 'red',linestyle ='--',linewidth=2)
# ax_dict["cA"].plot(t_sim,sim_max['_x','cA'],label = "p_max",color = 'red')
# ax_dict["cA"].plot(t_sim,sim_min['_x','cA'],label = "p_min",color = 'green')
ax_dict["cA"].set_xlabel("time [h]")
ax_dict["cA"].xaxis.labelpad = 20
ax_dict["cA"].set_ylabel("cA [mol/l]",rotation = 0)
ax_dict["cA"].yaxis.labelpad = 80
ax_dict["cA"].grid(False)
ax_dict["cA"].legend(fontsize ='xx-small',loc='upper right')

ax_dict["cB"].plot(t_sim,dd['CL'][0]['_x','cB'],label = "cB_CL")
ax_dict["cB"].plot(t_sim,cB_NN,color='black',label = "cB_NN")
# ax_dict["cB"].plot(t_sim,cB_NN_max,color='red')
# ax_dict["cB"].plot(t_sim,cB_NN_min,color='green')
ax_dict["cB"].axhline(y=cBs,color = 'red',linestyle ='-',linewidth=2)
ax_dict["cB"].axhline(y=lb_cB ,color = 'red',linestyle ='--',linewidth=2)
ax_dict["cB"].axhline(y=ub_cB ,color = 'red',linestyle ='--',linewidth=2)
# ax_dict["cB"].plot(t_sim,sim_max['_x','cB'],label = "p_max",color = 'red')
# ax_dict["cB"].plot(t_sim,sim_min['_x','cB'],label = "p_min",color = 'green')
ax_dict["cB"].set_xlabel("time [h]", labelpad=1)
ax_dict["cB"].xaxis.labelpad = 20
ax_dict["cB"].set_ylabel("cB [mol/l]",rotation = 0, labelpad=12)
ax_dict["cB"].yaxis.labelpad = 80
ax_dict["cB"].grid(False)
ax_dict["cB"].legend(fontsize ='xx-small',loc='upper right')

ax_dict["teta"].plot(t_sim,dd['CL'][0]['_x','teta'],label = "teta_CL")
ax_dict["teta"].plot(t_sim,teta_NN,color='black',label = "teta_NN")
# ax_dict["teta"].plot(t_sim,teta_NN_max,color='red')
# ax_dict["teta"].plot(t_sim,teta_NN_min,color='green')
ax_dict["teta"].axhline(y=tetas,color = 'red',linestyle ='-',linewidth=2)
ax_dict["teta"].axhline(y=lb_teta ,color = 'red',linestyle ='--',linewidth=2)
ax_dict["teta"].axhline(y=ub_teta ,color = 'red',linestyle ='--',linewidth=2)
# ax_dict["teta"].plot(t_sim,sim_max['_x','teta'],label = "p_max",color = 'red')
# ax_dict["teta"].plot(t_sim,sim_min['_x','teta'],label = "p_min",color = 'green')
ax_dict["teta"].set_xlabel("time [h]", labelpad=1)
ax_dict["teta"].xaxis.labelpad = 20
ax_dict["teta"].set_ylabel("teta [°C]",rotation = 0, labelpad=12)
ax_dict["teta"].yaxis.labelpad = 80
ax_dict["teta"].grid(False)
ax_dict["teta"].legend(fontsize ='xx-small',loc='upper right')

ax_dict["teta_K"].plot(t_sim,dd['CL'][0]['_x','teta_K'],label = "teta_K_CL")
ax_dict["teta_K"].plot(t_sim,teta_K_NN,color='black',label="tetaK_NN")
# ax_dict["teta_K"].plot(t_sim,teta_K_NN_max,color='red')
# ax_dict["teta_K"].plot(t_sim,teta_K_NN_min,color='green')
ax_dict["teta_K"].axhline(y=tetaKs,color = 'red',linestyle ='-',linewidth=2)
ax_dict["teta_K"].axhline(y=lb_teta_K ,color = 'red',linestyle ='--',linewidth=2)
ax_dict["teta_K"].axhline(y=ub_teta_K ,color = 'red',linestyle ='--',linewidth=2)
# ax_dict["teta_K"].plot(t_sim,sim_max['_x','teta_K'],label = "p_max",color = 'red')
# ax_dict["teta_K"].plot(t_sim,sim_min['_x','teta_K'],label = "p_min",color = 'green')
ax_dict["teta_K"].set_xlabel("time [h]", labelpad=1)
ax_dict["teta_K"].xaxis.labelpad = 20
ax_dict["teta_K"].set_ylabel("teta_K [°C]",rotation = 0, labelpad=12)
ax_dict["teta_K"].yaxis.labelpad = 80
ax_dict["teta_K"].grid(False)
ax_dict["teta_K"].legend(fontsize ='xx-small',loc='upper right')
# inputs 
# ax_dict["F"].plot(t_sim,dd['CL'][0]['_u','u_F'],label = "F_CL")
# ax_dict["F"].axhline(y=Fs,color = 'red',linestyle ='-',linewidth=2)
# ax_dict["F"].axhline(y=lb_F ,color = 'red',linestyle ='--',linewidth=2)
# ax_dict["F"].axhline(y=ub_F ,color = 'red',linestyle ='--',linewidth=2)
# ax_dict["F"].set_xlabel("time [h]", labelpad=1)
# ax_dict["F"].xaxis.labelpad = 20
# ax_dict["F"].set_ylabel("F [l/h]",rotation = 0, labelpad=12)
# ax_dict["F"].yaxis.labelpad = 100
# ax_dict["F"].grid(False)
# ax_dict["F"].legend(fontsize ='xx-small',loc='upper right')

# ax_dict["Qk"].plot(t_sim,dd['CL'][0]['_u','u_Qk'],label = "Qk_CL")
# ax_dict["Qk"].axhline(y=Qks,color = 'red',linestyle ='-',linewidth=2)
# ax_dict["Qk"].axhline(y=lb_Qk ,color = 'red',linestyle ='--',linewidth=2)
# ax_dict["Qk"].axhline(y=ub_Qk ,color = 'red',linestyle ='--',linewidth=2)
# ax_dict["Qk"].set_xlabel("time [h]", labelpad=1)
# ax_dict["Qk"].xaxis.labelpad = 20
# ax_dict["Qk"].set_ylabel("Qk [kJ/h]",rotation = 0, labelpad=12)
# ax_dict["Qk"].yaxis.labelpad = 80
# ax_dict["Qk"].grid(False)
# ax_dict["Qk"].legend(fontsize ='xx-small',loc='upper right')
plt.savefig("nominal_one_step.pdf", format="pdf")
plt.savefig("nominal_one_step.svg", format="svg")
