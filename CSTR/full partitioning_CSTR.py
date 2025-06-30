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


#constants
dt = 0.1 # [h] timestep
cA0s = 5.1 #2 # mol/l
k1_0 = 1.287*10**12 # [h^-1]
k2_0 = 1.287*10**12 # [h^-1]
k3_0 = 9.043*10**9 # [l/molA*h]
EA_1R_0 = 9758.3 # K
EA_2R_0 = 9758.3 # K
EA_3R_0 = 8560.0 #K
deltaH_AB_0 = 4.2 # kJ/mol A
deltaH_BC_0 = -11.0 # kJ/mol B
deltaH_AD_0 = -41.85 # kJ/mol A
rho_0 = 0.9342 # kg/l
Cp_0 = 3.01 # kJ/kg*K
Cpk_0 = 2.0 # kJ/(kg*K)
Ar = 0.215 # m^2
Vr = 10.0 # l
mk = 5.0 # kg
teta0 = 104.9 # °C
kw_0 = 4032 # kJ/h*m^2*K

#operating point 

cAs = 2.14 # mol/l
cBs = 1.09 # mol/l
tetas = 114.2 # °C
tetaKs = 112.9 # °C
Fs = 14.9 # h^-1
Qks = -1113.5 # kJ/h


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
rhs_teta_K.append((u_Qk+kw*Ar*(teta-teta_K))/(mk*Cpk))

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
ub_cA = 3 # mol/l given by the maximal inflow concentration of A
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

lb_F = 3 # h^-1
ub_F = 35 # h^-1
lb_Qk = -9000 #kJ/h
ub_Qk = 0.1 #kJ/h should be set 0,but in that case there would exist a division through 0

lb_u[0] = lb_F
ub_u[0] = ub_F
lb_u[1] = lb_Qk
ub_u[1] = ub_Qk


# scaling 
scaling_x = ub_x
scaling_u = ub_u

# create min and max decomposition functions 

cA_min = SX.sym('cA_min',1)
cA_max = SX.sym('cA_max',1)
cB_min = SX.sym('cB_min',1)
cB_max = SX.sym('cB_max',1)
teta_min = SX.sym('teta_min',1)
teta_max = SX.sym('teta_max',1)
teta_K_min = SX.sym('teta_K_min',1)
teta_K_max = SX.sym('teta_K_max',1)

uF = SX.sym('uF',1)
uQk = SX.sym('uQk',1)

cA0_min = SX.sym('cA0_min',1)
cA0_max = SX.sym('cA0_max',1)
k1_min = SX.sym('k1_min',1)
k1_max = SX.sym('k1_max',1)
k2_min = SX.sym('k2_min',1)
k2_max = SX.sym('k2_max',1)
k3_min = SX.sym('k3_min',1)
k3_max = SX.sym('k3_max',1)
deltaH_AB_min = SX.sym('deltaH_AB_min',1)
deltaH_AB_max = SX.sym('deltaH_AB_max',1)
deltaH_BC_min = SX.sym('deltaH_BC_min',1)
deltaH_BC_max = SX.sym('deltaH_BC_max',1)
deltaH_AD_min = SX.sym('deltaH_AD_min',1)
deltaH_AD_max = SX.sym('deltaH_AD_max',1)
rho_min = SX.sym('rho_min',1)
rho_max = SX.sym('rho_max',1)
Cp_min = SX.sym('Cp_min',1)
Cp_max = SX.sym('Cp_max',1)
Cpk_min = SX.sym('Cpk_min',1)
Cpk_max = SX.sym('Cpk_max',1)
kw_min = SX.sym('kw_min',1)
kw_max = SX.sym('kw_max',1)

rhs_cA_min = []
rhs_cB_min = []
rhs_teta_min = []
rhs_tetaK_min = []

rhs_cA_min.append(uF*(cA0_min-cA_min)-k1_max*exp(-EA_1R_0/(273.15+teta_max))*cA_min-k3_max*exp(-EA_3R_0/(273.15+teta_max))*cA_min*cA_min)
rhs_cB_min.append(-uF*cB_min+k1_min*exp(-EA_1R_0/(273.15+teta_min))*cA_min-k2_max*exp(-EA_2R_0/(273.15+teta_max))*cB_min)
rhs_teta_min.append(uF*(teta0-teta_min)+(kw_max*Ar*(teta_K_min-teta_min))/(rho_min*Cp_min*Vr)-(k1_max*exp(-EA_1R_0/(teta_min+273.15))*cA_max*deltaH_AB_max/(rho_min*Cp_min))-(k2_min*exp(-EA_2R_0/(teta_min+273.15))*cB_min*deltaH_BC_min/(rho_max*Cp_max))-(k3_min*exp(-EA_3R_0/(teta_min+273.15))*cA_min**2*deltaH_AD_min/(rho_max*Cp_max)))
rhs_tetaK_min.append((uQk+kw_min*Ar*(teta_min-teta_K_min))/(mk*Cpk_max))

dmin = Function('dmin',[vertcat(cA_min,cB_min,teta_min,teta_K_min),vertcat(cA_max,cB_max,teta_max,teta_K_max),vertcat(uF,uQk),vertcat(cA0_min,k1_min,k2_min,k3_min,deltaH_AB_min,deltaH_BC_min,deltaH_AD_min,rho_min,Cp_min,Cpk_min,kw_min),vertcat(cA0_max,k1_max,k2_max,k3_max,deltaH_AB_max,deltaH_BC_max,deltaH_AD_max,rho_max,Cp_max,Cpk_max,kw_max)],[vertcat(*rhs_cA_min,*rhs_cB_min,*rhs_teta_min,*rhs_tetaK_min)])

rhs_cA_max = []
rhs_cB_max = []
rhs_teta_max = []
rhs_tetaK_max = []

rhs_cA_max.append(uF*(cA0_max-cA_max)-k1_min*exp(-EA_1R_0/(273.15+teta_max))*cA_max-k3_min*exp(-EA_3R_0/(273.15+teta_max))*cA_max**2)
rhs_cB_max.append(-uF*cB_max+k1_max*exp(-EA_1R_0/(273.15+teta_max))*cA_max-k2_min*exp(-EA_2R_0/(273.15+teta_max))*cB_max)
rhs_teta_max.append(uF*(teta0-teta_max)+(kw_min*Ar*(teta_K_max-teta_max))/(rho_max*Cp_max*Vr)-(k1_min*exp(-EA_1R_0/(teta_max+273.15))*cA_min*deltaH_AB_min/(rho_max*Cp_max))-(k2_max*exp(-EA_2R_0/(teta_max+273.15))*cB_max*deltaH_BC_max/(rho_min*Cp_min))-(k3_max*exp(-EA_3R_0/(teta_max+273.15))*cA_max**2*deltaH_AD_max/(rho_min*Cp_min)))
rhs_tetaK_max.append((uQk+kw_max*Ar*(teta_max-teta_K_max))/(mk*Cpk_min))

dmax = Function('dmax',[vertcat(cA_min,cB_min,teta_min,teta_K_min),vertcat(cA_max,cB_max,teta_max,teta_K_max),vertcat(uF,uQk),vertcat(cA0_min,k1_min,k2_min,k3_min,deltaH_AB_min,deltaH_BC_min,deltaH_AD_min,rho_min,Cp_min,Cpk_min,kw_min),vertcat(cA0_max,k1_max,k2_max,k3_max,deltaH_AB_max,deltaH_BC_max,deltaH_AD_max,rho_max,Cp_max,Cpk_max,kw_max)],[vertcat(*rhs_cA_max,*rhs_cB_max,*rhs_teta_max,*rhs_tetaK_max)])

# create the simulator

simulator = do_mpc.simulator.Simulator(model)
simulator.set_param(t_step = dt)
p_template = simulator.get_tvp_template()

cA0_var = 2/17 # mol/l
k1_var = 40/1287 # l/mol A * h
k2_var = 40/1287 # l/mol A * h
k3_var = 270/9043 # l/mol A * h
deltaH_AB_var = 59/105 # kJ/mol A
deltaH_BC_var = 48/275 # kJ/mol B
deltaH_AD_var = 47/1395 # kJ/mol A
rho_var = 2/4671 # kg/l
Cp_var = 4/301 # kJ/kg K
Cpk_var = 1/40 # kJ/(kg*K)
kw_var = 5/168 # kJ/h*m^2*K

# p0 = np.concatenate((np.array([cA0s]),np.array([k1_0]),np.array([k2_0]),np.array([k3_0]),np.array([deltaH_AB_0]),np.array([deltaH_BC_0]),np.array([deltaH_AD_0]),np.array([rho_0]),np.array([Cp_0]),np.array([Cpk_0]),np.array([kw_0])),axis=0)
# p_max = np.concatenate((np.array([(1+cA0_var)*cA0s]),np.array([(1+k1_var)*k1_0]),np.array([(1-k2_var)*k2_0]),np.array([(1-k3_var)*k3_0]),np.array([(1-deltaH_AB_var)*deltaH_AB_0]),np.array([(1+deltaH_BC_var)*deltaH_BC_0]),np.array([(1+deltaH_AD_var)*deltaH_AD_0]),np.array([(1-rho_var)*rho_0]),np.array([(1-Cp_var)*Cp_0]),np.array([(1-Cpk_var)*Cpk_0]),np.array([(1+kw_var)*kw_0])),axis=0)
# p_min = np.concatenate((np.array([(1-cA0_var)*cA0s]),np.array([(1-k1_var)*k1_0]),np.array([(1+k2_var)*k2_0]),np.array([(1+k3_var)*k3_0]),np.array([(1+deltaH_AB_var)*deltaH_AB_0]),np.array([(1-deltaH_BC_var)*deltaH_BC_0]),np.array([(1-deltaH_AD_var)*deltaH_AD_0]),np.array([(1+rho_var)*rho_0]),np.array([(1+Cp_var)*Cp_0]),np.array([(1+Cpk_var)*Cpk_0]),np.array([(1-kw_var)*kw_0])),axis=0)


def p_fun_wcmax(t):
    p_template['cA0'] = (1+cA0_var)*cA0s
    p_template['k1'] = (1+k1_var)*k1_0
    p_template['k2'] = (1+k2_var)*k2_0
    p_template['k3'] = (1+k3_var)*k3_0
    p_template['deltaH_AB']= (1+deltaH_AB_var)*deltaH_AB_0
    p_template['deltaH_BC']= (1-deltaH_BC_var)*deltaH_BC_0
    p_template['deltaH_AD']= (1-deltaH_AD_var)*deltaH_AD_0
    p_template['rho'] = (1+rho_var)*rho_0
    p_template['Cp'] = (1+Cp_var)*Cp_0
    p_template['Cpk'] = (1+Cpk_var)*Cpk_0
    p_template['kw'] = (1+kw_var)*kw_0
    return p_template

def p_fun_wcmin(t):
    p_template['cA0'] = (1-cA0_var)*cA0s
    p_template['k1'] = (1-k1_var)*k1_0
    p_template['k2'] = (1-k2_var)*k2_0
    p_template['k3'] = (1-k3_var)*k3_0
    p_template['deltaH_AB']= (1-deltaH_AB_var)*deltaH_AB_0
    p_template['deltaH_BC']= (1+deltaH_BC_var)*deltaH_BC_0
    p_template['deltaH_AD']= (1+deltaH_AD_var)*deltaH_AD_0
    p_template['rho'] = (1-rho_var)*rho_0
    p_template['Cp'] = (1-Cp_var)*Cp_0
    p_template['Cpk'] = (1-Cpk_var)*Cpk_0
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

simulator.set_tvp_fun(p_fun_0)
simulator.setup()


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

# pass the ordering and the number of partitions in the different dimensions     

# seed defines the corresponding uncertainty case
# x0 = number of partitions in the first dimension(0)
# x1 = number of partitions in the second dimension(1)
# x2 = number of partitions in the third dimension(2)
# x3 = number of partitions in the fourth dimension(3) 
# ord1 = define the order the dimensions are partitioned 

def closed_loop_comp(seed,x0,x1,x2,x3,ord1):
    N = 30 #chosen same as in paper "Determinstic safety guarantees for learning-based control of monotone systems"
    N_RCIS = 1
    
    x = SX.sym('x',nx,1)
    u = SX.sym('u',nu,1)
    p = SX.sym('p',nd,1)

    #stage cost
    Q1 = 1; Q2 = 1; Q3 = 1; Q4 = 1; R1 = 1; R2 = 1; R3 = np.diag(np.ones(nu))
    
    stage_cost = (model.x['cA']-cAs).T@Q1@(model.x['cA']-cAs)+(model.x['cB']-cBs).T@Q2@(model.x['cB']-cBs)+(model.x['teta']-tetas).T@Q3@(model.x['teta']-tetas)+(model.x['teta_K']-tetaKs).T@Q4@(model.x['teta_K']-tetaKs)+(model.u['u_F']-Fs).T@R1@(model.u['u_F']-Fs)+(model.u['u_Qk']-Qks).T@R2@(model.u['u_Qk']-Qks)+(model.u-u).T@R3@(model.u-u)
    stage_cost_fcn = Function('stage_cost',[model.x,model.u,u],[stage_cost]) #(model.x['teta']-tetas).T@Q3@(model.x['teta']-tetas)+(model.x['teta_K']-tetaKs).T@Q4@(model.x['teta_K']-tetaKs)+(model.u['u_F']-Fs).T@R1@(model.u['u_F']-Fs)+(model.u['u_Qk']-Qks).T@R2@(model.u['u_Qk']-Qks)
    
    terminal_cost = 10*((model.x['cA']-cAs).T@Q1@(model.x['cA']-cAs)+(model.x['cB']-cBs).T@Q2@(model.x['cB']-cBs)+(model.x['teta']-tetas).T@Q3@(model.x['teta']-tetas)+(model.x['teta_K']-tetaKs).T@Q4@(model.x['teta_K']-tetaKs))
    terminal_cost_fcn = Function('terminal_cost',[model.x],[terminal_cost])
    
# define the functions that are used recursively to create the constraints for the cuts

    def flatten(xs):
        if isinstance(xs, list):
            res = []
            def loop(ys):
                for i in ys:
                    if isinstance(i, list):
                        loop(i)
                    else:
                        res.append(i);
            loop(xs)
        else:
            res=[xs]
        return res

    def depth(l):
        if isinstance(l, list):
            return 1 + max(depth(item) for item in l) if l else 1
        else:
            return 0 

# in this approach we will cut in each timestep in both dimensions having a constant number of subregions
# therefore only list and one cutting vector are necessary
# cutvector contains the number of cuts in the respective dimensions  
    
    cut1 = np.zeros((nx,1)) 
    cut1[0] = x0 # cuts in the first dimension (0,cA)
    cut1[1] = x1 # cuts in the second dimension (1,cB)
    cut1[2] = x2 # cuts in the third dimension (2,teta)
    cut1[3] = x3 # cuts in the fourth dimension (3,teta_K)

    ordering1 = ord1
    
# define number of subregions with the number of cuts in each dimensio# formula is valid if the total number of cuts for a specific dimension is the same in each subregion  
    ns = 1 # these are the initial values and they will be updated both in the following according to the respective cuts 
        
    for i in range(nx):
        ns*=(cut1[i]+1)
    ns = int(ns[0])
    
    def create_lis(cuts, ordering_dimension, dim=0, lis=[], last_idx=0):
        # Write a recursive function, that creates a search tree like list of lists with increasing numbers
        # The most inner list contains the index of the number of rectangles=number of cuts+1 of the dimension specified last in ordering dimension
        # dim is the dimension, that denotes the depth of recursion
        # The ordering dimension is a list of the dimensions, that are cut in the order of the list
        for i in range(int(cuts[ordering_dimension[dim]] + 1)):
            if dim == len(ordering_dimension) - 1:
                # print('Problem in last layer')
                lis.append(last_idx)
                last_idx += 1
            else:
                # print('Problem in layer '+str(dim+1)+' of '+str(len(ordering_dimension))+' layers')
                new_new_lis, last_idx = create_lis(cuts, ordering_dimension, dim + 1, [], last_idx)
                lis.append(new_new_lis)
        return lis, last_idx
    
    lis1 = create_lis(cut1, ordering1, dim=0, lis=[], last_idx=0)[0]
    
    opt_x = struct_symSX([
        entry('x_min', shape=nx, repeat=[N+1,ns,K+1]),
        entry('x_max', shape=nx, repeat=[N+1,ns,K+1]),
        entry('u', shape=nu, repeat=[N,ns]),
        entry('u_RCIS',shape=nu, repeat=[ns]),
        entry('x_min_RCIS',shape=nx, repeat=[ns,K+1]),
        entry('x_max_RCIS',shape=nx, repeat=[ns,K+1])
        ])
    
    
# set the bounds on opt_x  
    lb_opt_x = opt_x(0)
    ub_opt_x = opt_x(np.inf)



    lb_opt_x['x_min'] = lb_x/scaling_x
    ub_opt_x['x_min'] = ub_x/scaling_x
    lb_opt_x['x_max'] = lb_x/scaling_x
    ub_opt_x['x_max'] = ub_x/scaling_x


    lb_opt_x['u'] = lb_u/scaling_u
    ub_opt_x['u'] = ub_u/scaling_u

###############################################################################


    lb_opt_x['x_min_RCIS'] = lb_x/scaling_x
    lb_opt_x['x_max_RCIS'] = lb_x/scaling_x
    ub_opt_x['x_min_RCIS'] = ub_x/scaling_x
    ub_opt_x['x_max_RCIS'] = ub_x/scaling_x

    lb_opt_x['u_RCIS'] = lb_u/scaling_u
    ub_opt_x['u_RCIS'] = ub_u/scaling_u
# define the constraint function 

    def constraint_function(l,ord_dim,opt_x,i,h,lbg,ubg):
        for k in range(len(l)):
            idx=flatten(l[k])
            dim=ord_dim[-depth(l)]
            for s in idx:
                if s==idx[0] and k==0:
                    h.append(opt_x['x_min',i,s,0,dim]-opt_x['x_min',i,0,0,dim])
                    lbg.append(0)
                    ubg.append(0)
                else:
                    h.append(opt_x['x_min',i,s,0,dim]-opt_x['x_min',i,idx[0],0,dim])
                    lbg.append(0)
                    ubg.append(0)
                if s==idx[-1] and k==len(l)-1:###????
                    h.append(opt_x['x_max',i,s,0,dim]-opt_x['x_max',i,-1,0,dim])
                    lbg.append(0)
                    ubg.append(0)
                else:
                    h.append(opt_x['x_max',i,s,0,dim]-opt_x['x_max',i,idx[-1],0,dim])
                    lbg.append(0)
                    ubg.append(0)
            if k>=1:
                prev_last=flatten(l[k-1])[-1]
                h.append(opt_x['x_min',i,idx[0],0,dim]-opt_x['x_max',i,prev_last,0,dim])
                lbg.append(0)
                ubg.append(0)
            if depth(l) >1:
                h,lbg,ubg=constraint_function(l[k],ord_dim,opt_x,i,h,lbg,ubg)
        
        return h,lbg,ubg
    
    def constraint_function_RCIS(l,ord_dim,opt_x,h,lbg,ubg):
       for k in range(len(l)):
           idx=flatten(l[k])
           dim=ord_dim[-depth(l)]
           for s in idx:
               if s==idx[0] and k==0:
                   h.append(opt_x['x_min_RCIS',s,0,dim]-opt_x['x_min_RCIS',0,0,dim])
                   lbg.append(0)
                   ubg.append(0)
               else:
                   h.append(opt_x['x_min_RCIS',s,0,dim]-opt_x['x_min_RCIS',idx[0],0,dim])
                   lbg.append(0)
                   ubg.append(0)
               if s==idx[-1] and k==len(l)-1:###????
                   h.append(opt_x['x_max_RCIS',s,0,dim]-opt_x['x_max_RCIS',-1,0,dim])
                   lbg.append(0)
                   ubg.append(0)
               else:
                   h.append(opt_x['x_max_RCIS',s,0,dim]-opt_x['x_max_RCIS',idx[-1],0,dim])
                   lbg.append(0)
                   ubg.append(0)
           if k>=1:
               prev_last=flatten(l[k-1])[-1]
               h.append(opt_x['x_min_RCIS',idx[0],0,dim]-opt_x['x_max_RCIS',prev_last,0,dim])
               lbg.append(0)
               ubg.append(0)
           if depth(l) >1:
               h,lbg,ubg=constraint_function_RCIS(l[k],ord_dim,opt_x,h,lbg,ubg)
       
       return h,lbg,ubg
#  Set up the objective and the constraints of the problem
    J = 0 # cost fct for normal prediction horizon N
    g = []    # constraint expression g
    lb_g = []  # lower bound for constraint expression g
    ub_g = []  # upper bound for constraint expression g
    
    J_RCIS = 0
    g_RCIS = []
    lb_g_RCIS = []
    ub_g_RCIS = []
    
    x_init = SX.sym('x_init', nx,1)
    u_bef = SX.sym('u_bef',nu,1)
    p_plus = SX.sym('p_plus', nd,1)
    p_minus = SX.sym('p_minus', nd,1)
    x_RCIS_plus = SX.sym('x_RCIS_plus', nx, 1)
    x_RCIS_minus = SX.sym('x_RCIS_minus', nx, 1)
    
# Set initial constraints
    
    for s in range(ns):
        g.append(opt_x['x_min',0,s,0]-x_init)
        g.append(opt_x['x_max',0,s,0]-x_init)
        lb_g.append(np.zeros((2*nx,1)))
        ub_g.append(np.zeros((2*nx,1)))
        if s>0:
            g.append(opt_x['u',0,s]-opt_x['u',0,0])
            lb_g.append(np.zeros((nu,1)))
            ub_g.append(np.zeros((nu,1)))
    
# objective and equality constraints can be set together as number of subregions remains same   
    
    for i in range(N):
        # objective
        for s in range(ns):
            if i==0:
                J += stage_cost_fcn(opt_x['x_max',i,s,0]*scaling_x, opt_x['u',i,s]*scaling_u,u_bef)
                J += stage_cost_fcn(opt_x['x_min',i,s,0]*scaling_x, opt_x['u',i,s]*scaling_u,u_bef)
            else:
                J += stage_cost_fcn(opt_x['x_max',i,s,0]*scaling_x, opt_x['u',i,s]*scaling_u,opt_x['u',i-1,s]*scaling_u)
                J += stage_cost_fcn(opt_x['x_min',i,s,0]*scaling_x, opt_x['u',i,s]*scaling_u,opt_x['u',i-1,s]*scaling_u)

            # equality constraints + inequality constraints (system equation)
            for k in range(1,K+1):
                #x_next_max = -dt*system(opt_x['x_max',i,s,k]*scaling_x, opt_x['u',i,s]*scaling_u, p_plus)/scaling_x
                #x_next_min = -dt*system(opt_x['x_min',i,s,k]*scaling_x, opt_x['u',i,s]*scaling_u, p_minus)/scaling_x
                x_next_max = -dt*dmax(opt_x['x_min',i,s,k]*scaling_x,opt_x['x_max',i,s,k]*scaling_x, opt_x['u',i,s]*scaling_u, p_minus, p_plus)/scaling_x
                x_next_min = -dt*dmin(opt_x['x_min',i,s,k]*scaling_x,opt_x['x_max',i,s,k]*scaling_x, opt_x['u',i,s]*scaling_u, p_minus, p_plus)/scaling_x
                for j in range(K+1):
                    x_next_max += A[j,k]*opt_x['x_max',i,s,j]
                    x_next_min += A[j,k]*opt_x['x_min',i,s,j]
                g.append(x_next_max)
                g.append(x_next_min)
                lb_g.append(np.zeros((2*nx,1)))
                ub_g.append(np.zeros((2*nx,1)))

            x_next_plus = horzcat(*opt_x['x_max',i,s,:])@D
            x_next_minus = horzcat(*opt_x['x_min',i,s,:])@D
            g.append( opt_x['x_max', i+1,-1,0]-x_next_plus)
            g.append(x_next_minus - opt_x['x_min', i+1,0,0])
            lb_g.append(np.zeros((2*nx,1)))
            ub_g.append(np.ones((2*nx,1))*inf)
            
# terminal cost                

    for s in range(ns):
        J += terminal_cost_fcn(opt_x['x_max',-1, s, 0]*scaling_x)
        J += terminal_cost_fcn(opt_x['x_min',-1, s, 0]*scaling_x)
        

# partitionining of the predicted sets
        
   # g1 = copy.deepcopy(g)
   # lb_g1 = copy.deepcopy(lb_g)
   # ub_g1 = copy.deepcopy(ub_g)
    
    for i in range(1,N+1): 
        g, lb_g, ub_g = constraint_function(lis1,ordering1,opt_x,i,g,lb_g,ub_g)              
        for s in range(ns):
            g.append(opt_x['x_max',i,s,0]-opt_x['x_min',i,0,0])
            g.append(opt_x['x_max',i,-1,0]-opt_x['x_min',i,s,0])
            g.append(opt_x['x_min',i,s,0]-opt_x['x_min',i,0,0])
            g.append(opt_x['x_max',i,-1,0]-opt_x['x_max',i,s,0])
            lb_g.append(np.zeros((4*nx,1)))
            ub_g.append(np.ones((4*nx,1))*inf)
            
# computation of 1-step RCIS
    
    
    
    #for i in range(N_RCIS):
    for s in range(ns):
        for k in range(1,K+1):
            #x_next_max_RCIS = -dt*system(opt_x['x_max_RCIS',s,k]*scaling_x, opt_x['u_RCIS',s]*scaling_u, p_plus)/scaling_x
            #x_next_min_RCIS = -dt*system(opt_x['x_min_RCIS',s,k]*scaling_x, opt_x['u_RCIS',s]*scaling_u, p_minus)/scaling_x
            x_next_max_RCIS = -dt*dmax(opt_x['x_min_RCIS',s,k]*scaling_x,opt_x['x_max_RCIS',s,k]*scaling_x, opt_x['u_RCIS',s]*scaling_u, p_minus, p_plus)/scaling_x
            x_next_min_RCIS = -dt*dmin(opt_x['x_min_RCIS',s,k]*scaling_x,opt_x['x_max_RCIS',s,k]*scaling_x, opt_x['u_RCIS',s]*scaling_u, p_minus, p_plus)/scaling_x
            for j in range(K+1):
                x_next_max_RCIS += A[j,k]*opt_x['x_max_RCIS',s,j]
                x_next_min_RCIS += A[j,k]*opt_x['x_min_RCIS',s,j]

        g.append(x_next_max_RCIS)
        g.append(x_next_min_RCIS)
        lb_g.append(np.zeros((2*nx,1)))
        ub_g.append(np.zeros((2*nx,1)))
                
        x_next_plus_RCIS = horzcat(*opt_x['x_max_RCIS',s,:])@D
        x_next_minus_RCIS = horzcat(*opt_x['x_min_RCIS',s,:])@D
        g.append(opt_x['x_max_RCIS',-1,0]-x_next_plus_RCIS)
        g.append(x_next_minus_RCIS - opt_x['x_min_RCIS',0,0])
        lb_g.append(np.zeros((2*nx,1)))
        ub_g.append(inf*np.ones((2*nx,1)))

# Constraining for RCIS
# adjust the corresponding set of the RCIS so that x(N)ExRCIS holds

    for s in range(ns):
        g.append( opt_x['x_max_RCIS',-1,0]-opt_x['x_max', N, s,0])
        g.append(opt_x['x_min', N, s,0] - opt_x['x_min_RCIS',0,0])
        lb_g.append(np.zeros((2*nx,1)))
        ub_g.append(inf*np.ones((2*nx,1)))           

# Cutting RCIS
                
    g,lb_g,ub_g= constraint_function_RCIS(lis1,ordering1,opt_x,g,lb_g,ub_g)
        
    for s in range(ns):
        g.append(opt_x['x_max_RCIS',s,0]-opt_x['x_min_RCIS',0,0])
        g.append(opt_x['x_max_RCIS',-1,0]-opt_x['x_min_RCIS',s,0])
        g.append(opt_x['x_min_RCIS',s,0]-opt_x['x_min_RCIS',0,0])
        g.append(opt_x['x_max_RCIS',-1,0]-opt_x['x_max_RCIS',s,0])
        lb_g.append(np.zeros((4*nx,1)))
        ub_g.append(np.ones((4*nx,1))*inf)
    

# Concatenate constraints
    g = vertcat(*g)
    lb_g = vertcat(*lb_g)
    ub_g = vertcat(*ub_g) 

                             
# one solver is enough for the full-partitioning case since the  partitioning remains the same in each step 
# such that the solution of on prediction aligns already with the previous solution
# long term short: for the full-partitioning case two solvers will be used but they are the same since the problem is defined equally and the switching in the even/odd simulation steps will not have an effect

    prob = {'f':J,'x':vertcat(opt_x),'g':g, 'p':vertcat(x_init,p_plus,p_minus,u_bef)}
    mpc_mon_solver_cut = nlpsol('solver','ipopt',prob,{'ipopt.max_iter':4000,'ipopt.resto_failure_feasibility_threshold':1e-9,'ipopt.required_infeasibility_reduction':0.99,'ipopt.linear_solver':'MA57','ipopt.ma86_u':1e-6,'ipopt.print_level':3, 'ipopt.sb': 'yes', 'print_time':1,'ipopt.ma57_automatic_scaling':'yes','ipopt.ma57_pre_alloc':10,'ipopt.ma27_meminc_factor':100,'ipopt.ma27_pivtol':1e-4,'ipopt.ma27_la_init_factor':100})


# run the closed-loop

    if seed>2:
        simulator.set_tvp_fun(lambda t:p_fun_var(t,seed,True))
    elif seed==0:
        simulator.set_tvp_fun(p_fun_0)
    elif seed==1:
        simulator.set_tvp_fun(p_fun_wcmax)
    elif seed==2:
        simulator.set_tvp_fun(p_fun_wcmin)
    simulator.setup()    
    
    x_0 = np.array([[0],[0],[100],[98]])/scaling_x
    simulator.reset_history()
    simulator.x0 = x_0*ub_x
    uinitial = np.array([[14.9],[-1113.5]])/scaling_u
    opt_x_k = opt_x(1)
    
    N_sim = 20
    pmin = vertcat(p_fun_wcmin(0))
    pmax = vertcat(p_fun_wcmax(0))
    # pmin = p_min
    # pmax = p_max
    
    CL_time = time.time()
    opt = []
    sol = []
    subreg = []
 
    for j in range(N_sim):
    # solve optimization problem 
        print(j)

        if j>0:           
            mpc_res = mpc_mon_solver_cut(p=vertcat(x_0,pmax,pmin,u_k), x0=opt_x_k, lbg=lb_g, ubg=ub_g, lbx = lb_opt_x, ubx = ub_opt_x)
        else: # j = 0 
            print('Run a first iteration to generate good Warmstart Values')            
            mpc_res = mpc_mon_solver_cut(p=vertcat(x_0,pmax,pmin, uinitial), x0=opt_x_k, lbg=lb_g, ubg=ub_g, lbx = lb_opt_x, ubx = ub_opt_x)
            opt_x_k = opt_x(mpc_res['x'])
            
            mpc_res = mpc_mon_solver_cut(p=vertcat(x_0,pmax,pmin, uinitial), x0=opt_x_k, lbg=lb_g, ubg=ub_g, lbx = lb_opt_x, ubx = ub_opt_x)         
       
                
       
        opt_x_k = opt_x(mpc_res['x'])
        u_k = opt_x_k['u',0,0]*ub_u
    # simulate the system
        x_next = simulator.make_step(u_k)

    
    # Update the initial state
        x_0 = x_next/ub_x
    
        opt.append(copy.deepcopy(mpc_res))
        sol.append(copy.deepcopy(opt_x_k))
    subreg.append(ns)
    
    CL_time = time.time()-copy.deepcopy(CL_time) #time-measurement to evaluate the performance of the closed-loop simulation 
    mpc_mon_cut_res=copy.copy(simulator.data)
    
    datadict = {'CL': [mpc_mon_cut_res,CL_time], 'OL': [opt,sol], 'ns': [subreg]}
    
    return datadict             

dd = closed_loop_comp(0,1,1,1,1,[0,1,2,3])


#computation of closed-loop cost
R = np.array([[1,0],[0,1]])
Q = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
x_ref = np.array(np.array([2.14,1.09,114.2,112.9]))
def closed_loop_cost(data):
    N_sim = data['CL'][0]['_x','cA'].shape[0] #retrieve the number of simulation steps by 
    J_cl = np.zeros((N_sim+1,1))
    u_prev = np.zeros((2,1))
    
    for i in range(N_sim):
        
        x_i = np.vstack((
           data['CL'][0]['_x','cA'][i],
           data['CL'][0]['_x','cB'][i],
           data['CL'][0]['_x','teta'][i],
           data['CL'][0]['_x','teta_K'][i]
       ))
        
        u_i = np.vstack((
            data['CL'][0]['_u','u_F'][i],
            data['CL'][0]['_u','u_Qk'][i]
        ))
        
        J_cl[i+1] = J_cl[i] \
            + (x_i - x_ref.reshape((4,1))).T @ Q @ (x_i - x_ref.reshape((4,1))) \
            + (u_i - u_prev).T @ R @ (u_i - u_prev)
        u_prev = u_i
    
    return J_cl

# plot the simulation-trajectories of states (cA,cB,teta and teta_K) and the inputs (F and Qk) over the time (simulation timesteps)

size = 1
fig = plt.figure(layout="constrained",figsize=(size*70,size*25)) #width and height
ax_dict = fig.subplot_mosaic(
    [
        ["cA", "cB","teta","teta_K"],
        ["F","Qk","J_CL","X"]
    ],empty_sentinel="X", gridspec_kw = {"wspace" : 0.2, "hspace" : 0.3}
)

t_sim = dd['CL'][0]['_time']
N_sim = dd['CL'][0]['_time'].shape[0]
t_pred = np.arange(0,3.1,0.1)
N = np.array(dd['OL'][1][0]['x_max',:,-1,0,0]).shape[0]
cA_max = [np.array(dd['OL'][1][i]['x_max',:,-1,0,0]).reshape(N,1)*scaling_x[0] for i in range(0,N_sim)]
cA_min = [np.array(dd['OL'][1][i]['x_min',:,0,0,0]).reshape(N,1)*scaling_x[0] for i in range(0,N_sim)]
cB_max = [np.array(dd['OL'][1][i]['x_max',:,-1,0,1]).reshape(N,1)*scaling_x[1] for i in range(0,N_sim)]
cB_min = [np.array(dd['OL'][1][i]['x_min',:,0,0,1]).reshape(N,1)*scaling_x[1] for i in range(0,N_sim)]
teta_max = [np.array(dd['OL'][1][i]['x_max',:,-1,0,2]).reshape(N,1)*scaling_x[2] for i in range(0,N_sim)]
teta_min = [np.array(dd['OL'][1][i]['x_min',:,0,0,2]).reshape(N,1)*scaling_x[2] for i in range(0,N_sim)]
tetaK_max = [np.array(dd['OL'][1][i]['x_max',:,-1,0,3]).reshape(N,1)*scaling_x[3] for i in range(0,N_sim)]
tetaK_min = [np.array(dd['OL'][1][i]['x_min',:,0,0,3]).reshape(N,1)*scaling_x[3] for i in range(0,N_sim)]

# states 
# ax_dict["cA"].plot(t_sim,dd['CL'][0]['_x','cA'])
# ax_dict["cA"].plot(t_pred,cA_max[0],color='red')
# ax_dict["cA"].plot(t_pred,cA_min[0],color='green')
# ax_dict["cA"].axhline(y=cAs,color = 'red',linestyle ='-',linewidth=2)
# ax_dict["cA"].axhline(y=lb_cA ,color = 'red',linestyle ='--',linewidth=2)
# ax_dict["cA"].axhline(y=ub_cA ,color = 'red',linestyle ='--',linewidth=2)
# ax_dict["cA"].set_xlabel("time [h]")
# ax_dict["cA"].xaxis.labelpad = 20
# ax_dict["cA"].set_ylabel("cA [mol/l]",rotation = 0)
# ax_dict["cA"].yaxis.labelpad = 80
# ax_dict["cA"].grid(False)

# ax_dict["cB"].plot(t_sim,dd['CL'][0]['_x','cB'])
# ax_dict["cB"].plot(t_pred,cB_max[0],color='red')
# ax_dict["cB"].plot(t_pred,cB_min[0],color='green')
# ax_dict["cB"].axhline(y=cBs,color = 'red',linestyle ='-',linewidth=2)
# ax_dict["cB"].axhline(y=lb_cB ,color = 'red',linestyle ='--',linewidth=2)
# ax_dict["cB"].axhline(y=ub_cB ,color = 'red',linestyle ='--',linewidth=2)
# ax_dict["cB"].set_xlabel("time [h]", labelpad=1)
# ax_dict["cB"].xaxis.labelpad = 20
# ax_dict["cB"].set_ylabel("cB [mol/l]",rotation = 0, labelpad=12)
# ax_dict["cB"].yaxis.labelpad = 80
# ax_dict["cB"].grid(False)

# ax_dict["teta"].plot(t_sim,dd['CL'][0]['_x','teta'])
# ax_dict["teta"].plot(t_pred,teta_max[0],color='red')
# ax_dict["teta"].plot(t_pred,teta_min[0],color='green')
# ax_dict["teta"].axhline(y=tetas,color = 'red',linestyle ='-',linewidth=2)
# ax_dict["teta"].axhline(y=lb_teta ,color = 'red',linestyle ='--',linewidth=2)
# ax_dict["teta"].axhline(y=ub_teta ,color = 'red',linestyle ='--',linewidth=2)
# ax_dict["teta"].set_xlabel("time [h]", labelpad=1)
# ax_dict["teta"].xaxis.labelpad = 20
# ax_dict["teta"].set_ylabel("teta [°C]",rotation = 0, labelpad=12)
# ax_dict["teta"].yaxis.labelpad = 80
# ax_dict["teta"].grid(False)

# ax_dict["teta_K"].plot(t_sim,dd['CL'][0]['_x','teta_K'])
# ax_dict["teta_K"].plot(t_pred,tetaK_max[0],color='red')
# ax_dict["teta_K"].plot(t_pred,tetaK_min[0],color='green')
# ax_dict["teta_K"].axhline(y=tetaKs,color = 'red',linestyle ='-',linewidth=2)
# ax_dict["teta_K"].axhline(y=lb_teta_K ,color = 'red',linestyle ='--',linewidth=2)
# ax_dict["teta_K"].axhline(y=ub_teta_K ,color = 'red',linestyle ='--',linewidth=2)
# ax_dict["teta_K"].set_xlabel("time [h]", labelpad=1)
# ax_dict["teta_K"].xaxis.labelpad = 20
# ax_dict["teta_K"].set_ylabel("teta_K [°C]",rotation = 0, labelpad=12)
# ax_dict["teta_K"].yaxis.labelpad = 80
# ax_dict["teta_K"].grid(False)
# # inputs 
# ax_dict["F"].plot(t_sim,dd['CL'][0]['_u','u_F'])
# ax_dict["F"].axhline(y=Fs,color = 'red',linestyle ='-',linewidth=2)
# ax_dict["F"].axhline(y=lb_F ,color = 'red',linestyle ='--',linewidth=2)
# ax_dict["F"].axhline(y=ub_F ,color = 'red',linestyle ='--',linewidth=2)
# ax_dict["F"].set_xlabel("time [h]", labelpad=1)
# ax_dict["F"].xaxis.labelpad = 20
# ax_dict["F"].set_ylabel("F [l/h]",rotation = 0, labelpad=12)
# ax_dict["F"].yaxis.labelpad = 100
# ax_dict["F"].grid(False)

# ax_dict["Qk"].plot(t_sim,dd['CL'][0]['_u','u_Qk'])
# ax_dict["Qk"].axhline(y=Qks,color = 'red',linestyle ='-',linewidth=2)
# ax_dict["Qk"].axhline(y=lb_Qk ,color = 'red',linestyle ='--',linewidth=2)
# ax_dict["Qk"].axhline(y=ub_Qk ,color = 'red',linestyle ='--',linewidth=2)
# ax_dict["Qk"].set_xlabel("time [h]", labelpad=1)
# ax_dict["Qk"].xaxis.labelpad = 20
# ax_dict["Qk"].set_ylabel("Qk [kJ/h]",rotation = 0, labelpad=12)
# ax_dict["Qk"].yaxis.labelpad = 20
# ax_dict["Qk"].grid(False)

ax_dict["cA"].plot(t_sim,dd['CL'][0]['_x','cA'],label="cA_closed-loop")
# ax_dict["cA"].plot(t_pred,cA_max[0],color='red')
# ax_dict["cA"].plot(t_pred,cA_min[0],color='green')
ax_dict["cA"].axhline(y=cAs,color = 'red',linestyle ='-',linewidth=2)
ax_dict["cA"].axhline(y=lb_cA ,color = 'red',linestyle ='--',linewidth=2)
ax_dict["cA"].axhline(y=ub_cA ,color = 'red',linestyle ='--',linewidth=2)
ax_dict["cA"].set_xlabel("time [h]",labelpad=1,fontweight = 'bold',fontsize = 24)
ax_dict["cA"].xaxis.labelpad = 20
ax_dict["cA"].set_ylabel("cA [mol/l]",labelpad=12,rotation = 0,fontweight = 'bold',fontsize = 24)
ax_dict["cA"].yaxis.labelpad = 80
ax_dict["cA"].legend(fontsize ='xx-small',loc='upper right')
ax_dict["cA"].set_title('cA over time',fontweight = 'bold',fontsize = 24, pad = 12)
ax_dict["cA"].grid(False)

ax_dict["cB"].plot(t_sim,dd['CL'][0]['_x','cB'],label="cB_closed-loop")
# ax_dict["cB"].plot(t_pred,cB_max[0],color='red')
# ax_dict["cB"].plot(t_pred,cB_min[0],color='green')
ax_dict["cB"].axhline(y=cBs,color = 'red',linestyle ='-',linewidth=2)
ax_dict["cB"].axhline(y=lb_cB ,color = 'red',linestyle ='--',linewidth=2)
ax_dict["cB"].axhline(y=ub_cB ,color = 'red',linestyle ='--',linewidth=2)
ax_dict["cB"].set_xlabel("time [h]",labelpad=1,fontweight = 'bold',fontsize = 24)
ax_dict["cB"].xaxis.labelpad = 20
ax_dict["cB"].set_ylabel("cB [mol/l]",rotation = 0, labelpad=12,fontweight = 'bold',fontsize = 24 )
ax_dict["cB"].yaxis.labelpad = 80
ax_dict["cB"].legend(fontsize ='xx-small',loc='upper right')
ax_dict["cB"].set_title('cB over time',fontweight = 'bold',fontsize = 24, pad = 12)
ax_dict["cB"].grid(False)

ax_dict["teta"].plot(t_sim,dd['CL'][0]['_x','teta'],label="teta_closed-loop")
# ax_dict["teta"].plot(t_pred,teta_max[0],color='red')
# ax_dict["teta"].plot(t_pred,teta_min[0],color='green')
ax_dict["teta"].axhline(y=tetas,color = 'red',linestyle ='-',linewidth=2)
ax_dict["teta"].axhline(y=lb_teta ,color = 'red',linestyle ='--',linewidth=2)
ax_dict["teta"].axhline(y=ub_teta ,color = 'red',linestyle ='--',linewidth=2)
ax_dict["teta"].set_xlabel("time [h]", labelpad=1,fontweight = 'bold',fontsize = 24)
ax_dict["teta"].xaxis.labelpad = 20
ax_dict["teta"].set_ylabel("teta [°C]",rotation = 0, labelpad=12,fontweight = 'bold',fontsize = 24 )
ax_dict["teta"].yaxis.labelpad = 80
ax_dict["teta"].legend(fontsize ='xx-small',loc='upper right')
ax_dict["teta"].set_title('teta over time',fontweight = 'bold',fontsize = 24, pad = 12)
ax_dict["teta"].grid(False)

ax_dict["teta_K"].plot(t_sim,dd['CL'][0]['_x','teta_K'],label="teta_K_closed-loop")
# ax_dict["teta_K"].plot(t_pred,tetaK_max[0],color='red')
# ax_dict["teta_K"].plot(t_pred,tetaK_min[0],color='green')
ax_dict["teta_K"].axhline(y=tetaKs,color = 'red',linestyle ='-',linewidth=2)
ax_dict["teta_K"].axhline(y=lb_teta_K ,color = 'red',linestyle ='--',linewidth=2)
ax_dict["teta_K"].axhline(y=ub_teta_K ,color = 'red',linestyle ='--',linewidth=2)
ax_dict["teta_K"].set_xlabel("time [h]", labelpad=1,fontweight = 'bold',fontsize = 24)
ax_dict["teta_K"].xaxis.labelpad = 20
ax_dict["teta_K"].set_ylabel("teta_K [°C]",rotation = 0, labelpad=12,fontweight = 'bold',fontsize = 24)
ax_dict["teta_K"].yaxis.labelpad = 80
ax_dict["teta_K"].legend(fontsize ='xx-small',loc='upper right')
ax_dict["teta_K"].set_title('teta_K over time',fontweight = 'bold',fontsize = 24, pad = 12)
ax_dict["teta_K"].grid(False)
# inputs 
ax_dict["F"].step(t_sim,dd['CL'][0]['_u','u_F'],where = 'post',label="F_closed-loop")
ax_dict["F"].axhline(y=Fs,color = 'red',linestyle ='-',linewidth=2)
ax_dict["F"].axhline(y=lb_F ,color = 'red',linestyle ='--',linewidth=2)
ax_dict["F"].axhline(y=ub_F ,color = 'red',linestyle ='--',linewidth=2)
ax_dict["F"].set_xlabel("time [h]", labelpad=1,fontweight = 'bold',fontsize = 24)
ax_dict["F"].xaxis.labelpad = 20
ax_dict["F"].set_ylabel("F [l/h]",rotation = 0, labelpad=12,fontweight = 'bold',fontsize = 24)
ax_dict["F"].yaxis.labelpad = 100
ax_dict["F"].legend(fontsize ='xx-small',loc='upper right')
ax_dict["F"].set_title('F over time',fontweight = 'bold',fontsize = 24, pad = 12)
ax_dict["F"].grid(False)

ax_dict["Qk"].step(t_sim,dd['CL'][0]['_u','u_Qk'],where = 'post',label="Qk_closed-loop")
ax_dict["Qk"].axhline(y=Qks,color = 'red',linestyle ='-',linewidth=2)
ax_dict["Qk"].axhline(y=lb_Qk ,color = 'red',linestyle ='--',linewidth=2)
ax_dict["Qk"].axhline(y=ub_Qk ,color = 'red',linestyle ='--',linewidth=2)
ax_dict["Qk"].set_xlabel("time [h]", labelpad=1,fontweight = 'bold',fontsize = 24)
ax_dict["Qk"].xaxis.labelpad = 20
ax_dict["Qk"].set_ylabel("Qk [kJ/h]",rotation = 0, labelpad=12,fontweight = 'bold',fontsize = 24)
ax_dict["Qk"].yaxis.labelpad = 80
ax_dict["Qk"].legend(fontsize ='xx-small',loc='upper right')
ax_dict["Qk"].set_title('Qk over time',fontweight = 'bold',fontsize = 24, pad = 12)
ax_dict["Qk"].grid(False)

ax_dict["J_CL"].step(t_sim,closed_loop_cost(dd)[1:],where = 'post',label="J_closed-loop")
ax_dict["J_CL"].set_xlabel("time [h]", labelpad=1,fontweight = 'bold',fontsize = 24)
ax_dict["J_CL"].xaxis.labelpad = 20
ax_dict["J_CL"].set_ylabel("J_CL",rotation = 0, labelpad=12,fontweight = 'bold',fontsize = 24)
ax_dict["J_CL"].yaxis.labelpad = 80
ax_dict["J_CL"].legend(fontsize ='xx-small',loc='upper right')
ax_dict["J_CL"].set_title('closed-loop cost',fontweight = 'bold',fontsize = 24, pad = 12)
ax_dict["J_CL"].grid(False)
