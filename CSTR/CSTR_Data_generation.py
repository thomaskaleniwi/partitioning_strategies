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
teta0 = 104.9 #max: 104.9 #gain: 130 # °C
kw_0 = 4032 # kJ/h*m^2*K

#operating point 

cAs = 2.14 # max:2.14 # gain:1.235  # mol/l
cBs = 1.09 # max:1.09 # gain:0.9  # mol/l
tetas = 114.2# max:114.2 # gain:134.14    # °C
tetaKs = 112.9 # max:112.9 # gain:128.95  # °C
Fs = 14.19 # max: 14.19 # gain: 18.83  # h^-1
Qks = -1113.5 # max:-1113.5 # gain: -4495.7 # kJ/h


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
ub_teta = 150 # °C
lb_teta_K = 90 # °C
ub_teta_K = 150 # °C

lb_x[0] = lb_cA
ub_x[0] = ub_cA
lb_x[1] = lb_cB
ub_x[1] = ub_cB
lb_x[2] = lb_teta
ub_x[2] = ub_teta
lb_x[3] = lb_teta_K
ub_x[3] = ub_teta_K

# input constraints

lb_F = 3 # max:3 #gain:5 # h^-1
ub_F = 35 # h^-1
lb_Qk = -9000 # max: -9000 #gain:-8500  #kJ/h
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
deltaH_AB_var = 0.1*59/105 # kJ/mol A
deltaH_BC_var = 0.1*48/275 # kJ/mol B
deltaH_AD_var = 0.1*47/1395 # kJ/mol A
rho_var = 2/4671 # kg/l
Cp_var = 4/301 # kJ/kg K
Cpk_var = 1/40 # kJ/(kg*K)
kw_var = 5/168 # kJ/h*m^2*K

p0 = np.concatenate((np.array([cA0s]),np.array([k1_0]),np.array([k2_0]),np.array([k3_0]),np.array([deltaH_AB_0]),np.array([deltaH_BC_0]),np.array([deltaH_AD_0]),np.array([rho_0]),np.array([Cp_0]),np.array([Cpk_0]),np.array([kw_0])),axis=0)
#p_max = np.concatenate((np.array([(1+cA0_var)*cA0s]),np.array([(1+k1_var)*k1_0]),np.array([(1+k2_var)*k2_0]),np.array([(1+k3_var)*k3_0]),np.array([(1+deltaH_AB_var)*deltaH_AB_0]),np.array([(1+deltaH_BC_var)*deltaH_BC_0]),np.array([(1+deltaH_AD_var)*deltaH_AD_0]),np.array([(1+rho_var)*rho_0]),np.array([(1+Cp_var)*Cp_0]),np.array([(1+Cpk_var)*Cpk_0]),np.array([(1+kw_var)*kw_0])),axis=0)
#p_min = np.concatenate((np.array([(1-cA0_var)*cA0s]),np.array([(1-k1_var)*k1_0]),np.array([(1-k2_var)*k2_0]),np.array([(1-k3_var)*k3_0]),np.array([(1-deltaH_AB_var)*deltaH_AB_0]),np.array([(1-deltaH_BC_var)*deltaH_BC_0]),np.array([(1-deltaH_AD_var)*deltaH_AD_0]),np.array([(1-rho_var)*rho_0]),np.array([(1-Cp_var)*Cp_0]),np.array([(1-Cpk_var)*Cpk_0]),np.array([(1-kw_var)*kw_0])),axis=0)

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
        np.random.seed(int(t//dt+in_seed))# auskommentieren oder t//dt mit sehr hoher Zahl multiplizieren
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
    # add measurement-noise for inputs
    
        sigma_u   = 0.02
        noise_u   = np.random.randn(nu,1) * sigma_u * (ub_u - lb_u)
        u_meas    = u_k + noise_u
        x_next = simulator.make_step(u_meas)
        
    # add measurement-noise for states 
        sigma_x = 0.01
        noise_x = np.random.randn(nx,1) * sigma_x * (ub_x - lb_x)
        x_meas  = x_next + noise_x       
    
    # Update the initial state
        x_0 = x_meas/ub_x
    

        opt.append(copy.deepcopy(mpc_res))
        sol.append(copy.deepcopy(opt_x_k))
  
    
    CL_time = time.time()-copy.deepcopy(CL_time) #time-measurement to evaluate the performance of the closed-loop simulation 
    mpc_mon_cut_res=copy.copy(simulator.data)
    
    datadict = {'CL': [mpc_mon_cut_res,CL_time], 'OL': [opt,sol]}
    
    return datadict             

#dd = closed_loop_comp(0,np.array([[0],[0],[100],[100]]),False) # with "print(data.data_fields)" the full list of available fields can be inspected 


# use the do-mpc sampler to run multiple experiments of the nominal approach in order to collect enough data to train the CSTR neuronal network
# the sampler is going to be used twice with two different setpoints (once the setpoint from the gain-scheduling paper and once with the setpoint from yield maximization)

sp = do_mpc.sampling.SamplingPlanner()
sp.set_param(overwrite = False)
sp.data_dir = './nominal_max_yieldB/'

sp.set_sampling_var('seed')
sp.set_sampling_var('starting_point')
sp.set_sampling_var('flag')


seed = [0, 1, 2, 3, 4, 5, 6, 17, 1024, 2048, 4096,
  8192,  16384,  32768,  65536, 131072,
262144, 524288, 786432, 999999, 123456,
234567, 345678, 456789, 567890, 678901,
789012, 890123, 901234, 250000, 750000] # 30 different seeds
use_const = [True, False] # create a new random seed at each time-step (if False) and therefore initialize the parameters randomly

starting_point = [np.array([[0],[0],[30],[30]]),np.array([[0],[0],[50],[50]]),np.array([[0],[0],[100],[100]]),np.array([[0],[0],[120],[120]]),np.array([[0],[0],[200],[200]]),
                  np.array([[5.1],[0],[30],[30]]),np.array([[5.1],[0.8],[30],[30]]),np.array([[5.1],[0.8],[100],[100]]),np.array([[5.1],[0.8],[120],[120]]),np.array([[5.1],[0.8],[200],[200]]),
                  np.array([[5.1],[1.09],[30],[30]]),np.array([[5.1],[1.09],[100],[100]]),np.array([[5.1],[1.09],[120],[120]]),np.array([[5.1],[1.09],[200],[200]]),
                  np.array([[2.14],[0.8],[30],[30]]),np.array([[2.14],[0.8],[100],[100]]),np.array([[2.14],[0.8],[120],[120]]),np.array([[2.14],[0.8],[200],[200]]),
                  np.array([[2.14],[1.09],[30],[30]]),np.array([[2.14],[1.09],[100],[100]]),np.array([[2.14],[1.09],[120],[120]]),np.array([[2.14],[1.09],[200],[200]]),
                  np.array([[0],[0.8],[30],[30]]),np.array([[0],[0.8],[100],[100]]),np.array([[0],[0.8],[120],[120]]),np.array([[0],[0.8],[200],[200]]),
                  np.array([[0],[1.09],[30],[30]]),np.array([[0],[1.09],[100],[100]]),np.array([[0],[1.09],[120],[120]]),np.array([[0],[1.09],[200],[200]]),
                  np.array([[0],[4],[30],[30]]),np.array([[0],[4],[100],[100]]),np.array([[0],[4],[120],[120]]),np.array([[0],[4],[200],[200]]),
                  np.array([[2.14],[4],[30],[30]]),np.array([[2.14],[4],[100],[100]]),np.array([[2.14],[4],[120],[120]]),np.array([[2.14],[4],[200],[200]]),
                  np.array([[5.1],[4],[30],[30]]),np.array([[5.1],[4],[100],[100]]),np.array([[5.1],[4],[120],[120]]),np.array([[5.1],[4],[200],[200]]),
                  np.array([[0],[0.8],[30],[100]]),np.array([[0],[0.8],[30],[120]]),np.array([[0],[0.8],[30],[200]]),np.array([[0],[0.8],[100],[30]]),np.array([[0],[0.8],[120],[30]]),np.array([[0],[0.8],[200],[30]]),
                  np.array([[2.14],[1.09],[30],[100]]),np.array([[2.14],[1.09],[30],[120]]),np.array([[2.14],[1.09],[30],[200]]),np.array([[2.14],[1.09],[100],[30]]),np.array([[2.14],[1.09],[120],[30]]),np.array([[2.14],[1.09],[200],[30]]),
                  np.array([[2.14],[1.09],[30],[100]]),np.array([[2.14],[1.09],[30],[120]]),np.array([[2.14],[1.09],[30],[200]]),np.array([[2.14],[1.09],[100],[30]]),np.array([[2.14],[1.09],[120],[30]]),np.array([[2.14],[1.09],[200],[30]]),
                  np.array([[5.1],[1.09],[30],[100]]),np.array([[5.1],[1.09],[30],[120]]),np.array([[5.1],[1.09],[30],[200]]),np.array([[5.1],[1.09],[100],[30]]),np.array([[5.1],[1.09],[120],[30]]),np.array([[5.1],[1.09],[200],[30]]),
                  np.array([[5.1],[4],[30],[100]]),np.array([[5.1],[4],[30],[120]]),np.array([[5.1],[4],[30],[200]]),np.array([[5.1],[4],[100],[30]]),np.array([[5.1],[4],[120],[30]]),np.array([[5.1],[4],[200],[30]])]

# in order to cover more from the state space add random initial points

n_random = 88

for _ in range(n_random):
    cA0_r   = np.random.uniform(0.0, 5.1)
    cB0_r   = np.random.uniform(0.0, 2.0)
    teta0_r = np.random.uniform(30.0,200.0)
    tK0_r   = np.random.uniform(30.0,200.0)
    ip = np.array([[cA0_r],[cB0_r],[teta0_r],[tK0_r]])
    starting_point.append(ip)

# create a sampling plan with n = 9280 samples (27 uncertainty seeds x (72 fix starting points + 68 starting points) x 2 flag values (for time-varying seed)+(4 seeds(min/max) x (72 fix starting points + 68 starting points))                                    [x 2 set-points (without changing the inlet-temperature teta0)] rn without this
# choose only 1 setpoint, as otherwise the systemfct. is changed, but the NN does not incoporate this parameter as an input. This can lead to wrong or missleading predictions as the systemfct. is learned for a changed system that is not explicitly taken into account by a NN parameter
# in a loop add each combination of the sampling variables as a distinct case to the sampling plan  

seeds = seed                  # 30 different seeds
use_const = [True, False]                # turn parameter variation with time on or off
for s in seeds:
    for start in starting_point:
        if s >= 4:
            for const_flag in use_const:
                plan_gain = sp.add_sampling_case(
                        seed=s,
                        starting_point=start,
                        flag=const_flag
                        )
        else:
            plan_gain = sp.add_sampling_case(
                seed=s,
                starting_point=start,
                flag=True
                )
                    

# generate different sampling plans for different setpoints
sampler = do_mpc.sampling.Sampler(plan_gain)
sampler.set_param(overwrite = False)

sp.export('sampling_plan_nominal_max_yieldB')


def sample_function(seed,starting_point,flag):
    return closed_loop_comp(seed,starting_point,flag)

sampler.set_sample_function(sample_function)

sampler.data_dir = './nominal_max_yieldB/'
sampler.set_param(sample_name = 'nominal_max_yieldB')

# if __name__=="__main__":
# for s in seed:
#     for start in starting_point:
#         plan_maxB = sp.add_sampling_case(seed = s, starting_point = start)

#     sp.export('sampling_plan_nominal_max_yieldB')

# generate different sampling plans for different setpoints

#     sampler = do_mpc.sampling.Sampler(plan_maxB)
#     sampler.set_param(overwrite = False)


#     sampler.set_sample_function(sample_function)

#     sampler.data_dir = './nominal_max_yieldB/'
#     sampler.set_param(sample_name = 'nominal_max_yieldB')    
#     # number of samples in sampling plan
#     N = sampler.n_samples  
#     idx_list = list(range(N))

#     # pool with 16 processes (or mp.cpu_count())
#     with mp.Pool(processes=16) as pool:
#         # each worker conducts sampler.sample_idx(i) 
#         results = pool.map(sampler.sample_idx, idx_list)
sampler.sample_data()

# plot the simulation-trajectories of states (cA,cB,teta and teta_K) and the inputs (F and Qk) over the time (simulation timesteps)
# also there are two more nominal trajectories plotted for the cases when the parameters reach their values with respect to the maximal or the minimal case 

# for i in range(2): #create nominal trajectories with both pmax and pmin but for nominal inputs
#     if i == 0:
#         simulator.set_tvp_fun(p_max)
#         simulator.setup()
#     else: 
#         simulator.set_tvp_fun(p_min)
#         simulator.setup()

#     x_0 = np.array([[0],[0],[30],[30]])/scaling_x
#     simulator.reset_history()
#     simulator.x0 = x_0*ub_x
#     N_sim = 20

#     for j in range(N_sim):
#         u = dd['OL'][1][j]['u',0]*ub_u
#         simulator.make_step(u)
    
#     if i == 0:
#         sim_max = copy.copy(simulator.data)
#     else:
#         sim_min = copy.copy(simulator.data)
            
# ###################################################################################
# size = 1
# fig = plt.figure(layout="constrained",figsize=(size*70,size*25)) #width and height
# ax_dict = fig.subplot_mosaic(
#     [
#         ["cA", "cB","teta","teta_K"],
#         ["F","Qk","X","X"]
#     ],empty_sentinel="X", gridspec_kw = {"wspace" : 0.2, "hspace" : 0.3}
# )

# t_sim = dd['CL'][0]['_time']
# # states 
# ax_dict["cA"].plot(t_sim,dd['CL'][0]['_x','cA'],label = "nominal")
# ax_dict["cA"].axhline(y=cAs,color = 'red',linestyle ='-',linewidth=2)
# ax_dict["cA"].axhline(y=lb_cA ,color = 'red',linestyle ='--',linewidth=2)
# ax_dict["cA"].axhline(y=ub_cA ,color = 'red',linestyle ='--',linewidth=2)
# ax_dict["cA"].plot(t_sim,sim_max['_x','cA'],label = "p_max",color = 'red')
# ax_dict["cA"].plot(t_sim,sim_min['_x','cA'],label = "p_min",color = 'green')
# ax_dict["cA"].set_xlabel("time [h]")
# ax_dict["cA"].xaxis.labelpad = 20
# ax_dict["cA"].set_ylabel("cA [mol/l]",rotation = 0)
# ax_dict["cA"].yaxis.labelpad = 80
# ax_dict["cA"].grid(False)

# ax_dict["cB"].plot(t_sim,dd['CL'][0]['_x','cB'],label = "nominal")
# ax_dict["cB"].axhline(y=cBs,color = 'red',linestyle ='-',linewidth=2)
# ax_dict["cB"].axhline(y=lb_cB ,color = 'red',linestyle ='--',linewidth=2)
# ax_dict["cB"].axhline(y=ub_cB ,color = 'red',linestyle ='--',linewidth=2)
# ax_dict["cB"].plot(t_sim,sim_max['_x','cB'],label = "p_max",color = 'red')
# ax_dict["cB"].plot(t_sim,sim_min['_x','cB'],label = "p_min",color = 'green')
# ax_dict["cB"].set_xlabel("time [h]", labelpad=1)
# ax_dict["cB"].xaxis.labelpad = 20
# ax_dict["cB"].set_ylabel("cB [mol/l]",rotation = 0, labelpad=12)
# ax_dict["cB"].yaxis.labelpad = 80
# ax_dict["cB"].grid(False)

# ax_dict["teta"].plot(t_sim,dd['CL'][0]['_x','teta'],label = "nominal")
# ax_dict["teta"].axhline(y=tetas,color = 'red',linestyle ='-',linewidth=2)
# ax_dict["teta"].axhline(y=lb_teta ,color = 'red',linestyle ='--',linewidth=2)
# ax_dict["teta"].axhline(y=ub_teta ,color = 'red',linestyle ='--',linewidth=2)
# ax_dict["teta"].plot(t_sim,sim_max['_x','teta'],label = "p_max",color = 'red')
# ax_dict["teta"].plot(t_sim,sim_min['_x','teta'],label = "p_min",color = 'green')
# ax_dict["teta"].set_xlabel("time [h]", labelpad=1)
# ax_dict["teta"].xaxis.labelpad = 20
# ax_dict["teta"].set_ylabel("teta [°C]",rotation = 0, labelpad=12)
# ax_dict["teta"].yaxis.labelpad = 80
# ax_dict["teta"].grid(False)

# ax_dict["teta_K"].plot(t_sim,dd['CL'][0]['_x','teta_K'],label = "nominal")
# ax_dict["teta_K"].axhline(y=tetaKs,color = 'red',linestyle ='-',linewidth=2)
# ax_dict["teta_K"].axhline(y=lb_teta_K ,color = 'red',linestyle ='--',linewidth=2)
# ax_dict["teta_K"].axhline(y=ub_teta_K ,color = 'red',linestyle ='--',linewidth=2)
# ax_dict["teta_K"].plot(t_sim,sim_max['_x','teta_K'],label = "p_max",color = 'red')
# ax_dict["teta_K"].plot(t_sim,sim_min['_x','teta_K'],label = "p_min",color = 'green')
# ax_dict["teta_K"].set_xlabel("time [h]", labelpad=1)
# ax_dict["teta_K"].xaxis.labelpad = 20
# ax_dict["teta_K"].set_ylabel("teta [°C]",rotation = 0, labelpad=12)
# ax_dict["teta_K"].yaxis.labelpad = 80
# ax_dict["teta_K"].grid(False)
# # inputs 
# ax_dict["F"].plot(t_sim,dd['CL'][0]['_u','u_F'],label = "nominal")
# ax_dict["F"].axhline(y=Fs,color = 'red',linestyle ='-',linewidth=2)
# ax_dict["F"].axhline(y=lb_F ,color = 'red',linestyle ='--',linewidth=2)
# ax_dict["F"].axhline(y=ub_F ,color = 'red',linestyle ='--',linewidth=2)
# ax_dict["F"].set_xlabel("time [h]", labelpad=1)
# ax_dict["F"].xaxis.labelpad = 20
# ax_dict["F"].set_ylabel("F [l/h]",rotation = 0, labelpad=12)
# ax_dict["F"].yaxis.labelpad = 100
# ax_dict["F"].grid(False)

# ax_dict["Qk"].plot(t_sim,dd['CL'][0]['_u','u_Qk'],label = "nominal")
# ax_dict["Qk"].axhline(y=Qks,color = 'red',linestyle ='-',linewidth=2)
# ax_dict["Qk"].axhline(y=lb_Qk ,color = 'red',linestyle ='--',linewidth=2)
# ax_dict["Qk"].axhline(y=ub_Qk ,color = 'red',linestyle ='--',linewidth=2)
# ax_dict["Qk"].set_xlabel("time [h]", labelpad=1)
# ax_dict["Qk"].xaxis.labelpad = 20
# ax_dict["Qk"].set_ylabel("Qk [kJ/h]",rotation = 0, labelpad=12)
# ax_dict["Qk"].yaxis.labelpad = 20
# ax_dict["Qk"].grid(False)

