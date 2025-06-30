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
cA0s = 5.1 # mol/l
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

cAs = 2.14 #mol/l
cBs = 1.09 #mol/l
tetas = 114.2 # °C
tetaKs = 112.95 # °C
Fs = 14.19 # h^-1
Qks = -1113.5 # kJ/h


#system definitions
nx = 4 # number of states (Ca,Cb,deltaT,deltaT1)
nu = 2 # number of inputs (F,Qk)
nd = 11 # number of (uncertain) parameters (cA0s;k1;k2;k3;deltaH_AB;deltaH_BC;deltaH_AD;rho;Cp;Cpk;kw)

model_type = 'continuous'
model = do_mpc.model.Model(model_type)


# Set model states, inputs and parameter(s)

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
ub_cA = 5 # mol/l given by the maximal inflow concentration of A
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



# Create the simulator

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

#p0 = np.concatenate((np.array([cA0s]),np.array([k1_0]),np.array([k2_0]),np.array([k3_0]),np.array([deltaH_AB_0]),np.array([deltaH_BC_0]),np.array([deltaH_AD_0]),np.array([rho_0]),np.array([Cp_0]),np.array([Cpk_0]),np.array([kw_0])),axis=0)
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

# pass the ordering, the number of partitions in the different dimensions     

# seed defines the corresponding uncertainty case
# x00 = number of partitions in the first dimension(0,cA) for the odd prediction step -> alternating such that in the second step the second dimension(1) will be partitioned like this
# x01 = number of partitions in the second dimension(1,cB) for the odd prediction step -> alternating such that in the second step the first dimension(0) will be partitioned like this
# x02 = number of partitions in the first dimension(0,teta) for the even prediction step -> alternating such that in the second step the second dimension(1) will be partitioned like this
# x03 = number of partitions in the second dimension(1,tetaK) for the even prediction step -> alternating such that in the second step the first dimension(0) will be partitioned like this
# x10 = number of partitions in the first dimension(0,cA) for the odd prediction step -> alternating such that in the second step the second dimension(1) will be partitioned like this
# x11 = number of partitions in the second dimension(1,cB) for the odd prediction step -> alternating such that in the second step the first dimension(0) will be partitioned like this
# x12 = number of partitions in the first dimension(0,teta) for the even prediction step -> alternating such that in the second step the second dimension(1) will be partitioned like this
# x13 = number of partitions in the second dimension(1,tetaK) for the even prediction step -> alternating such that in the second step the first dimension(0) will be partitioned like this
# ord1 = define which dimension to be partitioned according to the ordering in the odd prediction step
# ord2 = define which dimension to be partioned according to the ordering in the even prediction step    

def closed_loop_comp(seed,x00,x01,x02,x03,x10,x11,x12,x13,ord1,ord2):
    
    N = 30 #chosen same as in paper "Determinstic safety guarantees for learning-based control of monotone systems"
    N_RCIS = 2
   
    x = SX.sym('x',nx,1)
    u = SX.sym('u',nu,1)
    p = SX.sym('p',nd,1)

    #stage cost
    Q1 = 1; Q2 = 1; Q3 = 1; Q4 = 1; R1 = 1; R2 = 1; R3 = np.diag(np.ones(nu))
    
    stage_cost = (model.x['cA']-cAs).T@Q1@(model.x['cA']-cAs)+(model.x['cB']-cBs).T@Q2@(model.x['cB']-cBs)+(model.x['teta']-tetas).T@Q3@(model.x['teta']-tetas)+(model.x['teta_K']-tetaKs).T@Q4@(model.x['teta_K']-tetaKs)+(model.u['u_F']-Fs).T@R1@(model.u['u_F']-Fs)+(model.u['u_Qk']-Qks).T@R2@(model.u['u_Qk']-Qks)+(model.u-u).T@R3@(model.u-u)
    stage_cost_fcn = Function('stage_cost',[model.x,model.u,u],[stage_cost])
    
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
                        res.append(i)
            loop(xs)
        else:
            res=[xs]
        return res

    def depth(l):
        if isinstance(l, list):
            return 1 + max(depth(item) for item in l) if l else 1
        else:
            return 0 
    
# in this approach we will cut in each timestep in alternating dimensions
# but in this approach we will have a different amount of subregions in each step since the number of cuts is not the same as in the previous partitioning approach
# therefore two lists and two cutting vectors are necessary
# each cutvector contains the number of cuts in the respective dimensions     
    
    cut1 = np.zeros((nx,1))
    cut1[0] = x00 # cuts in the first dimension  (0,cA)
    cut1[1] = x01 # cuts in the second dimension (1,cB)
    cut1[2] = x02 # cuts in the third dimension (2,teta)
    cut1[3] = x03 # cuts in the fourth dimension (3,teta_K)
    
    cut2 = np.zeros((nx,1))
    cut2[0] = x10 # cuts in the first dimension  (0,cA)
    cut2[1] = x11 # cuts in the second dimension (1,cB)
    cut2[2] = x12 # cuts in the third dimension (2,teta)
    cut2[3] = x13 # cuts in the fourth dimension (3,teta_K)
    
    ordering1 = ord1
    ordering2 = ord2 
    
    ordering1 = ord1
    ordering2 = ord2
    
# define number of subregions with the number of cuts in each dimension
# formula is valid if the total number of cuts for a specific dimension is the same in each subregion  
    n1 = 1 # these are the initial values and they will be updated both in the following according to the respective cuts 
    n2 = 1
    
    for i in range(nx):
        n1*=(cut1[i]+1)
    n1 = int(n1[0]) # number of subregions if the first dimension (0) is cut only (odd_case)
    
    for i in range(nx):
        n2*=(cut2[i]+1)
    n2 = int(n2[0]) # number of subregions if the second dimension (1) is cut only (even case)
    
    # define a recursive function that creates the corresponding lists
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
    lis2 = create_lis(cut2, ordering2, dim=0, lis=[], last_idx=0)[0]
    
# ns = n1 = n2 is not a valid assumption anymore    
# N + 1 = N1(odd prediction steps in total)+N2(even prediction steps in total)
# indexing starts from 0 and thus N+1 is the number of prediction steps as it includes the 0 index as the first prediction step

    N1 = 0 # odd-count -> cutting 0th dimension
    N2 = 0 # even-count -> cutting 1 dimension

    for i in range(0,N+1): 
        if i%2 == 0:
            N2 += 1
        elif i%2 == 1:
            N1 += 1
# define two optimization variables for the two different scenarios
# scenario 1 (cut with respect to lis1 in the first odd step)
# depending on which count is higher the corresponding u will have to be adapted   
    if N1 == N2:
        opt_x1 = struct_symSX([
            entry('x_min_odd', shape=nx, repeat=[N1,n1,K+1]),
            entry('x_max_odd', shape=nx, repeat=[N1,n1,K+1]),
            entry('u_odd', shape=nu, repeat=[N1-1,n1]),
            entry('x_min_even', shape=nx, repeat=[N2,n2,K+1]),
            entry('x_max_even', shape=nx, repeat=[N2,n2,K+1]),
            entry('u_even', shape=nu, repeat=[N2,n2]),
            ])
        
       
    elif N1 < N2:
        opt_x1 = struct_symSX([
            entry('x_min_odd', shape=nx, repeat=[N1,n1,K+1]),
            entry('x_max_odd', shape=nx, repeat=[N1,n1,K+1]),
            entry('u_odd', shape=nu, repeat=[N1,n1]),
            entry('x_min_even', shape=nx, repeat=[N2,n2,K+1]),
            entry('x_max_even', shape=nx, repeat=[N2,n2,K+1]),
            entry('u_even', shape=nu, repeat=[N2-1,n2]),
            ])

# scenario 2 (cut with respect to lis2 in the first odd step)
# depending on which count is higher the corresponding u will have to be adapted
    if N1 == N2:
        opt_x2 = struct_symSX([
            entry('x_min_odd', shape=nx, repeat=[N1,n2,K+1]),
            entry('x_max_odd', shape=nx, repeat=[N1,n2,K+1]),
            entry('u_odd', shape=nu, repeat=[N1-1,n2]),
            entry('x_min_even', shape=nx, repeat=[N2,n1,K+1]),
            entry('x_max_even', shape=nx, repeat=[N2,n1,K+1]),
            entry('u_even', shape=nu, repeat=[N2,n1]),
            ])
        
       
    elif N1 < N2:
        opt_x2 = struct_symSX([
            entry('x_min_odd', shape=nx, repeat=[N1,n2,K+1]),
            entry('x_max_odd', shape=nx, repeat=[N1,n2,K+1]),
            entry('u_odd', shape=nu, repeat=[N1,n2]),
            entry('x_min_even', shape=nx, repeat=[N2,n1,K+1]),
            entry('x_max_even', shape=nx, repeat=[N2,n1,K+1]),
            entry('u_even', shape=nu, repeat=[N2-1,n1]),
            ])    
        
# the corresponding 2-step RCIS must be in the right shape but no case distinction is necessary here           
    x_RCIS = struct_symSX([
       entry('x_min_odd', shape=nx ,repeat=[1,n1,K+1]),
       entry('x_max_odd', shape=nx ,repeat=[1,n1,K+1]),
       entry('u_odd', shape=nu, repeat=[1,n1]),
       entry('x_min_even', shape=nx ,repeat=[1,n2,K+1]),
       entry('x_max_even', shape=nx ,repeat=[1,n2,K+1]),
       entry('u_even', shape=nu, repeat=[1,n2])
       ])

# set the bounds on opt_x  and x_RCIS   
    lb_opt_x1 = opt_x1(0)
    ub_opt_x1 = opt_x1(np.inf)

    lb_opt_x1['x_min_odd'] = lb_x/scaling_x
    ub_opt_x1['x_min_odd'] = ub_x/scaling_x
    lb_opt_x1['x_max_odd'] = lb_x/scaling_x
    ub_opt_x1['x_max_odd'] = ub_x/scaling_x
    
    lb_opt_x1['x_min_even'] = lb_x/scaling_x
    ub_opt_x1['x_min_even'] = ub_x/scaling_x
    lb_opt_x1['x_max_even'] = lb_x/scaling_x
    ub_opt_x1['x_max_even'] = ub_x/scaling_x

    lb_opt_x1['u_odd'] = lb_u/scaling_u
    ub_opt_x1['u_odd'] = ub_u/scaling_u
    
    lb_opt_x1['u_even'] = lb_u/scaling_u
    ub_opt_x1['u_even'] = ub_u/scaling_u

###############################################################################    
    
    lb_opt_x2 = opt_x2(0)
    ub_opt_x2 = opt_x2(np.inf)

    lb_opt_x2['x_min_odd'] = lb_x/scaling_x
    ub_opt_x2['x_min_odd'] = ub_x/scaling_x
    lb_opt_x2['x_max_odd'] = lb_x/scaling_x
    ub_opt_x2['x_max_odd'] = ub_x/scaling_x
    
    lb_opt_x2['x_min_even'] = lb_x/scaling_x
    ub_opt_x2['x_min_even'] = ub_x/scaling_x
    lb_opt_x2['x_max_even'] = lb_x/scaling_x
    ub_opt_x2['x_max_even'] = ub_x/scaling_x

    lb_opt_x2['u_odd'] = lb_u/scaling_u
    ub_opt_x2['u_odd'] = ub_u/scaling_u
    
    lb_opt_x2['u_even'] = lb_u/scaling_u
    ub_opt_x2['u_even'] = ub_u/scaling_u

###############################################################################

    lb_opt_x_RCIS = x_RCIS(0)
    ub_opt_x_RCIS = x_RCIS(np.inf)

    lb_opt_x_RCIS['x_min_odd'] = lb_x/scaling_x
    lb_opt_x_RCIS['x_max_odd'] = lb_x/scaling_x
    ub_opt_x_RCIS['x_min_odd'] = ub_x/scaling_x
    ub_opt_x_RCIS['x_max_odd'] = ub_x/scaling_x
    
    lb_opt_x_RCIS['x_min_even'] = lb_x/scaling_x
    lb_opt_x_RCIS['x_max_even'] = lb_x/scaling_x
    ub_opt_x_RCIS['x_min_even'] = ub_x/scaling_x
    ub_opt_x_RCIS['x_max_even'] = ub_x/scaling_x

    lb_opt_x_RCIS['u_odd'] = lb_u/scaling_u
    ub_opt_x_RCIS['u_odd'] = ub_u/scaling_u
    
    lb_opt_x_RCIS['u_even'] = lb_u/scaling_u
    ub_opt_x_RCIS['u_even'] = ub_u/scaling_u
# define the constraint functions
# constraint_function1 is defined for partitioning the odd prediction steps 
# constraint_function2 is defined for partitioning the even prediction steps

    def constraint_function1(l,ord_dim,opt_x,i,h,lbg,ubg):
        for k in range(len(l)):
            idx=flatten(l[k])
            dim=ord_dim[-depth(l)]
            for s in idx:
                if s==idx[0] and k==0:
                    h.append(opt_x['x_min_odd',i,s,0,dim]-opt_x['x_min_odd',i,0,0,dim])
                    lbg.append(0)
                    ubg.append(0)
                else:
                    h.append(opt_x['x_min_odd',i,s,0,dim]-opt_x['x_min_odd',i,idx[0],0,dim])
                    lbg.append(0)
                    ubg.append(0)
                if s==idx[-1] and k==len(l)-1:
                    h.append(opt_x['x_max_odd',i,s,0,dim]-opt_x['x_max_odd',i,-1,0,dim])
                    lbg.append(0)
                    ubg.append(0)
                else:
                    h.append(opt_x['x_max_odd',i,s,0,dim]-opt_x['x_max_odd',i,idx[-1],0,dim])
                    lbg.append(0)
                    ubg.append(0)
            if k>=1:
                prev_last=flatten(l[k-1])[-1]
                h.append(opt_x['x_min_odd',i,idx[0],0,dim]-opt_x['x_max_odd',i,prev_last,0,dim])
                lbg.append(0)
                ubg.append(0)
            if depth(l) >1:
                h,lbg,ubg=constraint_function1(l[k],ord_dim,opt_x,i,h,lbg,ubg)
        
        return h,lbg,ubg
    
    def constraint_function2(l,ord_dim,opt_x,i,h,lbg,ubg):
        for k in range(len(l)):
            idx=flatten(l[k])
            dim=ord_dim[-depth(l)];
            for s in idx:
                if s==idx[0] and k==0:
                    h.append(opt_x['x_min_even',i,s,0,dim]-opt_x['x_min_even',i,0,0,dim])
                    lbg.append(0)
                    ubg.append(0)
                else:
                    h.append(opt_x['x_min_even',i,s,0,dim]-opt_x['x_min_even',i,idx[0],0,dim])
                    lbg.append(0)
                    ubg.append(0)
                if s==idx[-1] and k==len(l)-1:
                    h.append(opt_x['x_max_even',i,s,0,dim]-opt_x['x_max_even',i,-1,0,dim])
                    lbg.append(0)
                    ubg.append(0)
                else:
                    h.append(opt_x['x_max_even',i,s,0,dim]-opt_x['x_max_even',i,idx[-1],0,dim])
                    lbg.append(0)
                    ubg.append(0)
            if k>=1:
                prev_last=flatten(l[k-1])[-1]
                h.append(opt_x['x_min_even',i,idx[0],0,dim]-opt_x['x_max_even',i,prev_last,0,dim])
                lbg.append(0)
                ubg.append(0)
            if depth(l) >1:
                h,lbg,ubg=constraint_function2(l[k],ord_dim,opt_x,i,h,lbg,ubg)
        
        return h,lbg,ubg

# Set up the objective and the constraints of the problem
# in contrast to the previous partitioning approach, here we have a different number of subregions in each prediction step, as the number of cuts per dimension also varies in each timestep
# therefore the cost function has to be adjusted according to if the partitioning occures with respect to lis1 or lis2 in the first odd step (where the cutting starts)
# of course the initial constraints, the systems equation (inequality constraints), the partitioning constraints and the terminal cost will be adjusted accordingly too
# in the end we create two solvers for the two scenarios where the set is partitioned with respect to lis1 or in the other scenario with respect to lis2 in the first odd prediction step and pass J1,g1 and J2,g2 so that we solve two different optimization problems in each simulation timestep
# by doing this a way to partition the predicted states in an alternating fashion is found and in addition also a different number of subregions in each prediction step can be set to circumvent the exponential growth of the number of subregions by partitioning the full set and on the same time achieve good prediction results 

# cost and constraints for scenario 1 (cut with regards to lis1 in the first odd prediction step)
    J1 = 0 # cost fct for normal prediction horizon N
    g1 = [] #constraint expression
    lb_g1 = [] #lower bound for constraint expression g1
    ub_g1 = [] #upper bound for constraint expression g1

# cost and constraints for scenario 2 (cut with regards to lis2 in the first odd step)
    J2 = 0 # cost fct for normal prediction horizon N
    g2 = [] #constraint expression
    lb_g2 = [] #lower bound for constraint expression g2
    ub_g2 = [] #upper bound for constraint expression g2
    
    J_RCIS = 0
    g_RCIS = []
    lb_g_RCIS = []
    ub_g_RCIS = []
    
    x_init = SX.sym('x_init', nx,1)
    u_bef=SX.sym('u_bef',nu,1)
    p_plus=SX.sym('p_plus', nd,1)
    p_minus=SX.sym('p_minus', nd,1)
    x_RCIS_plus = SX.sym('x_RCIS_plus', nx, 1)
    x_RCIS_minus = SX.sym('x_RCIS_minus', nx, 1)
    
######################### Scenario 1 (cut with regards to lis1 in the first odd prediction step)   
# Set initial constraints
# start with even case because indexing starts at 0    
    for s in range(n2):
        g1.append(opt_x1['x_min_even',0,s,0]-x_init)
        g1.append(opt_x1['x_max_even',0,s,0]-x_init)
        lb_g1.append(np.zeros((2*nx,1)))
        ub_g1.append(np.zeros((2*nx,1)))
        if s>0:
            g1.append(opt_x1['u_even',0,s]-opt_x1['u_even',0,0])
            lb_g1.append(np.zeros((nu,1)))
            ub_g1.append(np.zeros((nu,1))) 

# objective and inequality constraints must be adapted in each step since the alignment changes then and consequently also the resulting number of subregions
# start looping from 0 to N and distinguish between even and odd prediction steps in each step
    even_count = 0
    odd_count = 0 
    for i in range(N):
        # objective
        if i%2 == 0: #even-case -> cut with regards to lis2
            for s in range(n2):
                if i==0:
                    J1 += stage_cost_fcn(opt_x1['x_max_even', even_count,s,0]*scaling_x, opt_x1['u_even', even_count,s]*scaling_u,u_bef)
                    J1 += stage_cost_fcn(opt_x1['x_min_even', even_count,s,0]*scaling_x, opt_x1['u_even', even_count,s]*scaling_u,u_bef)
                else:  
                    # u_odd_vecs = [u * scaling_u for u in opt_x1['u_odd', odd_count-1, :]]
                    # u_odd_mean = sum1(vertcat(*u_odd_vecs)) / len(u_odd_vecs)
                    
                    u_odd_vecs = [u * scaling_u for u in opt_x1['u_odd', odd_count-1, :]]
                    u_odd_mean = sum2(horzcat(*u_odd_vecs)) / len(u_odd_vecs)
                    J1 += stage_cost_fcn(opt_x1['x_max_even', even_count,s,0]*scaling_x, opt_x1['u_even', even_count,s]*scaling_u,u_odd_mean) #refers to the previous odd step, which is the previous prediction step
                    J1 += stage_cost_fcn(opt_x1['x_min_even', even_count,s,0]*scaling_x, opt_x1['u_even', even_count,s]*scaling_u,u_odd_mean) #refers to the previous odd step, which is the previous prediction step
                
        # equality + inequality constraints (system equation)
                for k in range(1,K+1):
                    # x_next_max = -dt*system(opt_x1['x_max_even',even_count,s,k]*scaling_x,opt_x1['u_even',even_count,s]*scaling_u,p_max)/scaling_x
                    # x_next_min = -dt*system(opt_x1['x_min_even',even_count,s,k]*scaling_x,opt_x1['u_even',even_count,s]*scaling_u,p_min)/scaling_x
                    x_next_max = -dt*dmax(opt_x1['x_min_even',even_count,s,k]*scaling_x,opt_x1['x_max_even',even_count,s,k]*scaling_x, opt_x1['u_even',even_count,s]*scaling_u, p_minus, p_plus)/scaling_x
                    x_next_min = -dt*dmin(opt_x1['x_min_even',even_count,s,k]*scaling_x,opt_x1['x_max_even',even_count,s,k]*scaling_x, opt_x1['u_even',even_count,s]*scaling_u, p_minus, p_plus)/scaling_x
                    
                    for j in range(K+1):
                        x_next_max += A[j,k]*opt_x1['x_max_even',even_count,s,j]
                        x_next_min += A[j,k]*opt_x1['x_min_even',even_count,s,j]
                    g1.append(x_next_max)
                    g1.append(x_next_min)
                    lb_g1.append(np.zeros((2*nx,1)))
                    ub_g1.append(np.zeros((2*nx,1)))
                    
                x_next_plus = horzcat(*opt_x1['x_max_even',even_count,s,:])@D
                x_next_minus = horzcat(*opt_x1['x_min_even',even_count,s,:])@D    
                g1.append(opt_x1['x_max_odd', odd_count,-1,0]-x_next_plus)
                g1.append(x_next_minus - opt_x1['x_min_odd', odd_count,0,0])
                lb_g1.append(np.zeros((2*nx,1)))
                ub_g1.append(np.ones((2*nx,1))*inf)

            
            even_count += 1
        elif i%2 == 1: #odd-case -> cut with regards to lis1 
            for s in range(n1):
                if odd_count == 0:
                    J1 += stage_cost_fcn(opt_x1['x_max_odd', odd_count,s,0]*scaling_x, opt_x1['u_odd', odd_count,s]*scaling_u,u_bef) 
                    J1 += stage_cost_fcn(opt_x1['x_min_odd', odd_count,s,0]*scaling_x, opt_x1['u_odd', odd_count,s]*scaling_u,u_bef) 
                else:
                    # u_even_vecs = [u * scaling_u for u in opt_x1['u_even', even_count-1, :]]
                    # u_even_mean = sum1(vertcat(*u_even_vecs)) / len(u_even_vecs)
                    u_even_vecs = [u * scaling_u for u in opt_x1['u_even', even_count-1, :]]
                    u_even_mean = sum2(horzcat(*u_even_vecs)) / len(u_even_vecs)
                    
                    J1 += stage_cost_fcn(opt_x1['x_max_odd', odd_count,s,0]*scaling_x, opt_x1['u_odd', odd_count,s]*scaling_u,u_even_mean) #refers to the previous even step, which is the previous prediction step
                    J1 += stage_cost_fcn(opt_x1['x_min_odd', odd_count,s,0]*scaling_x, opt_x1['u_odd', odd_count,s]*scaling_u,u_even_mean) #refers to the previous even step, which is the previous prediction step
        
        # equality + inequality constraints (system equation)
                for k in range(1,K+1):
                    # x_next_max = -dt*system(opt_x1['x_max_odd',odd_count,s,k]*scaling_x,opt_x1['u_odd',odd_count,s]*scaling_u,p_max)/scaling_x
                    # x_next_min = -dt*system(opt_x1['x_min_odd',odd_count,s,k]*scaling_x,opt_x1['u_odd',odd_count,s]*scaling_u,p_min)/scaling_x
                    x_next_max = -dt*dmax(opt_x1['x_min_odd',odd_count,s,k]*scaling_x,opt_x1['x_max_odd',odd_count,s,k]*scaling_x, opt_x1['u_odd',odd_count,s]*scaling_u, p_minus, p_plus)/scaling_x
                    x_next_min = -dt*dmin(opt_x1['x_min_odd',odd_count,s,k]*scaling_x,opt_x1['x_max_odd',odd_count,s,k]*scaling_x, opt_x1['u_odd',odd_count,s]*scaling_u, p_minus, p_plus)/scaling_x
                
                    for j in range(K+1):
                        x_next_max += A[j,k]*opt_x1['x_max_odd',odd_count,s,j]
                        x_next_min += A[j,k]*opt_x1['x_min_odd',odd_count,s,j]
                    g1.append(x_next_max)
                    g1.append(x_next_min)
                    lb_g1.append(np.zeros((2*nx,1)))
                    ub_g1.append(np.zeros((2*nx,1)))
                
                x_next_plus = horzcat(*opt_x1['x_max_odd',odd_count,s,:])@D
                x_next_minus = horzcat(*opt_x1['x_min_odd',odd_count,s,:])@D  
                g1.append( opt_x1['x_max_even', even_count,-1,0]-x_next_plus)
                g1.append(x_next_minus - opt_x1['x_min_even', even_count,0,0])
                lb_g1.append(np.zeros((2*nx,1)))
                ub_g1.append(np.ones((2*nx,1))*inf)

            odd_count += 1

# terminal cost                
# case distinction depending on if the last prediction step is even or odd
    
    if N1 == N2:
        for s in range(n1):
            J1 += terminal_cost_fcn(opt_x1['x_max_odd',-1, s,0]*scaling_x)
            J1 += terminal_cost_fcn(opt_x1['x_min_odd',-1, s,0]*scaling_x)
    elif N1 < N2:
        for s in range(n2):
            J1 += terminal_cost_fcn(opt_x1['x_max_even',-1, s,0]*scaling_x)
            J1 += terminal_cost_fcn(opt_x1['x_min_even',-1, s,0]*scaling_x)

# cutting1 (cut the predicted set of states in the first odd prediction step(1) with regards to lis1) 
# now we partition the predicted state sets with respect to lis1 and lis2 in consecutive prediction steps such that the set partitioning occures in an alternating fashion
# therefore we pass a different list "ordering dimension" at each prediction (time)-step        
    
    even_count1 = 1 # even_count1 = 0 would mean that first set (rather element since it is a distinct initial point) of the prediction horizon (x_max_even,0) would be partitioned/cut too.
    odd_count1 = 0 # Therefore we start indexing the even steps at 1 and after the loop we can subtract the 1 in order to get the correct number of even cutting steps 
    for i in range(1,N+1): # at i = 0 there should be no cutting since we have a distinct initial point
        # if i is even, cut w.r.t. lis2 and ordering2
        # if i is odd, cut w.r.t. lis1 and ordering1
        if i % 2 == 0:
            g1, lb_g1, ub_g1 = constraint_function2(lis2,ordering2,opt_x1,even_count1,g1,lb_g1,ub_g1)
        
            for s in range(n2):
                g1.append(opt_x1['x_max_even',even_count1,s,0]-opt_x1['x_min_even',even_count1,0,0])
                g1.append(opt_x1['x_max_even',even_count1,-1,0]-opt_x1['x_min_even',even_count1,s,0])
                g1.append(opt_x1['x_min_even',even_count1,s,0]-opt_x1['x_min_even',even_count1,0,0])
                g1.append(opt_x1['x_max_even',even_count1,-1,0]-opt_x1['x_max_even',even_count1,s,0])
                lb_g1.append(np.zeros((4*nx,1)))
                ub_g1.append(np.ones((4*nx,1))*inf)
            
            even_count1 += 1
        
        elif i % 2 == 1:
            g1, lb_g1, ub_g1 = constraint_function1(lis1,ordering1,opt_x1,odd_count1,g1,lb_g1,ub_g1)
            
        
            for s in range(n1):
                g1.append(opt_x1['x_max_odd',odd_count1,s,0]-opt_x1['x_min_odd',odd_count1,0,0])
                g1.append(opt_x1['x_max_odd',odd_count1,-1,0]-opt_x1['x_min_odd',odd_count1,s,0])
                g1.append(opt_x1['x_min_odd',odd_count1,s,0]-opt_x1['x_min_odd',odd_count1,0,0])
                g1.append(opt_x1['x_max_odd',odd_count1,-1,0]-opt_x1['x_max_odd',odd_count1,s,0])
                lb_g1.append(np.zeros((4*nx,1)))
                ub_g1.append(np.ones((4*nx,1))*inf)
            
            odd_count1 += 1    
    
    even_count1 -= 1
######################### Scenario 2 (cut with regards to lis2 in the first odd prediction step)

# Set initial constraints
# start with even case because indexing starts at 0    
    for s in range(n1):
        g2.append(opt_x2['x_min_even',0,s,0]-x_init)
        g2.append(opt_x2['x_max_even',0,s,0]-x_init)
        lb_g2.append(np.zeros((2*nx,1)))
        ub_g2.append(np.zeros((2*nx,1)))
        if s>0:
            g2.append(opt_x2['u_even',0,s]-opt_x2['u_even',0,0])
            lb_g2.append(np.zeros((nu,1)))
            ub_g2.append(np.zeros((nu,1))) 

# objective and inequality constraints must be adapted in each step since the alignment changes then and consequently also the resulting number of subregions
# start looping from 0 to N and distinguish between even and odd prediction steps in each step
    even_count = 0
    odd_count = 0 
    for i in range(N):
        # objective
        if i%2 == 0: #even-case -> cut with regards to lis1
            for s in range(n1):
                if i==0:
                    J2 += stage_cost_fcn(opt_x2['x_max_even', even_count,s,0]*scaling_x, opt_x2['u_even', even_count,s]*scaling_u,u_bef)
                    J2 += stage_cost_fcn(opt_x2['x_min_even', even_count,s,0]*scaling_x, opt_x2['u_even', even_count,s]*scaling_u,u_bef)
                else:
                    # u_odd_vecs = [u * scaling_u for u in opt_x2['u_odd', odd_count-1, :]]
                    # u_odd_mean = sum1(vertcat(*u_odd_vecs)) / len(u_odd_vecs)
                    
                    u_odd_vecs = [u * scaling_u for u in opt_x2['u_odd', odd_count-1, :]]
                    u_odd_mean = sum2(horzcat(*u_odd_vecs)) / len(u_odd_vecs)
                    J2 += stage_cost_fcn(opt_x2['x_max_even', even_count,s,0]*scaling_x, opt_x2['u_even', even_count,s]*scaling_u,u_odd_mean) #refers to the previous odd step, which is the previous prediction step 
                    J2 += stage_cost_fcn(opt_x2['x_min_even', even_count,s,0]*scaling_x, opt_x2['u_even', even_count,s]*scaling_u,u_odd_mean) #refers to the previous odd step, which is the previous prediction step 
                
        # inequality constraints (system equation)
                for k in range(1,K+1):
                    # x_next_max = -dt*system(opt_x2['x_max_even',even_count,s,k]*scaling_x,opt_x2['u_even',even_count,s]*scaling_u,p_max)/scaling_x
                    # x_next_min = -dt*system(opt_x2['x_min_even',even_count,s,k]*scaling_x,opt_x2['u_even',even_count,s]*scaling_u,p_min)/scaling_x
                    x_next_max = -dt*dmax(opt_x2['x_min_even',even_count,s,k]*scaling_x,opt_x2['x_max_even',even_count,s,k]*scaling_x, opt_x2['u_even',even_count,s]*scaling_u, p_minus, p_plus)/scaling_x
                    x_next_min = -dt*dmin(opt_x2['x_min_even',even_count,s,k]*scaling_x,opt_x2['x_max_even',even_count,s,k]*scaling_x, opt_x2['u_even',even_count,s]*scaling_u, p_minus, p_plus)/scaling_x
                    
                    for j in range(K+1):
                        x_next_max += A[j,k]*opt_x2['x_max_even',even_count,s,j]
                        x_next_min += A[j,k]*opt_x2['x_min_even',even_count,s,j]
                    g2.append(x_next_max)
                    g2.append(x_next_min)
                    lb_g2.append(np.zeros((2*nx,1)))
                    ub_g2.append(np.zeros((2*nx,1)))
                    
                x_next_plus = horzcat(*opt_x2['x_max_even',even_count,s,:])@D
                x_next_minus = horzcat(*opt_x2['x_min_even',even_count,s,:])@D    
                g2.append(opt_x2['x_max_odd', odd_count,-1,0]-x_next_plus)
                g2.append(x_next_minus - opt_x2['x_min_odd', odd_count,0,0])
                lb_g2.append(np.zeros((2*nx,1)))
                ub_g2.append(np.ones((2*nx,1))*inf)

            
            even_count += 1
        elif i%2 == 1: #odd-case -> cut with regards to lis2
            for s in range(n2):
                if odd_count == 0:
                    J2 += stage_cost_fcn(opt_x2['x_max_odd', odd_count,s,0]*scaling_x, opt_x2['u_odd', odd_count,s]*scaling_u,u_bef)  
                    J2 += stage_cost_fcn(opt_x2['x_min_odd', odd_count,s,0]*scaling_x, opt_x2['u_odd', odd_count,s]*scaling_u,u_bef)
                else:
                    # u_even_vecs = [u * scaling_u for u in opt_x2['u_even', even_count-1, :]]
                    # u_even_mean = sum1(vertcat(*u_even_vecs)) / len(u_even_vecs)
                    
                    u_even_vecs = [u * scaling_u for u in opt_x2['u_even', even_count-1, :]]
                    u_even_mean = sum2(horzcat(*u_even_vecs)) / len(u_even_vecs)
                    J2 += stage_cost_fcn(opt_x2['x_max_odd', odd_count,s,0]*scaling_x, opt_x2['u_odd', odd_count,s]*scaling_u,u_even_mean) #refers to the previous even step, which is the previous prediction step 
                    J2 += stage_cost_fcn(opt_x2['x_min_odd', odd_count,s,0]*scaling_x, opt_x2['u_odd', odd_count,s]*scaling_u,u_even_mean) #refers to the previous even step, which is the previous prediction step 
        
        # inequality constraints (system equation)
          
                for k in range(1,K+1):
                    # x_next_max = -dt*system(opt_x2['x_max_odd',odd_count,s,k]*scaling_x,opt_x2['u_odd',odd_count,s]*scaling_u,p_max)/scaling_x
                    # x_next_min = -dt*system(opt_x2['x_min_odd',odd_count,s,k]*scaling_x,opt_x2['u_odd',odd_count,s]*scaling_u,p_min)/scaling_x
                    x_next_max = -dt*dmax(opt_x2['x_min_odd',odd_count,s,k]*scaling_x,opt_x2['x_max_odd',odd_count,s,k]*scaling_x, opt_x2['u_odd',odd_count,s]*scaling_u, p_minus, p_plus)/scaling_x
                    x_next_min = -dt*dmin(opt_x2['x_min_odd',odd_count,s,k]*scaling_x,opt_x2['x_max_odd',odd_count,s,k]*scaling_x, opt_x2['u_odd',odd_count,s]*scaling_u, p_minus, p_plus)/scaling_x
                
                    for j in range(K+1):
                        x_next_max += A[j,k]*opt_x2['x_max_odd',odd_count,s,j]
                        x_next_min += A[j,k]*opt_x2['x_min_odd',odd_count,s,j]
                    g2.append(x_next_max)
                    g2.append(x_next_min)
                    lb_g2.append(np.zeros((2*nx,1)))
                    ub_g2.append(np.zeros((2*nx,1)))
                
                x_next_plus = horzcat(*opt_x2['x_max_odd',odd_count,s,:])@D
                x_next_minus = horzcat(*opt_x2['x_min_odd',odd_count,s,:])@D  
                g2.append( opt_x2['x_max_even', even_count,-1,0]-x_next_plus)
                g2.append(x_next_minus - opt_x2['x_min_even', even_count,0,0])
                lb_g2.append(np.zeros((2*nx,1)))
                ub_g2.append(np.ones((2*nx,1))*inf)

            odd_count += 1
# terminal cost                
# case distinction depending on if the last prediction step is even or odd
    if N1 == N2:
        for s in range(n2):
            J2 += terminal_cost_fcn(opt_x2['x_max_odd',-1, s,0]*scaling_x)
            J2 += terminal_cost_fcn(opt_x2['x_min_odd',-1, s,0]*scaling_x)
    elif N1 < N2:
        for s in range(n1):
            J2 += terminal_cost_fcn(opt_x2['x_max_even',-1, s,0]*scaling_x)
            J2 += terminal_cost_fcn(opt_x2['x_min_even',-1, s,0]*scaling_x)

# cutting2 (cut the predicted set of states in the first odd prediction step(1) with regards to lis2) 
# now we partition the predicted state sets with respect to lis1 and lis2 in consecutive prediction steps such that the set partitioning occures in an alternating fashion
# therefore we pass a different list "ordering dimension" at each prediction (time)-step
    even_count2 = 1  # even_count1 = 0 would mean that first set (rather element since it is a distinct initial point) of the prediction horizon (x_max_even,0) would be partitioned/cut too.
    odd_count2 = 0 # Therefore we start indexing the even steps at 1 and after the loop we can subtract the 1 in order to get the correct number of even cutting steps

    for i in range(1,N+1):
        # if i is even, cut w.r.t. lis1 and ordering1
        # if i is odd, cut w.r.t. lis2 and ordering2
        
        if i % 2 == 0:
            g2, lb_g2, ub_g2 = constraint_function2(lis1,ordering1,opt_x2,even_count2,g2,lb_g2,ub_g2) 
            
            for s in range(n1):
                g2.append(opt_x2['x_max_even',even_count2,s,0]-opt_x2['x_min_even',even_count2,0,0])
                g2.append(opt_x2['x_max_even',even_count2,-1,0]-opt_x2['x_min_even',even_count2,s,0])
                g2.append(opt_x2['x_min_even',even_count2,s,0]-opt_x2['x_min_even',even_count2,0,0])
                g2.append(opt_x2['x_max_even',even_count2,-1,0]-opt_x2['x_max_even',even_count2,s,0])
                lb_g2.append(np.zeros((4*nx,1)))
                ub_g2.append(np.ones((4*nx,1))*inf)
            
            even_count2 += 1
        
        elif i % 2 == 1:
            g2, lb_g2, ub_g2 = constraint_function1(lis2,ordering2,opt_x2,odd_count2,g2,lb_g2,ub_g2)  
            
            for s in range(n2):
                g2.append(opt_x2['x_max_odd',odd_count2,s,0]-opt_x2['x_min_odd',odd_count2,0,0])
                g2.append(opt_x2['x_max_odd',odd_count2,-1,0]-opt_x2['x_min_odd',odd_count2,s,0])
                g2.append(opt_x2['x_min_odd',odd_count2,s,0]-opt_x2['x_min_odd',odd_count2,0,0])
                g2.append(opt_x2['x_max_odd',odd_count2,-1,0]-opt_x2['x_max_odd',odd_count2,s,0])
                lb_g2.append(np.zeros((4*nx,1)))
                ub_g2.append(np.ones((4*nx,1))*inf)
            
            odd_count2 += 1
    
    even_count2 -= 1                
###############################################################################
# computation of 2-step RCIS
    x_rcis = time.time()
    N_RCIS = 2
    even_count_RCIS = 0
    odd_count_RCIS = 0
    for i in range(N_RCIS):
        if i%2 == 0: #-> even case (cut w.r.t. lis1)
            for s in range(n2):
                for k in range(1,K+1):
                    # x_next_plus_RCIS = -dt*system(x_RCIS['x_max_even',even_count_RCIS,s,k]*scaling_x, x_RCIS['u_even',even_count_RCIS,s]*scaling_u,p_max)/scaling_x
                    # x_next_minus_RCIS = -dt*system(x_RCIS['x_min_even',even_count_RCIS,s,k]*scaling_x, x_RCIS['u_even',even_count_RCIS,s]*scaling_u,p_min)/scaling_x
                    x_next_plus_RCIS = -dt*dmax(x_RCIS['x_min_even',even_count_RCIS,s,k]*scaling_x,x_RCIS['x_max_even',even_count_RCIS,s,k]*scaling_x, x_RCIS['u_even',even_count_RCIS,s]*scaling_u, p_minus, p_plus)/scaling_x
                    x_next_minus_RCIS = -dt*dmin(x_RCIS['x_min_even',even_count_RCIS,s,k]*scaling_x,x_RCIS['x_max_even',even_count_RCIS,s,k]*scaling_x, x_RCIS['u_even',even_count_RCIS,s]*scaling_u, p_minus, p_plus)/scaling_x
                    for j in range(K+1):
                        x_next_plus_RCIS += A[j,k]*x_RCIS['x_max_even',even_count_RCIS,s,j]
                        x_next_minus_RCIS += A[j,k]*x_RCIS['x_min_even',even_count_RCIS,s,j]
                    
                    g_RCIS.append(x_next_plus_RCIS)
                    g_RCIS.append(x_next_minus_RCIS)
                    lb_g_RCIS.append(np.zeros((2*nx,1)))
                    ub_g_RCIS.append(np.zeros((2*nx,1)))
                
                x_next_plus_RCIS = horzcat(*x_RCIS['x_max_even',even_count_RCIS,s,:])@D
                x_next_minus_RCIS = horzcat(*x_RCIS['x_min_even',even_count_RCIS,s,:])@D   
                
                if i == N_RCIS-1: #constrain the propagation of the last hyperrectangle to lie again in the first hyperrectangle
                    g_RCIS.append(x_RCIS['x_max_even',0,-1,0] - x_next_plus_RCIS)
                    g_RCIS.append(x_next_minus_RCIS - x_RCIS['x_min_even',0, 0,0])
                else:
                    g_RCIS.append(x_RCIS['x_max_odd',odd_count_RCIS,-1,0] - x_next_plus_RCIS)
                    g_RCIS.append(x_next_minus_RCIS - x_RCIS['x_min_odd',odd_count_RCIS, 0,0])

                lb_g_RCIS.append(np.zeros((2*nx,1)))
                ub_g_RCIS.append(inf*np.ones((2*nx,1)))
            
            even_count_RCIS += 1
        
        if i%2 == 1: #-> odd case (cut w.r.t. lis2)
            for s in range(n1):
                for k in range(1,K+1):
                    # x_next_plus_RCIS = -dt*system(x_RCIS['x_max_odd',odd_count_RCIS,s,k]*scaling_x, x_RCIS['u_odd',odd_count_RCIS,s]*scaling_u,p_max)/scaling_x
                    # x_next_minus_RCIS = -dt*system(x_RCIS['x_min_odd',odd_count_RCIS,s,k]*scaling_x, x_RCIS['u_odd',odd_count_RCIS,s]*scaling_u,p_min)/scaling_x
                    x_next_plus_RCIS = -dt*dmax(x_RCIS['x_min_odd',odd_count_RCIS,s,k]*scaling_x,x_RCIS['x_max_odd',odd_count_RCIS,s,k]*scaling_x, x_RCIS['u_odd',odd_count_RCIS,s]*scaling_u, p_minus, p_plus)/scaling_x
                    x_next_minus_RCIS = -dt*dmin(x_RCIS['x_min_odd',odd_count_RCIS,s,k]*scaling_x,x_RCIS['x_max_odd',odd_count_RCIS,s,k]*scaling_x, x_RCIS['u_odd',odd_count_RCIS,s]*scaling_u, p_minus, p_plus)/scaling_x
                    for j in range(K+1):
                        x_next_plus_RCIS += A[j,k]*x_RCIS['x_max_odd',odd_count_RCIS,s,j]
                        x_next_minus_RCIS += A[j,k]*x_RCIS['x_min_odd',odd_count_RCIS,s,j]
                    
                    g_RCIS.append(x_next_plus_RCIS)
                    g_RCIS.append(x_next_minus_RCIS)
                    lb_g_RCIS.append(np.zeros((2*nx,1)))
                    ub_g_RCIS.append(np.zeros((2*nx,1)))
                    
                x_next_plus_RCIS = horzcat(*x_RCIS['x_max_even',odd_count_RCIS,s,:])@D
                x_next_minus_RCIS = horzcat(*x_RCIS['x_min_even',odd_count_RCIS,s,:])@D 
                
                if i == N_RCIS-1: #constrain the propagation of the last hyperrectangle to lie again in the first hyperrectangle
                    g_RCIS.append(x_RCIS['x_max_even',0,-1,0] - x_next_plus_RCIS)
                    g_RCIS.append(x_next_minus_RCIS - x_RCIS['x_min_even',0, 0,0])
                else:
                    g_RCIS.append(x_RCIS['x_max_even',even_count_RCIS ,-1,0] - x_next_plus_RCIS)
                    g_RCIS.append(x_next_minus_RCIS - x_RCIS['x_min_even',even_count_RCIS , 0,0])
                
                lb_g_RCIS.append(np.zeros((2*nx,1)))
                ub_g_RCIS.append(inf*np.ones((2*nx,1)))
            
            odd_count_RCIS += 1
                                
        # Cutting RCIS
        # cut in an alternating fashion with different lists depending on whether the prediction step is even or odd
        
        if i % 2 == 0: # cut w.r.t. lis2
            g_RCIS,lb_g_RCIS,ub_g_RCIS = constraint_function2(lis2,ordering2,x_RCIS,even_count_RCIS-1,g_RCIS,lb_g_RCIS,ub_g_RCIS);
            for s in range(n2):
                g_RCIS.append(x_RCIS['x_max_even',even_count_RCIS-1,s,0]-x_RCIS['x_min_even',even_count_RCIS-1,0,0])
                g_RCIS.append(x_RCIS['x_max_even',even_count_RCIS-1,-1,0]-x_RCIS['x_min_even',even_count_RCIS-1,s,0])
                g_RCIS.append(x_RCIS['x_min_even',even_count_RCIS-1,s,0]-x_RCIS['x_min_even',even_count_RCIS-1,0,0])
                g_RCIS.append(x_RCIS['x_max_even',even_count_RCIS-1,-1,0]-x_RCIS['x_max_even',even_count_RCIS-1,s,0])
                lb_g_RCIS.append(np.zeros((4*nx,1)))
                ub_g_RCIS.append(np.ones((4*nx,1))*inf)
        elif i % 2 == 1: # cut w.r.t lis1
            g_RCIS,lb_g_RCIS,ub_g_RCIS = constraint_function1(lis1,ordering1,x_RCIS,odd_count_RCIS-1,g_RCIS,lb_g_RCIS,ub_g_RCIS);
            for s in range(n1):
                g_RCIS.append(x_RCIS['x_max_odd',odd_count_RCIS-1,s,0]-x_RCIS['x_min_odd',odd_count_RCIS-1,0,0])
                g_RCIS.append(x_RCIS['x_max_odd',odd_count_RCIS-1,-1,0]-x_RCIS['x_min_odd',odd_count_RCIS-1,s,0])
                g_RCIS.append(x_RCIS['x_min_odd',odd_count_RCIS-1,s,0]-x_RCIS['x_min_odd',odd_count_RCIS-1,0,0])
                g_RCIS.append(x_RCIS['x_max_odd',odd_count_RCIS-1,-1,0]-x_RCIS['x_max_odd',odd_count_RCIS-1,s,0])
                lb_g_RCIS.append(np.zeros((4*nx,1)))
                ub_g_RCIS.append(np.ones((4*nx,1))*inf)
            
    J_RCIS = -1
    even = 0
    odd = 0
    for i in range(N_RCIS):
        J_mini = -1
        if i % 2 == 0:
            for ix in range(nx):
                J_mini = J_mini*(x_RCIS['x_max_even',even,-1,0,ix]-x_RCIS['x_min_even',even,0,0,ix])
            even += 1
        elif i % 2 == 1:
            for ix in range(nx):
                J_mini = J_mini*(x_RCIS['x_max_odd',odd,-1,0,ix]-x_RCIS['x_min_odd',odd,0,0,ix])
            odd += 1
        
        J_RCIS += J_mini
        
    g_RCIS = vertcat(*g_RCIS)
    lb_g_RCIS = vertcat(*lb_g_RCIS)
    ub_g_RCIS = vertcat(*ub_g_RCIS)
    x_rcis = time.time()
    prob = {'f':J_RCIS,'x':vertcat(x_RCIS),'g':g_RCIS, 'p':vertcat(p_plus,p_minus)}
    solver_mx_inv_set = nlpsol('solver','ipopt',prob)
   
    # now solve the optimization problem 

    x_set = np.array([[0,0,100,98]]).T
    x_set = x_set/scaling_x 
    opt_ro_initial = x_RCIS(0)
    opt_ro_initial['x_min_even'] = x_set
    opt_ro_initial['x_max_even'] = x_set
    opt_ro_initial['x_min_odd'] = x_set
    opt_ro_initial['x_max_odd'] = x_set
    p_min = vertcat(p_fun_wcmin(0))
    p_max = vertcat(p_fun_wcmax(0))
    results = solver_mx_inv_set(p=vertcat(p_max,p_min),x0=opt_ro_initial, lbg=lb_g_RCIS,ubg=ub_g_RCIS,lbx=lb_opt_x_RCIS,ubx=ub_opt_x_RCIS);
    y_rcis = time.time()
    print("time to compute the 2-step RCIS:" , y_rcis-x_rcis)
    # transform the optimizer results into a structured symbolic variable
    res = x_RCIS(results['x']);

    # # plot the simulation results (RCIS for each timestep and the total RCIS)
    # even = 0
    # odd = 0
    
    
    # # as a performance metric the total size of the RCIS is computed in the following
    # # total area (RCIS) = areas of each step - intersection area (for 2 steps: total area = area1 + area2 - intersection of area1 and area2 )
    # width_1 = float(res['x_max_even',0,-1,0,]-res['x_min_even',0,0,0])
    # height_1 = float(res['x_max_even',0,-1,1]-res['x_min_even',0,0,1])
    # width_2 = float(res['x_max_odd',0,-1,0]-res['x_min_odd',0,0,0])
    # height_2 = float(res['x_max_odd',0,-1,1]-res['x_min_odd',0,0,1])
    # overlap_width = max(0,min(float(res['x_min_even',0,0,0]) + width_1, float(res['x_min_odd',0,0,0]) + width_2) - max(float(res['x_min_even',0,0,0]), float(res['x_min_odd',0,0,0])))
    # overlap_height = max(0,min(float(res['x_min_even',0,0,1]) + height_1, float(res['x_min_odd',0,0,1]) + height_2) - max(float(res['x_min_even',0,0,1]), float(res['x_min_odd',0,0,1])))
    # overlap_area = overlap_width*overlap_height

    # RCIS_step1 = width_1*height_1
    # RCIS_step2 = width_2*height_2
    # RCIS_total = width_1*height_1 + width_2*height_2 - overlap_area


# Constraining1 for RCIS
# depending on which dimension was cut lastly we should adjust the corresponding set of the RCIS so that x(N)ExRCIS holds

    if odd_count1 > even_count1:
        g1.append(x_RCIS_plus - opt_x1['x_max_odd', -1, -1,0])
        g1.append(opt_x1['x_min_odd', -1, 0,0] - x_RCIS_minus)
        lb_g1.append(np.zeros((2 * nx, 1)))
        ub_g1.append(inf * np.ones((2 * nx, 1)))
    elif odd_count1 == even_count1:
        g1.append(x_RCIS_plus - opt_x1['x_max_even', -1, -1,0])
        g1.append(opt_x1['x_min_even', -1, 0,0] - x_RCIS_minus)
        lb_g1.append(np.zeros((2 * nx, 1)))
        ub_g1.append(inf * np.ones((2 * nx, 1)))
        
    # Concatenate constraints
    g1 = vertcat(*g1);
    lb_g1 = vertcat(*lb_g1);
    ub_g1 = vertcat(*ub_g1);

# Constraining2 for RCIS
# depending on which dimension was cut lastly we should adjust the corresponding set of the RCIS so that x(N)ExRCIS holds


    if odd_count2 > even_count2:
        g2.append(x_RCIS_plus - opt_x2['x_max_odd', -1, -1,0])
        g2.append(opt_x2['x_min_odd', -1, 0,0] - x_RCIS_minus)
        lb_g2.append(np.zeros((2 * nx, 1)))
        ub_g2.append(inf * np.ones((2 * nx, 1)))
    elif odd_count2 == even_count2:
        g2.append(x_RCIS_plus - opt_x2['x_max_even', -1, -1,0])
        g2.append(opt_x2['x_min_even', -1, 0,0] - x_RCIS_minus)
        lb_g2.append(np.zeros((2 * nx, 1)))
        ub_g2.append(inf * np.ones((2 * nx, 1)))
    
    #Concatenate constraints
    g2 = vertcat(*g2)
    lb_g2 = vertcat(*lb_g2)
    ub_g2 = vertcat(*ub_g2)


# set the problem and initialize the optimizer for the first scenario (dim 0 is cut in the first odd step (1) )

    prob1 = {'f':J1,'x':vertcat(opt_x1),'g':g1, 'p':vertcat(x_init,p_plus,p_minus,u_bef, x_RCIS_plus, x_RCIS_minus)}
    mpc_mon_solver_cut1 = nlpsol('solver','ipopt',prob1,{'ipopt.max_iter':4000,'ipopt.resto_failure_feasibility_threshold':1e-9,'ipopt.required_infeasibility_reduction':0.99,'ipopt.linear_solver':'MA57','ipopt.ma86_u':1e-6,'ipopt.print_level':3, 'ipopt.sb': 'yes', 'print_time':1,'ipopt.ma57_automatic_scaling':'yes','ipopt.ma57_pre_alloc':10,'ipopt.ma27_meminc_factor':100,'ipopt.ma27_pivtol':1e-4,'ipopt.ma27_la_init_factor':100})

# set the problem and initialize the optimizer for the second scenario (dim 1 is cut in the first odd step (1) )
    prob2 = {'f':J2,'x':vertcat(opt_x2),'g':g2, 'p':vertcat(x_init,p_plus,p_minus,u_bef, x_RCIS_plus, x_RCIS_minus)}
    mpc_mon_solver_cut2 = nlpsol('solver','ipopt',prob2,{'ipopt.max_iter':4000,'ipopt.resto_failure_feasibility_threshold':1e-9,'ipopt.required_infeasibility_reduction':0.99,'ipopt.linear_solver':'MA57','ipopt.ma86_u':1e-6,'ipopt.print_level':3, 'ipopt.sb': 'yes', 'print_time':1,'ipopt.ma57_automatic_scaling':'yes','ipopt.ma57_pre_alloc':10,'ipopt.ma27_meminc_factor':100,'ipopt.ma27_pivtol':1e-4,'ipopt.ma27_la_init_factor':100})
# for recursive feasibility two solvers each having a prediction horizon of N whereas the use of one solver is dependant on if we have an even or odd simulation step "i"
# each solver has the corresponding cost, optimization variables and the constraints for the problem
# the partitioning in the next simulation step must align with previous partitioning to enable use of previous optimal solution


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
    opt_x_k1 = opt_x1(1)
    opt_x_k2 = opt_x2(1)
    
    N_sim = 20
    pmin = vertcat(p_fun_wcmin(0))
    pmax = vertcat(p_fun_wcmax(0))
    
    CL_time = time.time()
    opt = []
    sol = []
    subreg = []
    for j in range(N_sim):
    # solve optimization problem
    # if j is odd use solver1 and if j is even use solver 2 so that optimiation problem remains recursively feasible
        print(j)

        if j>0:
            if j % 2 == 0: #cut w.r.t. lis1 in the first (odd) prediction step (1)
                if odd_count1 > even_count1: # set the corresponding terminal constraint according to the N prediction steps
                    mpc_res1 = mpc_mon_solver_cut1(p=vertcat(x_0,pmax,pmin,u_k,np.array(res['x_max_odd',0,-1,0]),np.array(res['x_min_odd',0, 0,0])), x0=opt_x_k1, lbg=lb_g1, ubg=ub_g1, lbx = lb_opt_x1, ubx = ub_opt_x1)
                elif odd_count1 == even_count1:
                    mpc_res1 = mpc_mon_solver_cut1(p=vertcat(x_0,pmax,pmin,u_k,np.array(res['x_max_even',0,-1,0]),np.array(res['x_min_even',0, 0,0])), x0=opt_x_k1, lbg=lb_g1, ubg=ub_g1, lbx = lb_opt_x1, ubx = ub_opt_x1)
            #print return status (mpc_mon_solverf)
                opt.append(copy.deepcopy(mpc_res1))
            elif j % 2 == 1: #cut w.r.t. lis2 in the first (odd) prediction step (1)
                if odd_count2 > even_count2:
                    mpc_res2 = mpc_mon_solver_cut2(p=vertcat(x_0,pmax,pmin,u_k,np.array(res['x_max_even',0,-1,0]),np.array(res['x_min_even',0, 0,0])), x0=opt_x_k2, lbg=lb_g2, ubg=ub_g2, lbx = lb_opt_x2, ubx = ub_opt_x2)
                elif  odd_count2 == even_count2:
                    mpc_res2 = mpc_mon_solver_cut2(p=vertcat(x_0,pmax,pmin,u_k,np.array(res['x_max_odd',0,-1,0]),np.array(res['x_min_odd',0, 0,0])), x0=opt_x_k2, lbg=lb_g2, ubg=ub_g2, lbx = lb_opt_x2, ubx = ub_opt_x2)
                opt.append(copy.deepcopy(mpc_res2))
            
          
                
        
       
        else: # j = 0 and then j % 2 == 0 holds and the first cut will be w.r.t. lis1 
            print('Run a first iteration to generate good Warmstart Values')
            if odd_count1 > even_count1: # set the corresponding terminal constraint according to the N prediction steps
                mpc_res1 = mpc_mon_solver_cut1(p=vertcat(x_0,pmax,pmin, uinitial,np.array(res['x_max_odd',0,-1,0]),np.array(res['x_min_odd',0, 0,0])), x0=opt_x_k1, lbg=lb_g1, ubg=ub_g1, lbx = lb_opt_x1, ubx = ub_opt_x1)
            elif odd_count1 == even_count1:
                mpc_res1 = mpc_mon_solver_cut1(p=vertcat(x_0,pmax,pmin, uinitial,np.array(res['x_max_even',0,-1,0]),np.array(res['x_min_even',0, 0,0])), x0=opt_x_k1, lbg=lb_g1, ubg=ub_g1, lbx = lb_opt_x1, ubx = ub_opt_x1)       
             
            opt_x_k1 = opt_x1(mpc_res1['x'])
            
            
            if odd_count1 > even_count1: # set the corresponding terminal constraint according to the N prediction steps
                mpc_res1 = mpc_mon_solver_cut1(p=vertcat(x_0,pmax,pmin, uinitial,np.array(res['x_max_odd',0,-1,0]),np.array(res['x_min_odd',0, 0,0])), x0=opt_x_k1, lbg=lb_g1, ubg=ub_g1, lbx = lb_opt_x1, ubx = ub_opt_x1)
            elif odd_count1 == even_count1:
                mpc_res1 = mpc_mon_solver_cut1(p=vertcat(x_0,pmax,pmin, uinitial,np.array(res['x_max_even',0,-1,0]),np.array(res['x_min_even',0, 0,0])), x0=opt_x_k1, lbg=lb_g1, ubg=ub_g1, lbx = lb_opt_x1, ubx = ub_opt_x1)
            
            opt.append(copy.deepcopy(mpc_res1))
        # the parameter opt_x_k must have the corresponding dimensions to be passed to the corresponding solver in the next time-step -> case distinction 
        if j % 2 == 0: # odd simulation step follows (j%2 = 1) -> use of mpc-mon_solver_cut2
            opt_x_k1 = opt_x1(mpc_res1['x']) # construction of warmstart "opt_x_k2" from previous solution necessary
            sol.append(copy.deepcopy(opt_x_k1))
            for n in range(1,N2):
                for s in range(n2):
                    opt_x_k2['x_min_odd',n-1,s,:] = opt_x_k1['x_min_even',n,s,:]
                    opt_x_k2['x_max_odd',n-1,s,:] = opt_x_k1['x_max_even',n,s,:]        
            for n in range(N1):
                for s in range(n1):
                    opt_x_k2['x_min_even',n,s,:] = opt_x_k1['x_min_odd',n,s,:]
                    opt_x_k2['x_max_even',n,s,:] = opt_x_k1['x_max_odd',n,s,:] 
            if N2 == N1:
                for s in range(n2):
                    opt_x_k2['x_min_odd',N1-1,s,:] = res['x_min_even',0,s,:]
                    opt_x_k2['x_max_odd',N1-1,s,:] = res['x_max_even',0,s,:]
                for n in range(1,N2):
                    for s in range(n2):
                        opt_x_k2['u_odd',n-1,s] = opt_x_k1['u_even',n,s]
                for n in range(N1-1):
                    for s in range(n1):
                        opt_x_k2['u_even',n,s] = opt_x_k1['u_odd',n,s]
            elif N2 > N1:
                for s in range(n1):
                    opt_x_k2['x_min_even',N2-1,s,:] = res['x_min_odd',0,s,:]
                    opt_x_k2['x_max_even',N2-1,s,:] = res['x_max_odd',0,s,:]
                for n in range(1,N2-1):
                    for s in range(n2):
                        opt_x_k2['u_odd',n-1,s] = opt_x_k1['u_even',n,s]
                for n in range(N1):
                    for s in range(n1):
                        opt_x_k2['u_even',n,s] = opt_x_k1['u_odd',n,s]
           
            u_k = opt_x_k1['u_even',0,0]*ub_u
            
        elif j % 2 == 1: # even simulation step follows (j%2 = 0) -> use of mpc-mon_solver_cut1
            opt_x_k2 = opt_x2(mpc_res2['x'])
            sol.append(copy.deepcopy(opt_x_k2))
            for n in range(N1):
                for s in range(n2):
                    opt_x_k1['x_min_even',n,s,:] = opt_x_k2['x_min_odd',n,s,:]
                    opt_x_k1['x_max_even',n,s,:] = opt_x_k2['x_max_odd',n,s,:]
            for n in range(1,N2):
                for s in range(n1):
                    opt_x_k1['x_min_odd',n-1,s,:] = opt_x_k2['x_min_even',n,s,:]
                    opt_x_k1['x_max_odd',n-1,s,:] = opt_x_k2['x_max_even',n,s,:]          
            if N2 == N1:
                for s in range(n1):
                    opt_x_k1['x_min_odd',N1-1,s,:] = res['x_min_odd',0,s,:]
                    opt_x_k1['x_max_odd',N1-1,s,:] = res['x_max_odd',0,s,:]
                for n in range(1,N2):
                    for s in range(n1):
                        opt_x_k1['u_odd',n-1,s] = opt_x_k2['u_even',n,s]
                for n in range(N1-1):
                    for s in range(n2):
                        opt_x_k1['u_even',n,s] = opt_x_k2['u_odd',n,s]
            elif N2 > N1:       
                for s in range(n2):
                    opt_x_k1['x_min_even',N2-1,s,:] = res['x_min_even',0,s,:]
                    opt_x_k1['x_max_even',N2-1,s,:] = res['x_max_even',0,s,:]
                for n in range(1,N2-1):
                    for s in range(n1):
                        opt_x_k1['u_odd',n-1,s] = opt_x_k2['u_even',n,s]
                for n in range(N1):
                    for s in range(n2):
                        opt_x_k1['u_even',n,s] = opt_x_k2['u_odd',n,s]
            
            u_k = opt_x_k2['u_even',0,0]*ub_u
           
    # simulate the system
        x_next = simulator.make_step(u_k)

    
    # Update the initial state
        x_0 = x_next/ub_x
    

    subreg.append(n1); subreg.append(n2)
    CL_time = time.time()-CL_time #time measurement for the closed-loop-run 
    mpc_mon_cut_res=copy.copy(simulator.data)
    
    datadict = {'CL': [mpc_mon_cut_res,CL_time], 'OL': [opt,sol], 'ns': [subreg], 'counter': [N1,N2]}
    
    return datadict

dd = closed_loop_comp(0,0,0,0,0,0,0,0,0,[0,1,2,3],[0,1,2,3])
np.save('datadict_CL_run_different_0',dd) 

#make hundred runs with modified decomposition function in order to generate scatter plots of the closed-loop trajectory for many different uncertainty-scenarios
# sim_p = []
# t_sim = dd['CL'][0]['_time']
# for i in range(100):
#     sim_p.append(closed_loop_comp(i,1,1,0,0,0,0,2,1,[0,1,2,3],[0,1,2,3])['CL'][0])

# size = 1
# fig = plt.figure(layout="constrained",figsize=(size*15,size*10)) #width and height
# ax_dict = fig.subplot_mosaic(
#     [
#         ["cA", "cB"],
#         ["teta","teta_K"],
#     ],empty_sentinel="X", gridspec_kw = {"wspace" : 0.2, "hspace" : 0.3}
# )

# for i in sim_p:
#     ax_dict["cA"].plot(t_sim,i['_x','cA'],label="cA_closed-loop")
#     ax_dict["cB"].plot(t_sim,i['_x','cB'],label="cB_closed-loop")
#     ax_dict["teta"].plot(t_sim,i['_x','teta'],label="teta_closed-loop")
#     ax_dict["teta_K"].plot(t_sim,i['_x','teta_K'],label="teta_K_closed-loop")

# ax_dict["cA"].grid(False)
# ax_dict["cA"].axhline(y=cAs,color = 'red',linestyle ='-',linewidth=2)
# ax_dict["cA"].axhline(y=lb_cA ,color = 'red',linestyle ='--',linewidth=2)
# ax_dict["cA"].axhline(y=ub_cA ,color = 'red',linestyle ='--',linewidth=2)
# ax_dict["cA"].set_xlabel("time [h]")
# ax_dict["cA"].xaxis.labelpad = 20
# ax_dict["cA"].set_ylabel("c_A [mol/l]",rotation = 0)
# ax_dict["cA"].yaxis.labelpad = 80
# ax_dict["cB"].grid(False)
# ax_dict["cB"].axhline(y=cBs,color = 'red',linestyle ='-',linewidth=2)
# ax_dict["cB"].axhline(y=lb_cB ,color = 'red',linestyle ='--',linewidth=2)
# ax_dict["cB"].axhline(y=ub_cB ,color = 'red',linestyle ='--',linewidth=2)
# ax_dict["cB"].set_xlabel("time [h]", labelpad=1)
# ax_dict["cB"].xaxis.labelpad = 20
# ax_dict["cB"].set_ylabel("c_B [mol/l]",rotation = 0, labelpad=12)
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
# ax_dict["teta_K"].set_ylabel("teta_K [°C]",rotation = 0, labelpad=12)
# ax_dict["teta_K"].yaxis.labelpad = 80
# plt.savefig("full_partitioning_with p_variation.pdf", format="pdf")
# plt.savefig("full_partitioning_with p_variation.svg", format="svg")





# plot the simulation-trajectories of states (cA,cB,teta and teta_K) and the inputs (F and Qk) over the time (simulation timesteps)

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

size = 1
fig = plt.figure(layout="constrained",figsize=(size*30,size*10)) #width and height
ax_dict = fig.subplot_mosaic(
    [
        ["cA", "cB","teta","teta_K"],
        ["F","Qk","J_CL","X"]
    ],empty_sentinel="X", gridspec_kw = {"wspace" : 0.2, "hspace" : 0.3}
)

t_sim = dd['CL'][0]['_time']
N_sim = dd['CL'][0]['_time'].shape[0]
t_pred = np.arange(0,3.1,0.1)
N = np.array(dd['OL'][1][0]['x_max_even',:,-1,0,0]).shape[0] + np.array(dd['OL'][1][0]['x_max_odd',:,-1,0,0]).shape[0]

#from the OL-predictions in the CL-simulation appropriate lists of the open-loop predictions have to be constructed for plotting
#construction of appropriate lists for plotting the predictions
#if current step is even then the next x will be odd and vice versa
even_count = 0
odd_count = 0
cA_max = []
cA_min = []
cB_max = []
cB_min = []
teta_max = []
teta_min = []
tetaK_max = []
tetaK_min = []

for i in range(N_sim):
    even_count = 0 #both have to be set 0 in every simulation step
    odd_count = 0
    cAmax = np.array(dd['OL'][1][i]['x_max_even',even_count,-1,0,0])*scaling_x[0]
    cAmin = np.array(dd['OL'][1][i]['x_min_even',even_count,0,0,0])*scaling_x[0]
    cBmax = np.array(dd['OL'][1][i]['x_max_even',even_count,-1,0,1])*scaling_x[1]
    cBmin = np.array(dd['OL'][1][i]['x_min_even',even_count,0,0,1])*scaling_x[1]
    tetamax = np.array(dd['OL'][1][i]['x_max_even',even_count,-1,0,2])*scaling_x[2]
    tetamin = np.array(dd['OL'][1][i]['x_min_even',even_count,0,0,2])*scaling_x[2]
    tetaKmax = np.array(dd['OL'][1][i]['x_max_even',even_count,-1,0,3])*scaling_x[3]
    tetaKmin = np.array(dd['OL'][1][i]['x_min_even',even_count,0,0,3])*scaling_x[3]
    
    even_count += 1
    
    for p in range(1,N):

        if p%2 == 0:
            cAmax = np.concatenate((copy.deepcopy(cAmax),np.array(dd['OL'][1][i]['x_max_even',even_count,-1,0,0])*scaling_x[0]),axis = 0)
            cAmin = np.concatenate((copy.deepcopy(cAmin),np.array(dd['OL'][1][i]['x_min_even',even_count,0,0,0])*scaling_x[0]),axis = 0)
            cBmax = np.concatenate((copy.deepcopy(cBmax),np.array(dd['OL'][1][i]['x_max_even',even_count,-1,0,1])*scaling_x[1]),axis = 0)
            cBmin = np.concatenate((copy.deepcopy(cBmin),np.array(dd['OL'][1][i]['x_min_even',even_count,0,0,1])*scaling_x[1]),axis = 0)
            tetamax = np.concatenate((copy.deepcopy(tetamax),np.array(dd['OL'][1][i]['x_max_even',even_count,-1,0,2])*scaling_x[2]),axis = 0)
            tetamin = np.concatenate((copy.deepcopy(tetamin),np.array(dd['OL'][1][i]['x_min_even',even_count,0,0,2])*scaling_x[2]),axis = 0)
            tetaKmax = np.concatenate((copy.deepcopy(tetaKmax),np.array(dd['OL'][1][i]['x_max_even',even_count,-1,0,3])*scaling_x[3]),axis = 0)
            tetaKmin = np.concatenate((copy.deepcopy(tetaKmin),np.array(dd['OL'][1][i]['x_min_even',even_count,0,0,3])*scaling_x[3]),axis = 0)
        
            even_count += 1
        elif p%2 == 1:
            cAmax = np.concatenate((copy.deepcopy(cAmax),np.array(dd['OL'][1][i]['x_max_odd',odd_count,-1,0,0])*scaling_x[0]),axis = 0)
            cAmin = np.concatenate((copy.deepcopy(cAmin),np.array(dd['OL'][1][i]['x_min_odd',odd_count,0,0,0])*scaling_x[0]),axis = 0)
            cBmax = np.concatenate((copy.deepcopy(cBmax),np.array(dd['OL'][1][i]['x_max_odd',odd_count,-1,0,1])*scaling_x[1]),axis = 0)
            cBmin = np.concatenate((copy.deepcopy(cBmin),np.array(dd['OL'][1][i]['x_min_odd',odd_count,0,0,1])*scaling_x[1]),axis = 0)
            tetamax = np.concatenate((copy.deepcopy(tetamax),np.array(dd['OL'][1][i]['x_max_odd',odd_count,-1,0,2])*scaling_x[2]),axis = 0)
            tetamin = np.concatenate((copy.deepcopy(tetamin),np.array(dd['OL'][1][i]['x_min_odd',odd_count,0,0,2])*scaling_x[2]),axis = 0)
            tetaKmax = np.concatenate((copy.deepcopy(tetaKmax),np.array(dd['OL'][1][i]['x_max_odd',odd_count,-1,0,3])*scaling_x[3]),axis = 0)
            tetaKmin = np.concatenate((copy.deepcopy(tetaKmin),np.array(dd['OL'][1][i]['x_min_odd',odd_count,0,0,3])*scaling_x[3]),axis = 0)
            
            odd_count += 1

    cA_max.append(cAmax)
    cA_min.append(cAmin)
    cB_max.append(cBmax)
    cB_min.append(cBmin)
    teta_max.append(tetamax)
    teta_min.append(tetamin)
    tetaK_max.append(tetaKmax)
    tetaK_min.append(tetaKmin)

# states 

ax_dict["cA"].plot(t_sim,dd['CL'][0]['_x','cA'],label="cA_closed-loop")
# ax_dict["cA"].plot(t_pred,cA_max[0],color='red')
# ax_dict["cA"].plot(t_pred,cA_min[0],color='green')
ax_dict["cA"].axhline(y=cAs,color = 'red',linestyle ='-',linewidth=2)
ax_dict["cA"].axhline(y=lb_cA ,color = 'red',linestyle ='--',linewidth=2)
ax_dict["cA"].axhline(y=ub_cA ,color = 'red',linestyle ='--',linewidth=2)
ax_dict["cA"].set_xlabel("time [h]",labelpad=1,fontsize = 24)
ax_dict["cA"].xaxis.labelpad = 20
ax_dict["cA"].set_ylabel("cA [mol/l]",labelpad=12,rotation = 0,fontsize = 24)
ax_dict["cA"].yaxis.labelpad = 80
ax_dict["cA"].legend(fontsize ='large',loc='upper right')
# ax_dict["cA"].set_title('cA over time',fontweight = 'bold',fontsize = 24, pad = 12)
ax_dict["cA"].grid(False)

ax_dict["cB"].plot(t_sim,dd['CL'][0]['_x','cB'],label="cB_closed-loop")
# ax_dict["cB"].plot(t_pred,cB_max[0],color='red')
# ax_dict["cB"].plot(t_pred,cB_min[0],color='green')
ax_dict["cB"].axhline(y=cBs,color = 'red',linestyle ='-',linewidth=2)
ax_dict["cB"].axhline(y=lb_cB ,color = 'red',linestyle ='--',linewidth=2)
ax_dict["cB"].axhline(y=ub_cB ,color = 'red',linestyle ='--',linewidth=2)
ax_dict["cB"].set_xlabel("time [h]",labelpad=1,fontsize = 24)
ax_dict["cB"].xaxis.labelpad = 20
ax_dict["cB"].set_ylabel("cB [mol/l]",rotation = 0, labelpad=12,fontsize = 24 )
ax_dict["cB"].yaxis.labelpad = 80
ax_dict["cB"].legend(fontsize ='large',loc='upper right')
# ax_dict["cB"].set_title('cB over time',fontsize = 24, pad = 12)
ax_dict["cB"].grid(False)

ax_dict["teta"].plot(t_sim,dd['CL'][0]['_x','teta'],label="teta_closed-loop")
# ax_dict["teta"].plot(t_pred,teta_max[0],color='red')
# ax_dict["teta"].plot(t_pred,teta_min[0],color='green')
ax_dict["teta"].axhline(y=tetas,color = 'red',linestyle ='-',linewidth=2)
ax_dict["teta"].axhline(y=lb_teta ,color = 'red',linestyle ='--',linewidth=2)
ax_dict["teta"].axhline(y=ub_teta ,color = 'red',linestyle ='--',linewidth=2)
ax_dict["teta"].set_xlabel("time [h]", labelpad=1,fontsize = 24)
ax_dict["teta"].xaxis.labelpad = 20
ax_dict["teta"].set_ylabel("teta [°C]",rotation = 0, labelpad=12,fontsize = 24 )
ax_dict["teta"].yaxis.labelpad = 80
ax_dict["teta"].legend(fontsize ='large',loc='upper right')
# ax_dict["teta"].set_title('teta over time',fontsize = 24, pad = 12)
ax_dict["teta"].grid(False)

ax_dict["teta_K"].plot(t_sim,dd['CL'][0]['_x','teta_K'],label="teta_K_closed-loop")
# ax_dict["teta_K"].plot(t_pred,tetaK_max[0],color='red')
# ax_dict["teta_K"].plot(t_pred,tetaK_min[0],color='green')
ax_dict["teta_K"].axhline(y=tetaKs,color = 'red',linestyle ='-',linewidth=2)
ax_dict["teta_K"].axhline(y=lb_teta_K ,color = 'red',linestyle ='--',linewidth=2)
ax_dict["teta_K"].axhline(y=ub_teta_K ,color = 'red',linestyle ='--',linewidth=2)
ax_dict["teta_K"].set_xlabel("time [h]", labelpad=1,fontsize = 24)
ax_dict["teta_K"].xaxis.labelpad = 20
ax_dict["teta_K"].set_ylabel("teta_K [°C]",rotation = 0, labelpad=12,fontsize = 24)
ax_dict["teta_K"].yaxis.labelpad = 80
ax_dict["teta_K"].legend(fontsize ='large',loc='upper right')
# ax_dict["teta_K"].set_title('teta_K over time',fontsize = 24, pad = 12)
ax_dict["teta_K"].grid(False)
# inputs 
ax_dict["F"].step(t_sim,dd['CL'][0]['_u','u_F'],where = 'post',label="F_closed-loop")
ax_dict["F"].axhline(y=Fs,color = 'red',linestyle ='-',linewidth=2)
ax_dict["F"].axhline(y=lb_F ,color = 'red',linestyle ='--',linewidth=2)
ax_dict["F"].axhline(y=ub_F ,color = 'red',linestyle ='--',linewidth=2)
ax_dict["F"].set_xlabel("time [h]", labelpad=1,fontsize = 24)
ax_dict["F"].xaxis.labelpad = 20
ax_dict["F"].set_ylabel("F [l/h]",rotation = 0, labelpad=12,fontsize = 24)
ax_dict["F"].yaxis.labelpad = 100
ax_dict["F"].legend(fontsize ='large',loc='upper right')
# ax_dict["F"].set_title('F over time',fontsize = 24, pad = 12)
ax_dict["F"].grid(False)

ax_dict["Qk"].step(t_sim,dd['CL'][0]['_u','u_Qk'],where = 'post',label="Qk_closed-loop")
ax_dict["Qk"].axhline(y=Qks,color = 'red',linestyle ='-',linewidth=2)
ax_dict["Qk"].axhline(y=lb_Qk ,color = 'red',linestyle ='--',linewidth=2)
ax_dict["Qk"].axhline(y=ub_Qk ,color = 'red',linestyle ='--',linewidth=2)
ax_dict["Qk"].set_xlabel("time [h]", labelpad=1,fontsize = 24)
ax_dict["Qk"].xaxis.labelpad = 20
ax_dict["Qk"].set_ylabel("Qk [kJ/h]",rotation = 0, labelpad=12,fontsize = 24)
ax_dict["Qk"].yaxis.labelpad = 80
ax_dict["Qk"].legend(fontsize ='large')
# ax_dict["Qk"].set_title('Qk over time',fontsize = 24, pad = 12)
ax_dict["Qk"].grid(False)

ax_dict["J_CL"].step(t_sim,closed_loop_cost(dd)[1:],where = 'post',label="J_closed-loop")
ax_dict["J_CL"].set_xlabel("time [h]", labelpad=1,fontsize = 24)
ax_dict["J_CL"].xaxis.labelpad = 20
ax_dict["J_CL"].set_ylabel("J_CL",rotation = 0, labelpad=12,fontsize = 24)
ax_dict["J_CL"].yaxis.labelpad = 80
ax_dict["J_CL"].legend(fontsize ='large')
# ax_dict["J_CL"].set_title('closed-loop cost',fontsize = 24, pad = 12)
ax_dict["J_CL"].grid(False)

# plt.savefig("alternating_variable_Ns_40.pdf", format="pdf")
# plt.savefig("alternating_variable_Ns_40.svg", format="svg")

