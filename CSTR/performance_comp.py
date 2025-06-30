# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 17:37:48 2025

@author: thomas
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pdb as pdb
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
import pickle
import matplotlib.ticker as ticker

# Custom packages
import do_mpc

datadict_CL_run_different_0 = np.load('datadict_CL_run_different_0.npy',allow_pickle = True).item()
datadict_CL_run_different_4 = np.load('datadict_CL_run_different_4.npy',allow_pickle = True).item()
datadict_CL_run_different_8 = np.load('datadict_CL_run_different_8.npy',allow_pickle = True).item() #item() converts loaded object to a python scalar object
datadict_CL_run_different_16 = np.load('datadict_CL_run_different_16.npy',allow_pickle = True).item()
datadict_CL_run_different_24 = np.load('datadict_CL_run_different_24.npy',allow_pickle = True).item()
datadict_CL_run_different_32 = np.load('datadict_CL_run_different_32.npy',allow_pickle = True).item()
datadict_CL_run_different_40 = np.load('datadict_CL_run_different_40.npy',allow_pickle = True).item()

datadict_CL_run_constant_0 = np.load('datadict_CL_run_constant_0.npy',allow_pickle = True).item()
datadict_CL_run_constant_4 = np.load('datadict_CL_run_constant_4.npy',allow_pickle = True).item()
datadict_CL_run_constant_8 = np.load('datadict_CL_run_constant_8.npy',allow_pickle = True).item()
datadict_CL_run_constant_16 = np.load('datadict_CL_run_constant_16.npy',allow_pickle = True).item()
datadict_CL_run_constant_24 = np.load('datadict_CL_run_constant_24.npy',allow_pickle = True).item()
datadict_CL_run_constant_32 = np.load('datadict_CL_run_constant_32.npy',allow_pickle = True).item()
datadict_CL_run_constant_40 = np.load('datadict_CL_run_constant_40.npy',allow_pickle = True).item()

datadict_CL_run_full_0 = np.load('datadict_CL_run_full_0.npy',allow_pickle = True).item()
datadict_CL_run_full_4 = np.load('datadict_CL_run_full_4.npy',allow_pickle = True).item()
datadict_CL_run_full_8 = np.load('datadict_CL_run_full_8.npy',allow_pickle = True).item()
datadict_CL_run_full_16 = np.load('datadict_CL_run_full_16.npy',allow_pickle = True).item()
datadict_CL_run_full_24 = np.load('datadict_CL_run_full_24.npy',allow_pickle = True).item()
datadict_CL_run_full_32 = np.load('datadict_CL_run_full_32.npy',allow_pickle = True).item()
datadict_CL_run_full_40 = np.load('datadict_CL_run_full_40.npy',allow_pickle = True).item()


# extract the the simulation-time for the closed-loop and plot then

t_CL_full = []
t_CL_alt_const = []
t_CL_alt_var = []

J_CL_full = []
J_CL_alt_const = []
J_CL_alt_var = []
ns = [0,4,8,16,24,32,40]
N_sim = 20

#computation of closed-loop cost
R = np.array([[1,0],[0,1]])
Q = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
x_ref = np.array(np.array([2.14,1.09,114.2,112.9]))

def closed_loop_cost(data):
    N_sim = data['CL'][0]['_x','cA'].shape[0] #retrieve the number of simulation steps by 
    J_cl = 0
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
        
        J_cl = J_cl \
            + (x_i - x_ref.reshape((4,1))).T @ Q @ (x_i - x_ref.reshape((4,1))) \
            + (u_i - u_prev).T @ R @ (u_i - u_prev)
        u_prev = u_i
    
    return J_cl

for n in ns:
    varname_full = f"datadict_CL_run_different_{n}"
    varname_const = f"datadict_CL_run_constant_{n}"
    varname_fullpart =   f"datadict_CL_run_full_{n}"
    
    CL_full      = globals()[varname_fullpart]
    CL_alt_const = globals()[varname_const]
    CL_alt_var   = globals()[varname_full]
    
    t_CL_full.append(CL_full['CL'][1]/N_sim)
    t_CL_alt_const.append(CL_alt_const['CL'][1]/N_sim)
    t_CL_alt_var.append(CL_alt_var['CL'][1]/N_sim)
    
    J_CL_full.append(closed_loop_cost(CL_full))
    J_CL_alt_const.append(closed_loop_cost(CL_alt_const))
    J_CL_alt_var.append(closed_loop_cost(CL_alt_var))

fig, ax = plt.subplots(1,2,figsize=(12,4))

ax[0].scatter(ns, t_CL_full,      label="full partitioning")
ax[0].scatter(ns, t_CL_alt_const, label="alternating constant partitioning")
ax[0].scatter(ns, t_CL_alt_var,   label="alternating variable partitioning")
ax[0].set_xlabel("Ns")
ax[0].set_ylabel("t_CL [s]")
ax[0].grid(False)
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[0].legend(fontsize ='small')

ax[1].scatter(ns, J_CL_full,      label="full partitioning")
ax[1].scatter(ns, J_CL_alt_const, label="alternating constant partitioning")
ax[1].scatter(ns, J_CL_alt_var,   label="alternating variable partitioning")
ax[1].set_xlabel("Ns")
ax[1].set_ylabel("J_CL [-]")
ax[1].grid(False)
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].legend(fontsize ='small')

# plt.savefig("performance_comp_CSTR.pdf", format="pdf")
# plt.savefig("performance_comp_CSTR.svg", format="svg")

#plot the states too

#operating point 

cAs = 2.14 #1.235 # mol/l
cBs = 1.09 #0.9 # mol/l
tetas = 114.2 #134.14  # °C
tetaKs = 112.9 #128.95 # °C
Fs = 14.19 #18.83  # h^-1
Qks = -1113.5 #-4495.7 # kJ/h

#bounds
lb_cA = 0 # mol/l
ub_cA = 5.1 # mol/l given by the maximal inflow concentration of A
lb_cB = 0 # mol/l
ub_cB = 2 # mol/l 
lb_teta = 90 # °C
ub_teta = 150 # °C
lb_teta_K = 90 # °C
ub_teta_K = 150 # °C

lb_F = 3 #5 # h^-1
ub_F = 35 # h^-1
lb_Qk = -9000 # -8500  #kJ/h
ub_Qk = 0.1 #kJ/h should be set 0,but in that case there would exist a division through 0

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





dd= datadict_CL_run_full_40
size = 1
fig = plt.figure(layout="constrained",figsize=(size*30,size*10)) #width and height
ax_dict = fig.subplot_mosaic(
    [
        ["cA", "cB","teta","teta_K"],
        ["F","Qk","J_CL","X"]
    ],empty_sentinel="X", gridspec_kw = {"wspace" : 0.2, "hspace" : 0.3}
)

t_sim = dd['CL'][0]['_time']


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
# ax_dict["teta"].set_title('teta over time',fontweight = 'bold',fontsize = 24, pad = 12)
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
ax_dict["J_CL"].set_title('closed-loop cost',fontsize = 24, pad = 12)
ax_dict["J_CL"].grid(False)

# plt.savefig("Full_NS32.pdf", format="pdf")
# plt.savefig("Full_NS32.svg", format="svg")







