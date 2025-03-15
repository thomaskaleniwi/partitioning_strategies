# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 14:22:28 2025

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

datadict_CL_run_different = np.load('datadict_CL_run_different.npy',allow_pickle = True).item() #item() converts loaded object to a python scalar object
datadict_CL_run_constant = np.load('datadict_CL_run_constant.npy',allow_pickle = True).item()
datadict_CL_run_full = np.load('datadict_CL_run_full.npy',allow_pickle = True).item()

######### visualization of the u-predictions for the three different partitioning scenarios

N = 31 #(np.array(datadict_CL_run_constant['OL'][1][0]['x_max',:,-1,0]).shape[0] to get the total amount of predictions whereas the first prediction is always the initial constraint)  prediction steps that should be plotted
N_sim = datadict_CL_run_constant['CL'][0]['_x','x'].shape[0] # determine the simulation steps
fig1, ax1 = plt.subplots(1,2,figsize=(20,12)) # create plot and assign an index to each plot
# create lists of the predictions to be plotted in the corresponding simulation time-steps
plot0 = [0]
# create a list of the(predicted 'OL') input-data to be plotted in order to determine which input-data should be plotted

time1u = [np.linspace(i,i+N-1,num=N-1,endpoint = False)for i in range(0,N_sim)]
ns_constant = datadict_CL_run_constant['ns'][0][0]
color = ['red', 'black', 'green', 'yellow', 'purple']

ax1[0].step(datadict_CL_run_constant['CL'][0]['_time'],datadict_CL_run_constant['CL'][0]['_u','u'][:,0],where = 'post', label = "closed-loop")

for i in plot0: #plotting of u0-open loop predictions in the i-th closed-loop simulation step for all subregions
    for s in range(ns_constant):
        ax1[0].step(time1u[i],np.array(datadict_CL_run_constant['OL'][1][i]['u',0:N-1,s,0]).reshape(N-1,1),where = 'post', color = color[0], label = "prediction: {0}".format(i),linewidth=0.5)
    for p in range(N):
        for s in range(ns_constant): #add a small number at the bottom right end of each prediction trajectory revealing to which subregion it belongs to
            ax1[0].text(time1u[i][-1],np.array(datadict_CL_run_constant['OL'][1][i]['u',0:N-1,s,0]).reshape(N-1,1)[-1],str(s),ha = 'left', va = 'bottom', color = 'black', fontweight = 'bold',fontsize ='xx-small')

ax1[1].step(datadict_CL_run_constant['CL'][0]['_time'],datadict_CL_run_constant['CL'][0]['_u','u'][:,1],where = 'post', label = "closed-loop")

for i in plot0: #plotting of u1-open loop predictions in the i-th closed-loop simulation step
    for s in range(ns_constant):
        ax1[1].step(time1u[i],np.array(datadict_CL_run_constant['OL'][1][i]['u',0:N-1,s,0]).reshape(N-1,1),where = 'post', color = color[0], label = "prediction: {0}".format(i),linewidth=0.5)
    for p in range(N):
        for s in range(ns_constant): #add a small number at the bottom right end of each prediction trajectory revealing to which subregion it belongs to
            ax1[1].text(time1u[i][-1],np.array(datadict_CL_run_constant['OL'][1][i]['u',0:N-1,s,1]).reshape(N-1,1)[-1],str(s),ha = 'left', va = 'bottom', color = 'black', fontweight = 'bold',fontsize ='xx-small')

ax1[0].set_ylabel('u0',rotation = 0,fontweight = 'bold',fontsize = 24,labelpad = 12)
ax1[0].set_ylim(-7,3)
ax1[0].set_xlabel('time [s]',fontweight = 'bold',fontsize = 24)
ax1[0].set_title('u0 over time',fontweight = 'bold',fontsize = 24, pad = 12)
ax1[0].grid(False)
ax1[0].tick_params(axis='both', labelsize=24)
ax1[0].legend(['closed-loop','prediction: {0}'.format(i)],fontsize ='xx-small',loc='upper right')
ax1[1].set_ylabel('u1',rotation = 0,fontweight = 'bold',fontsize = 24,labelpad = 18)
ax1[1].set_ylim(-7,2)
ax1[1].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
ax1[1].set_xlabel('time [s]',fontweight = 'bold',fontsize = 24)
ax1[1].set_title('u1 over time',fontweight = 'bold',fontsize = 24, pad = 12)
ax1[1].grid(False)
ax1[1].tick_params(axis='both', labelsize=24)
ax1[1].legend(['closed-loop','prediction: {0}'.format(i)],fontsize ='xx-small',loc='upper right')

fig1.suptitle("input trajectories (alternating-constant [ns = {0}])".format(datadict_CL_run_constant['ns'][0][0]),fontweight = 'bold',fontsize = 28)
fig1.align_labels()
fig1.tight_layout(pad = 3)

plt.draw()

####### visualization of the input predictions in case the number of subregions differs from one partition-step to the following

fig2, ax2 = plt.subplots(1,2,figsize=(20,12))

N1 = datadict_CL_run_different['counter'][0] #number of odd prediction steps
n1 = datadict_CL_run_different['ns'][0][0] #number of odd subregions
N2 = datadict_CL_run_different['counter'][1] #number of even prediction steps
n2 = datadict_CL_run_different['ns'][0][1] #number of even subregions
time2u = [np.arange(i,i+N-1) for i in range(N_sim)] #create a list for the corresponding prediction-time horizon for each simulation step and afterwards create two lists with the corresponding even and odd time steps 
time2u_even = [time2u[i][time2u[i]%2==0] for i in range(N_sim)]
time2u_odd = [time2u[i][time2u[i]%2==1] for i in range(N_sim)]


line1, = ax2[0].step(datadict_CL_run_different['CL'][0]['_time'],datadict_CL_run_different['CL'][0]['_u','u'][:,0],where = 'post',label='closed loop')

for i in plot0: #plotting of u0-open loop predictions in the i-th closed-loop simulation step
    even_count = 0 #both have to be set 0 in every simulation step
    odd_count = 0
    
    if N2 > N1: #for determining the last input step and reshaping it correctly
        for p in range(N): #for the length of the prediction horizon plot both (even and odd) predictions and assign them different colors
            if p%2 == 0:
                for s2 in range(n2):
                    line2, = ax2[0].step(time2u_even[i],np.array(datadict_CL_run_different['OL'][1][i]['u_even',:,s2,0]).reshape(N2-1,1),where = 'post', color = color[0], label = "even prediction[n2={1}]: {0}".format(i,n2),linewidth=0.5)
                    if p == N-1:
                        ax2[0].text(time2u_even[i][-1],np.array(datadict_CL_run_different['OL'][1][i]['u_even',:,s2,0]).reshape(N2-1,1)[-1],str(s2),ha = 'left', va = 'bottom', color = 'black', fontweight = 'bold',fontsize ='xx-small')
                even_count += 1
            elif p%2 == 1:
                for s1 in range(n1):
                    line3, = ax2[0].step(time2u_odd[i],np.array(datadict_CL_run_different['OL'][1][i]['u_odd',:,s1,0]).reshape(N1,1),where = 'post', color = color[1], label = "odd prediction[n1={1}]: {0}".format(i,n1),linewidth=0.5)
                    if p == N-2:
                        ax2[0].text(time2u_odd[i][-1],np.array(datadict_CL_run_different['OL'][1][i]['u_odd',:,s1,0]).reshape(N1,1)[-1],str(s1),ha = 'left', va = 'bottom', color = 'black', fontweight = 'bold',fontsize ='xx-small')
                odd_count += 1
    elif N2 == N1:
        for p in range(N): #for the length of the prediction horizon plot both (even and odd) predictions and assign them different colors
            if p%2 == 0:
                for s2 in range(n2):
                    line2, = ax2[0].step(time2u_even[i],np.array(datadict_CL_run_different['OL'][1][i]['u_even',:,s2,0]).reshape(N2,1),where = 'post', color = color[0], label = "even prediction[n2={1}]: {0}".format(i,n2),linewidth=0.5)
                    if p == N-2:
                        ax2[0].text(time2u_even[i][-1],np.array(datadict_CL_run_different['OL'][1][i]['u_even',:,s2,0]).reshape(N2,1)[-1],str(s2),ha = 'left', va = 'bottom', color = 'black', fontweight = 'bold',fontsize ='xx-small')
                even_count += 1
            elif p%2 == 1:
                for s1 in range(n1):
                    line3, = ax2[0].step(time2u_odd[i],np.array(datadict_CL_run_different['OL'][1][i]['u_odd',:,s1,0]).reshape(N1-1,1),where = 'post', color = color[1], label = "odd prediction[n1={1}]: {0}".format(i,n1),linewidth=0.5)
                    if p == N-1:
                        ax2[0].text(time2u_odd[i][-1],np.array(datadict_CL_run_different['OL'][1][i]['u_odd',:,s1,0]).reshape(N1-1,1)[-1],str(s1),ha = 'left', va = 'bottom', color = 'black', fontweight = 'bold',fontsize ='xx-small')
                odd_count += 1
line4, = ax2[1].step(datadict_CL_run_different['CL'][0]['_time'],datadict_CL_run_different['CL'][0]['_u','u'][:,1],where = 'post',label='closed loop')

for i in plot0: #plotting of u1-open loop predictions in the i-th closed-loop simulation step
    even_count = 0 #both have to be set 0 in every simulation step
    odd_count = 0
    
    if N2 > N1: #for determining the last input step and reshaping it correctly
        for p in range(N): #for the length of the prediction horizon plot both (even and odd) predictions and assign them different colors
            if p%2 == 0:
                for s2 in range(n2):
                    line5, = ax2[1].step(time2u_even[i],np.array(datadict_CL_run_different['OL'][1][i]['u_even',:,s2,1]).reshape(N2-1,1),where = 'post', color = color[0], label = "even prediction[n2={1}]: {0}".format(i,n2),linewidth=0.5)
                    if p == N-1:
                        ax2[1].text(time2u_even[i][-1],np.array(datadict_CL_run_different['OL'][1][i]['u_even',:,s2,1]).reshape(N2-1,1)[-1],str(s2),ha = 'left', va = 'bottom', color = 'black', fontweight = 'bold',fontsize ='xx-small')
                even_count += 1
            elif p%2 == 1:
                for s1 in range(n1):
                    line6, = ax2[1].step(time2u_odd[i],np.array(datadict_CL_run_different['OL'][1][i]['u_odd',:,s1,1]).reshape(N1,1),where = 'post', color = color[1], label = "odd prediction[n1={1}]: {0}".format(i,n1),linewidth=0.5)
                    if p == N-2:
                        ax2[1].text(time2u_odd[i][-1],np.array(datadict_CL_run_different['OL'][1][i]['u_odd',:,s1,1]).reshape(N1,1)[-1],str(s1),ha = 'left', va = 'bottom', color = 'black', fontweight = 'bold',fontsize ='xx-small')
                odd_count += 1
    elif N2 == N1:
        for p in range(N): #for the length of the prediction horizon plot both (even and odd) predictions and assign them different colors
            if p%2 == 0:
                for s2 in range(n2):
                    line5, = ax2[1].step(time2u_even[i],np.array(datadict_CL_run_different['OL'][1][i]['u_even',:,s2,1]).reshape(N2,1),where = 'post', color = color[0], label = "even prediction[n2={1}]: {0}".format(i,n2),linewidth=0.5)
                    if p == N-2:
                        ax2[1].text(time2u_even[i][-1],np.array(datadict_CL_run_different['OL'][1][i]['u_even',:,s2,1]).reshape(N2,1)[-1],str(s2),ha = 'left', va = 'bottom', color = 'black', fontweight = 'bold',fontsize ='xx-small')
                even_count += 1
            elif p%2 == 1:
                for s1 in range(n1):
                    line6, = ax2[1].step(time2u_odd[i],np.array(datadict_CL_run_different['OL'][1][i]['u_odd',:,s1,1]).reshape(N1-1,1),where = 'post', color = color[1], label = "odd prediction[n1={1}]: {0}".format(i,n1),linewidth=0.5)
                    if p == N-1:
                        ax2[1].text(time2u_odd[i][-1],np.array(datadict_CL_run_different['OL'][1][i]['u_odd',:,s1,1]).reshape(N1-1,1)[-1],str(s1),ha = 'left', va = 'bottom', color = 'black', fontweight = 'bold',fontsize ='xx-small')
                odd_count += 1
    
ax2[0].set_ylabel('u0',rotation = 0,fontweight = 'bold',fontsize = 24,labelpad = 12)
ax2[0].set_ylim(-7,3)
ax2[0].set_xlabel('time [s]',fontweight = 'bold',fontsize = 24)
ax2[0].set_title('u0 over time',fontweight = 'bold',fontsize = 24, pad = 12)
ax2[0].grid(False)
ax2[0].tick_params(axis='both', labelsize=24)
ax2[0].legend([line1, line2, line3],['closed-loop','prediction_even: {0}'.format(i),'prediction_odd: {0}'.format(i)],fontsize ='xx-small',loc='upper right')
ax2[1].set_ylabel('u1',rotation = 0,fontweight = 'bold',fontsize = 24,labelpad = 18)
ax2[1].set_ylim(-7,2)
ax2[1].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
ax2[1].set_xlabel('time [s]',fontweight = 'bold',fontsize = 24)
ax2[1].set_title('u1 over time',fontweight = 'bold',fontsize = 24, pad = 12)
ax2[1].grid(False)
ax2[1].tick_params(axis='both', labelsize=24)
ax2[1].legend([line4, line5, line6],['closed-loop','prediction_even: {0}'.format(i),'prediction_odd: {0}'.format(i)],fontsize ='xx-small',loc='upper right')

fig2.suptitle("input trajectories (alternating-different [n1 = {0}, n2 = {1}])".format(n1,n2),fontweight = 'bold',fontsize = 28)
fig2.align_labels()
fig2.tight_layout(pad = 3)

####### visualization of the input predictions for the case that the both dimensions are cut fully in each timestep in the same way 

fig3, ax3 = plt.subplots(1,2,figsize=(20,12))

time3u = [np.linspace(i,i+N-1,num=N-1,endpoint = False)for i in range(0,N_sim)]
ns_full = datadict_CL_run_full['ns'][0][0]

ax3[0].step(datadict_CL_run_full['CL'][0]['_time'],datadict_CL_run_full['CL'][0]['_u','u'][:,0],where = 'post', label = "closed-loop")

for i in plot0: #plotting of u0-open loop predictions in the i-th closed-loop simulation step for all subregions
    for s in range(ns_full):
        ax3[0].step(time3u[i],np.array(datadict_CL_run_full['OL'][1][i]['u',0:N-1,s,0]).reshape(N-1,1),where = 'post', color = color[0], label = "prediction: {0}".format(i),linewidth=0.5)
    for p in range(N):
        for s in range(ns_full): #add a small number at the bottom right end of each prediction trajectory revealing to which subregion it belongs to
            ax3[0].text(time3u[i][-1],np.array(datadict_CL_run_full['OL'][1][i]['u',0:N-1,s,0]).reshape(N-1,1)[-1],str(s),ha = 'left', va = 'bottom', color = 'black', fontweight = 'bold',fontsize ='xx-small')

ax3[1].step(datadict_CL_run_full['CL'][0]['_time'],datadict_CL_run_full['CL'][0]['_u','u'][:,1],where = 'post', label = "closed-loop")

for i in plot0: #plotting of u1-open loop predictions in the i-th closed-loop simulation step
    for s in range(ns_full):
        ax3[1].step(time3u[i],np.array(datadict_CL_run_full['OL'][1][i]['u',0:N-1,s,0]).reshape(N-1,1),where = 'post', color = color[0], label = "prediction: {0}".format(i),linewidth=0.5)
    for p in range(N):
        for s in range(ns_full): #add a small number at the bottom right end of each prediction trajectory revealing to which subregion it belongs to
            ax3[1].text(time3u[i][-1],np.array(datadict_CL_run_full['OL'][1][i]['u',0:N-1,s,1]).reshape(N-1,1)[-1],str(s),ha = 'left', va = 'bottom', color = 'black', fontweight = 'bold',fontsize ='xx-small')

ax3[0].set_ylabel('u0',rotation = 0,fontweight = 'bold',fontsize = 24,labelpad = 12)
ax3[0].set_ylim(-7,3)
ax3[0].set_xlabel('time [s]',fontweight = 'bold',fontsize = 24)
ax3[0].set_title('u0 over time',fontweight = 'bold',fontsize = 24, pad = 12)
ax3[0].grid(False)
ax3[0].tick_params(axis='both', labelsize=24)
ax3[0].legend(['closed-loop','prediction: {0}'.format(i)],fontsize ='xx-small',loc='upper right')
ax3[1].set_ylabel('u1',rotation = 0,fontweight = 'bold',fontsize = 24,labelpad = 18)
ax3[1].set_ylim(-7,2)
ax3[1].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
ax3[1].set_xlabel('time [s]',fontweight = 'bold',fontsize = 24)
ax3[1].set_title('u1 over time',fontweight = 'bold',fontsize = 24, pad = 12)
ax3[1].grid(False)
ax3[1].tick_params(axis='both', labelsize=24)
ax3[1].legend(['closed-loop','prediction: {0}'.format(i)],fontsize ='xx-small',loc='upper right')

fig3.suptitle("input trajectories (full partitioning [ns = {0}])".format(datadict_CL_run_full['ns'][0][0]),fontweight = 'bold',fontsize = 28)
fig3.align_labels()
fig3.tight_layout(pad = 3)

plt.draw()