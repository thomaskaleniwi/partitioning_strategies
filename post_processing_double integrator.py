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

datadict_CL_run_different = np.load('datadict_CL_run_different.npy',allow_pickle = True).item() #item() converts loaded object to a python scalar object
datadict_CL_run_constant = np.load('datadict_CL_run_constant.npy',allow_pickle = True).item()
datadict_CL_run_full = np.load('datadict_CL_run_full.npy',allow_pickle = True).item()

# computation of closed loop cost
# first determine the simulation steps (N_sim) from the shape of the simulation data

R = np.array([[1,0],[0,1]])
Q = np.array([[1,0],[0,1]])
x_ref = np.array([5,2])

def closed_loop_cost(data):
    N_sim = data['CL'][0]['_x','x'].shape[0] #retrieve the number of simulation steps by 
    J_cl = np.zeros((N_sim+1,1))
    u_prev = np.zeros((2,1))
    
    for i in range(N_sim):
        J_cl[i+1] = J_cl[i] + (data['CL'][0]['_x','x'][i].reshape((2,1))-x_ref.reshape((2,1))).T@Q@(data['CL'][0]['_x','x'][i].reshape((2,1))-x_ref.reshape((2,1))) + (data['CL'][0]['_u','u'][i].reshape((2,1))-u_prev.reshape((2,1))).T@R@(data['CL'][0]['_u','u'][i].reshape((2,1))-u_prev.reshape((2,1)))
        u_prev = data['CL'][0]['_u','u'][i]
    
    return J_cl
#######evaluation of the closed-loop-performance for the case that the number of subregions remains constant
N = 31 #(np.array(datadict_CL_run_constant['OL'][1][0]['x_max',:,-1,0]).shape[0] to get the total amount of predictions whereas the first prediction is always the initial constraint)  prediction steps that should be plotted
N_sim = datadict_CL_run_constant['CL'][0]['_x','x'].shape[0] # determine the simulation steps
fig1, ax1 = plt.subplots(2,3,figsize=(20,12)) # create plot and assign an index to each plot
# create lists of the predictions to be plotted in the corresponding simulation time-steps
plot0 = [0]; plot1 = [0]; plot2 = [0]; plot3 = [0]; plot4 = [0]
# create a list of the(predicted 'OL') data to be plotted in order to determine what data should be plotted



time1x = [np.linspace(i,i+N,num=N,endpoint = False)for i in range(0,N_sim)]
time1u = [np.linspace(i,i+N-1,num=N-1,endpoint = False)for i in range(0,N_sim)]
x0max = [np.array(datadict_CL_run_constant['OL'][1][i]['x_max',0:N,-1,0]).reshape(N,1) for i in range(0,N_sim)] 
x0min = [np.array(datadict_CL_run_constant['OL'][1][i]['x_min',0:N,0,0]).reshape(N,1) for i in range(0,N_sim)]
x1max = [np.array(datadict_CL_run_constant['OL'][1][i]['x_max',0:N,-1,1]).reshape(N,1) for i in range(0,N_sim)]
x1min = [np.array(datadict_CL_run_constant['OL'][1][i]['x_min',0:N,0,1]).reshape(N,1) for i in range(0,N_sim)]
u0 = [np.array(datadict_CL_run_constant['OL'][1][i]['u',0:N-1,0,0]).reshape(N-1,1) for i in range(0,N_sim)]
u1 = [np.array(datadict_CL_run_constant['OL'][1][i]['u',0:N-1,0,1]).reshape(N-1,1) for i in range(0,N_sim)]
                                                                                 
color = ['red', 'black', 'green', 'yellow', 'purple']

ax1[0,0].plot(datadict_CL_run_constant['CL'][0]['_time'],datadict_CL_run_constant['CL'][0]['_x','x'][:,0], label = "closed-loop") # plot evolution of state x0 over simulation time                                                                                                                                                    

for i in plot0: #plotting of x0-open loop predictions in the i-th closed-loop simulation step
    ax1[0,0].plot(time1x[i],x0max[i],color = color[0],label = "prediction: {0}".format(i),linewidth=0.5)
    ax1[0,0].plot(time1x[i],x0min[i],color = color[0],linewidth=0.5)

ax1[0,1].plot(datadict_CL_run_constant['CL'][0]['_time'],datadict_CL_run_constant['CL'][0]['_x','x'][:,1], label = "closed-loop") # plot evolution of state x1 over simulation time

for i in plot1: #plotting of x1-open loop predictions in the i-th closed-loop simulation step
    ax1[0,1].plot(time1x[i],x1max[i],color = color[0],label = "prediction: {0}".format(i),linewidth=0.5)
    ax1[0,1].plot(time1x[i],x1min[i],color = color[0],linewidth=0.5)
    
ax1[0,2].plot(datadict_CL_run_constant['CL'][0]['_x','x'][:,0],datadict_CL_run_constant['CL'][0]['_x','x'][:,1],label = "closed-loop") # plot evolution of state x1 over x0

for i in plot2: #plotting of x0/x1-open loop predictions (rectangles) in the i-th closed-loop simulation step
    for p in range(0,N):
        ax1[0,2].add_patch(mpl.patches.Rectangle(np.concatenate((x0min[i],x1min[i]),axis=1)[p],np.concatenate((x0max[i]-x0min[i],x1max[i]-x1min[i]),axis=1)[p][0],np.concatenate((x0max[i]-x0min[i],x1max[i]-x1min[i]),axis=1)[p][1], color = 'None', ec = color[0],linewidth=0.5 ))
        ax1[0,2].text(x0min[i][p],x1min[i][p],str(p),ha = 'right', va = 'bottom', color = 'black', fontweight = 'bold',fontsize ='xx-small')
# there is no initial rectangle starting at the initial point of the closed loop trajectory, because the first point in the prediction is always constrained to be a point and not a rectangle
    
ax1[1,0].step(datadict_CL_run_constant['CL'][0]['_time'],datadict_CL_run_constant['CL'][0]['_u','u'][:,0],where = 'post', label = "closed-loop")

for i in plot3: #plotting of u0-open loop predictions in the i-th closed-loop simulation step
    ax1[1,0].step(time1u[i],u0[i],where = 'post', color = color[0], label = "prediction: {0}".format(i),linewidth=0.5)

ax1[1,1].step(datadict_CL_run_constant['CL'][0]['_time'],datadict_CL_run_constant['CL'][0]['_u','u'][:,1],where = 'post', label = "closed-loop")

for i in plot4: #plotting of u1-open loop predictions in the i-th closed-loop simulation step
    ax1[1,1].step(time1u[i],u1[i],where = 'post', color = color[0], label = "prediction: {0}".format(i),linewidth=0.5)
#plotting of closed-loop-cost
ax1[1,2].step(datadict_CL_run_constant['CL'][0]['_time'],closed_loop_cost(datadict_CL_run_constant)[0:N_sim],where = 'post', label = "closed-loop cost")



ax1[0,0].set_ylabel('x0',rotation = 0,fontweight = 'bold',fontsize = 24,labelpad = 12)
ax1[0,0].set_xlabel('time [s]',fontweight = 'bold',fontsize = 24)
ax1[0,0].set_ylim(0,7)
ax1[0,0].set_title('x0 over time',fontweight = 'bold',fontsize = 24, pad = 12)
ax1[0,0].grid(False)
ax1[0,0].tick_params(axis='both', labelsize=24)
ax1[0,0].legend(fontsize ='xx-small',loc='upper right')
ax1[0,1].set_ylabel('x1',rotation = 0,fontweight = 'bold',fontsize = 24,labelpad = 12)
ax1[0,1].set_xlabel('time [s]',fontweight = 'bold',fontsize = 24)
ax1[0,1].set_ylim(0,4)
ax1[0,1].set_title('x1 over time',fontweight = 'bold',fontsize = 24, pad = 12)
ax1[0,1].grid(False)
ax1[0,1].tick_params(axis='both', labelsize=24)
ax1[0,1].legend(fontsize ='xx-small',loc='upper right')
ax1[0,2].set_ylabel('x1',rotation = 0,fontweight = 'bold',fontsize = 24,labelpad = 12)
ax1[0,2].set_xlabel('x0',fontweight = 'bold',fontsize = 24)
ax1[0,2].set_ylim(0,5)
ax1[0,2].set_xlim(0,8)
ax1[0,2].set_title('x1 over x0',fontweight = 'bold',fontsize = 24, pad = 12)
ax1[0,2].grid(False)
ax1[0,2].tick_params(axis='both', labelsize=24)
ax1[0,2].legend(fontsize ='xx-small',loc='upper right')
ax1[1,0].set_ylabel('u0',rotation = 0,fontweight = 'bold',fontsize = 24,labelpad = 12)
ax1[1,0].set_ylim(-3,2)
ax1[1,0].set_xlabel('time [s]',fontweight = 'bold',fontsize = 24)
ax1[1,0].set_title('u0 over time',fontweight = 'bold',fontsize = 24, pad = 12)
ax1[1,0].grid(False)
ax1[1,0].tick_params(axis='both', labelsize=24)
ax1[1,0].legend(fontsize ='xx-small',loc='upper right')
ax1[1,1].set_ylabel('u1',rotation = 0,fontweight = 'bold',fontsize = 24,labelpad = 12)
ax1[1,1].set_ylim(-1,1)
ax1[1,1].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
ax1[1,1].set_xlabel('time [s]',fontweight = 'bold',fontsize = 24)
ax1[1,1].set_title('u1 over time',fontweight = 'bold',fontsize = 24, pad = 12)
ax1[1,1].grid(False)
ax1[1,1].tick_params(axis='both', labelsize=24)
ax1[1,1].legend(fontsize ='xx-small',loc='upper right')
ax1[1,2].set_ylim(0,100)
ax1[1,2].set_ylabel('J_cl',rotation = 0,fontweight = 'bold',fontsize = 24, labelpad = 12)
ax1[1,2].set_xlabel('time [s]',fontweight = 'bold',fontsize = 24)
ax1[1,2].set_title('closed loop cost',fontweight = 'bold',fontsize = 24, pad = 12)
ax1[1,2].grid(False)
ax1[1,2].tick_params(axis='both', labelsize=24)

suptitle1=fig1.suptitle("closed_loop-behavior (alternating-constant [ns = {0}])".format(datadict_CL_run_constant['ns'][0][0]),fontweight = 'bold',fontsize = 28)
fig1.text(0.95,suptitle1.get_position()[1],"simulation-time [s]: {0}".format(round(datadict_CL_run_constant['CL'][1],3)),ha = 'center', va = 'top', color = 'black', fontsize = 12)
fig1.align_labels()
fig1.tight_layout(pad = 3)

plt.draw()
####### evaluation of the closed-loop-performance in case the number of subregions differs from partition-step to the other


fig2, ax2 = plt.subplots(2,3,figsize=(20,12))

N1 = datadict_CL_run_different['counter'][0] #number of odd prediction steps
N2 = datadict_CL_run_different['counter'][1] #number of even prediction steps
time2x = [np.linspace(i,i+N,num=N,endpoint = False)for i in range(0,N_sim)]
time2u = [np.linspace(i,i+N-1,num=N-1,endpoint = False)for i in range(0,N_sim)]

#from the OL-predictions in the CL-simulation appropriate lists of the open-loop predictions have to be constructed for plotting
#construction of appropriate lists for plotting the predictions
#if current step is even then the next x will be odd and vice versa
even_count = 0
odd_count = 0
x0_max2 = []
x0_min2 = []
x1_max2 = []
x1_min2 = []
u0_2 = []
u1_2 = []
for i in range(N_sim):
    even_count = 0 #both have to be set 0 in every simulation step
    odd_count = 0
    x0max2 = np.array(datadict_CL_run_different['OL'][1][i]['x_max_even',even_count,-1,0])
    x0min2 = np.array(datadict_CL_run_different['OL'][1][i]['x_min_even',even_count,0,0])
    x1max2 = np.array(datadict_CL_run_different['OL'][1][i]['x_max_even',even_count,-1,1])
    x1min2 = np.array(datadict_CL_run_different['OL'][1][i]['x_min_even',even_count,0,1])
    #if N1 < N2:
    u0 = np.array(datadict_CL_run_different['OL'][1][i]['u_even',even_count,0,0])
    u1 = np.array(datadict_CL_run_different['OL'][1][i]['u_even',even_count,0,1])
    #else:
        #u0 = np.array(datadict_CL_run_different['OL'][1][i]['u_even',even_count,0,0])
        #u1 = np.array(datadict_CL_run_different['OL'][1][i]['u_even',even_count,0,1])
    even_count += 1
    
    for p in range(1,N):

        if p%2 == 0:
            x0max2 = np.concatenate((copy.deepcopy(x0max2),np.array(datadict_CL_run_different['OL'][1][i]['x_max_even',even_count,-1,0])),axis = 0)
            x0min2 = np.concatenate((copy.deepcopy(x0min2),np.array(datadict_CL_run_different['OL'][1][i]['x_min_even',even_count,0,0])),axis = 0)
            x1max2 = np.concatenate((copy.deepcopy(x1max2),np.array(datadict_CL_run_different['OL'][1][i]['x_max_even',even_count,-1,1])),axis = 0)
            x1min2 = np.concatenate((copy.deepcopy(x1min2),np.array(datadict_CL_run_different['OL'][1][i]['x_min_even',even_count,0,1])),axis = 0)
            if p == N-1: #terminal condition that should be satisfied if the last prediction step is reached. No input should be applied anymore as the final state is reached and therefore a criterion is chosen so that nothing is done in that case and this step will be passed           
                continue
                #if N1 == N2:
                    #u0 = np.concatenate((copy.deepcopy(u0),np.array(datadict_CL_run_different['OL'][1][i]['u_even',even_count,0,0])),axis = 0)
                    #u1 = np.concatenate((copy.deepcopy(u1),np.array(datadict_CL_run_different['OL'][1][i]['u_even',even_count,0,1])),axis = 0) 
            else:
                u0 = np.concatenate((copy.deepcopy(u0),np.array(datadict_CL_run_different['OL'][1][i]['u_even',even_count,0,0])),axis = 0)
                u1 = np.concatenate((copy.deepcopy(u1),np.array(datadict_CL_run_different['OL'][1][i]['u_even',even_count,0,1])),axis = 0)
            even_count += 1
        elif p%2 == 1:
            x0max2 = np.concatenate((copy.deepcopy(x0max2),np.array(datadict_CL_run_different['OL'][1][i]['x_max_odd',odd_count,-1,0])),axis = 0)
            x0min2 = np.concatenate((copy.deepcopy(x0min2),np.array(datadict_CL_run_different['OL'][1][i]['x_min_odd',odd_count,0,0])),axis = 0)
            x1max2 = np.concatenate((copy.deepcopy(x1max2),np.array(datadict_CL_run_different['OL'][1][i]['x_max_odd',odd_count,-1,1])),axis = 0)
            x1min2 = np.concatenate((copy.deepcopy(x1min2),np.array(datadict_CL_run_different['OL'][1][i]['x_min_odd',odd_count,0,1])),axis = 0)           
            if p == N-1: 
                continue
                #if N1 < N2:
                    #u0 = np.concatenate((copy.deepcopy(u0),np.array(datadict_CL_run_different['OL'][1][i]['u_odd',odd_count,0,0])),axis = 0)
                    #u1 = np.concatenate((copy.deepcopy(u1),np.array(datadict_CL_run_different['OL'][1][i]['u_odd',odd_count,0,1])),axis = 0)
            else:
                u0 = np.concatenate((copy.deepcopy(u0),np.array(datadict_CL_run_different['OL'][1][i]['u_odd',odd_count,0,0])),axis = 0)
                u1 = np.concatenate((copy.deepcopy(u1),np.array(datadict_CL_run_different['OL'][1][i]['u_odd',odd_count,0,1])),axis = 0)           
            odd_count += 1

    x0_max2.append(x0max2)
    x0_min2.append(x0min2)
    x1_max2.append(x1max2)
    x1_min2.append(x1min2)
    u0_2.append(u0)
    u1_2.append(u1)
ax2[0,0].plot(datadict_CL_run_different['CL'][0]['_time'],datadict_CL_run_different['CL'][0]['_x','x'][:,0]) # plot evolution of state x0 over simulation time

for i in plot0: #plotting of x0-open loop predictions in the i-th closed-loop simulation step
    ax2[0,0].plot(time2x[i],x0max[i],color = color[0],label = "prediction: {0}".format(i),linewidth=0.5)
    ax2[0,0].plot(time2x[i],x0min[i],color = color[0],linewidth=0.5)

ax2[0,1].plot(datadict_CL_run_different['CL'][0]['_time'],datadict_CL_run_different['CL'][0]['_x','x'][:,1]) # plot evolution of state x1 over simulation time

for i in plot1: #plotting of x1-open loop predictions in the i-th closed-loop simulation step
    ax2[0,1].plot(time2x[i],x1max[i],color = color[0],label = "prediction: {0}".format(i),linewidth=0.5)
    ax2[0,1].plot(time2x[i],x1min[i],color = color[0],linewidth=0.5)

ax2[0,2].plot(datadict_CL_run_different['CL'][0]['_x','x'][:,0],datadict_CL_run_different['CL'][0]['_x','x'][:,1],label = "closed-loop") # plot evolution of state x1 over x0

for i in plot2: #plotting of x0/x1-open loop predictions (rectangles) in the i-th closed-loop simulation step
    for p in range(0,N):
        ax2[0,2].add_patch(mpl.patches.Rectangle(np.concatenate((x0_min2[i],x1_min2[i]),axis=1)[p],np.concatenate((x0_max2[i]-x0_min2[i],x1_max2[i]-x1_min2[i]),axis=1)[p][0],np.concatenate((x0_max2[i]-x0_min2[i],x1_max2[i]-x1_min2[i]),axis=1)[p][1], color = 'None', ec = color[0],linewidth=0.5 ))
        ax2[0,2].text(x0min[i][p],x1min[i][p],str(p),ha = 'right', va = 'bottom', color = 'black', fontweight = 'bold',fontsize ='xx-small')
# there is no initial rectangle starting at the initial point of the closed loop trajectory, because the first point in the prediction is always constrained to be a point and not a rectanglet(time2x,x1min[i],color = color[0],linewidth=0.5)

ax2[1,0].step(datadict_CL_run_different['CL'][0]['_time'],datadict_CL_run_different['CL'][0]['_u','u'][:,0],where = 'post')

for i in plot3: #plotting of u0-open loop predictions in the i-th closed-loop simulation step
    ax2[1,0].step(time2u[i],u0_2[i],where = 'post', color = color[0], label = "prediction: {0}".format(i),linewidth=0.5)

ax2[1,1].step(datadict_CL_run_different['CL'][0]['_time'],datadict_CL_run_different['CL'][0]['_u','u'][:,1],where = 'post')

for i in plot4: #plotting of u1-open loop predictions in the i-th closed-loop simulation step
    ax2[1,1].step(time2u[i],u1_2[i],where = 'post', color = color[0], label = "prediction: {0}".format(i),linewidth=0.5)

ax2[1,2].step(datadict_CL_run_different['CL'][0]['_time'],closed_loop_cost(datadict_CL_run_different)[0:N_sim],where = 'post')


ax2[0,0].set_ylabel('x0',rotation = 0,fontweight = 'bold',fontsize = 24,labelpad = 12)
ax2[0,0].set_xlabel('time [s]',fontweight = 'bold',fontsize = 24)
ax2[0,0].set_ylim(0,7)
ax2[0,0].set_title('x0 over time',fontweight = 'bold', fontsize = 24, pad = 12)
ax2[0,0].grid(False)
ax2[0,0].tick_params(axis='both', labelsize=24)
ax2[0,0].legend(fontsize ='xx-small',loc='upper right')
ax2[0,1].set_ylabel('x1',rotation = 0,fontweight = 'bold', fontsize = 24,labelpad = 12)
ax2[0,1].set_xlabel('time [s]',fontweight = 'bold', fontsize = 24)
ax2[0,1].set_ylim(0,4)
ax2[0,1].set_title('x1 over time',fontweight = 'bold',fontsize = 24, pad = 12 )
ax2[0,1].grid(False)
ax2[0,1].tick_params(axis='both', labelsize=24)
ax2[0,1].legend(fontsize ='xx-small',loc='upper right')
ax2[0,2].set_ylabel('x1',rotation = 0,fontweight = 'bold',fontsize = 24,labelpad = 12)
ax2[0,2].set_xlabel('x0',fontweight = 'bold',fontsize = 24)
ax2[0,2].set_ylim(0,5)
ax2[0,2].set_xlim(0,8)
ax2[0,2].set_title('x1 over x0',fontweight = 'bold',fontsize = 24, pad = 12)
ax2[0,2].grid(False)
ax2[0,2].tick_params(axis='both', labelsize=24)
ax2[0,2].legend(fontsize ='xx-small',loc='upper right')
ax2[1,0].set_ylabel('u0',rotation = 0,fontweight = 'bold',fontsize = 24,labelpad = 12)
ax2[1,0].set_ylim(-3,2)
ax2[1,0].set_xlabel('time [s]',fontweight = 'bold',fontsize = 24)
ax2[1,0].set_title('u0 over time',fontweight = 'bold',fontsize = 24, pad = 12)
ax2[1,0].grid(False)
ax2[1,0].tick_params(axis='both', labelsize=24)
ax2[1,0].legend(fontsize ='xx-small',loc='upper right')
ax2[1,1].set_ylabel('u1',rotation = 0,fontweight = 'bold',fontsize = 24,labelpad = 12)
ax2[1,1].set_ylim(-1,1)
ax2[1,1].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
ax2[1,1].set_xlabel('time [s]',fontweight = 'bold',fontsize = 24)
ax2[1,1].set_title('u1 over time',fontweight = 'bold',fontsize = 24, pad = 12)
ax2[1,1].grid(False)
ax2[1,1].tick_params(axis='both', labelsize=24)
ax2[1,1].legend(fontsize ='xx-small',loc='upper right')
ax2[1,2].set_ylim(0,100)
ax2[1,2].set_ylabel('J_cl',rotation = 0,fontweight = 'bold',fontsize = 24, labelpad = 12)
ax2[1,2].set_xlabel('time [s]',fontweight = 'bold',fontsize = 24)
ax2[1,2].set_title('closed loop cost',fontweight = 'bold',fontsize = 24, pad = 12)
ax2[1,2].grid(False)
ax2[1,2].tick_params(axis='both', labelsize=24)

suptitle2=fig2.suptitle("closed_loop-behavior (alternating-different [n1 = {0}, n2 = {1}])".format(datadict_CL_run_different['ns'][0][0],datadict_CL_run_different['ns'][0][1]),fontweight = 'bold',fontsize = 28)
fig2.text(0.95,suptitle2.get_position()[1],"simulation-time [s]: {0}".format(round(datadict_CL_run_different['CL'][1],3)),ha = 'center', va = 'top', color = 'black', fontsize = 12)
fig2.align_labels()
fig2.tight_layout(pad = 3)

####### evaluation of the closed-loop-performance in case the number of subregions remains equal in each prediction step and all dimensions are cut


fig3, ax3 = plt.subplots(2,3,figsize=(20,12))

time3x = [np.linspace(i,i+N,num=N,endpoint = False)for i in range(0,N_sim)]
time3u = [np.linspace(i,i+N-1,num=N-1,endpoint = False)for i in range(0,N_sim)]
x0max_3 = [np.array(datadict_CL_run_full['OL'][1][i]['x_max',0:N,-1,0]).reshape(N,1) for i in range(0,N_sim)] 
x0min_3 = [np.array(datadict_CL_run_full['OL'][1][i]['x_min',0:N,0,0]).reshape(N,1) for i in range(0,N_sim)]
x1max_3 = [np.array(datadict_CL_run_full['OL'][1][i]['x_max',0:N,-1,1]).reshape(N,1) for i in range(0,N_sim)]
x1min_3 = [np.array(datadict_CL_run_full['OL'][1][i]['x_min',0:N,0,1]).reshape(N,1) for i in range(0,N_sim)]
u0_3 = [np.array(datadict_CL_run_full['OL'][1][i]['u',0:N-1,0,0]).reshape(N-1,1) for i in range(0,N_sim)]
u1_3 = [np.array(datadict_CL_run_full['OL'][1][i]['u',0:N-1,0,1]).reshape(N-1,1) for i in range(0,N_sim)]

ax3[0,0].plot(datadict_CL_run_full['CL'][0]['_time'],datadict_CL_run_full['CL'][0]['_x','x'][:,0], label = "closed-loop") # plot evolution of state x0 over simulation time                                                                                                                                                    

for i in plot0: #plotting of x0-open loop predictions in the i-th closed-loop simulation step
    ax3[0,0].plot(time3x[i],x0max_3[i],color = color[0],label = "prediction: {0}".format(i),linewidth=0.5)
    ax3[0,0].plot(time3x[i],x0min_3[i],color = color[0],linewidth=0.5)

ax3[0,1].plot(datadict_CL_run_full['CL'][0]['_time'],datadict_CL_run_full['CL'][0]['_x','x'][:,1], label = "closed-loop") # plot evolution of state x1 over simulation time

for i in plot1: #plotting of x1-open loop predictions in the i-th closed-loop simulation step
    ax3[0,1].plot(time3x[i],x1max_3[i],color = color[0],label = "prediction: {0}".format(i),linewidth=0.5)
    ax3[0,1].plot(time3x[i],x1min_3[i],color = color[0],linewidth=0.5)
    
ax3[0,2].plot(datadict_CL_run_full['CL'][0]['_x','x'][:,0],datadict_CL_run_full['CL'][0]['_x','x'][:,1],label = "closed-loop") # plot evolution of state x1 over x0

for i in plot2: #plotting of x0/x1-open loop predictions (rectangles) in the i-th closed-loop simulation step
    for p in range(0,N):
        ax3[0,2].add_patch(mpl.patches.Rectangle(np.concatenate((x0min_3[i],x1min_3[i]),axis=1)[p],np.concatenate((x0max_3[i]-x0min_3[i],x1max_3[i]-x1min_3[i]),axis=1)[p][0],np.concatenate((x0max_3[i]-x0min_3[i],x1max_3[i]-x1min_3[i]),axis=1)[p][1], color = 'None', ec = color[0],linewidth=0.5 ))
        ax3[0,2].text(x0min_3[i][p],x1min_3[i][p],str(p),ha = 'right', va = 'bottom', color = 'black', fontweight = 'bold',fontsize ='xx-small')
# there is no initial rectangle starting at the initial point of the closed loop trajectory, because the first point in the prediction is always constrained to be a point and not a rectangle
    
ax3[1,0].step(datadict_CL_run_full['CL'][0]['_time'],datadict_CL_run_full['CL'][0]['_u','u'][:,0],where = 'post', label = "closed-loop")

for i in plot3: #plotting of u0-open loop predictions in the i-th closed-loop simulation step
    ax3[1,0].step(time3u[i],u0_3[i],where = 'post', color = color[0], label = "prediction: {0}".format(i),linewidth=0.5)

ax3[1,1].step(datadict_CL_run_full['CL'][0]['_time'],datadict_CL_run_full['CL'][0]['_u','u'][:,1],where = 'post', label = "closed-loop")

for i in plot4: #plotting of u1-open loop predictions in the i-th closed-loop simulation step
    ax3[1,1].step(time3u[i],u1_3[i],where = 'post', color = color[0], label = "prediction: {0}".format(i),linewidth=0.5)
#plotting of closed-loop-cost
ax3[1,2].step(datadict_CL_run_full['CL'][0]['_time'],closed_loop_cost(datadict_CL_run_full)[0:N_sim],where = 'post', label = "closed-loop cost")



ax3[0,0].set_ylabel('x0',rotation = 0,fontweight = 'bold',fontsize = 24, labelpad = 12)
ax3[0,0].set_xlabel('time [s]',fontweight = 'bold',fontsize = 24)
ax3[0,0].set_ylim(0,7)
ax3[0,0].set_title('x0 over time',fontweight = 'bold',fontsize = 24, pad = 12)
ax3[0,0].grid(False)
ax3[0,0].tick_params(axis='both', labelsize=24)
ax3[0,0].legend(fontsize ='xx-small',loc='upper right')
ax3[0,1].set_ylabel('x1',rotation = 0,fontweight = 'bold',fontsize = 24,labelpad = 12)
ax3[0,1].set_xlabel('time [s]',fontweight = 'bold',fontsize = 24)
ax3[0,1].set_ylim(0,4)
ax3[0,1].set_title('x1 over time',fontweight = 'bold',fontsize = 24, pad = 12)
ax3[0,1].grid(False)
ax3[0,1].tick_params(axis='both', labelsize=24)
ax3[0,1].legend(fontsize ='xx-small',loc='upper right')
ax3[0,2].set_ylabel('x1',rotation = 0,fontweight = 'bold',fontsize = 24, labelpad = 12)
ax3[0,2].set_xlabel('x0',fontweight = 'bold',fontsize = 24)
ax3[0,2].set_ylim(0,5)
ax3[0,2].set_xlim(0,8)
ax3[0,2].set_title('x1 over x0',fontweight = 'bold',fontsize = 24, pad = 12)
ax3[0,2].grid(False)
ax3[0,2].tick_params(axis='both', labelsize=24)
ax3[0,2].legend(fontsize ='xx-small',loc='upper right')
ax3[1,0].set_ylabel('u0',rotation = 0,fontweight = 'bold',fontsize = 24, labelpad = 12)
ax3[1,0].set_ylim(-3,2)
ax3[1,0].set_xlabel('time [s]',fontweight = 'bold',fontsize = 24)
ax3[1,0].set_title('u0 over time',fontweight = 'bold',fontsize = 24, pad = 12)
ax3[1,0].grid(False)
ax3[1,0].tick_params(axis='both', labelsize=24)
ax3[1,0].legend(fontsize ='xx-small',loc='upper right')
ax3[1,1].set_ylabel('u1',rotation = 0,fontweight = 'bold',fontsize = 24,labelpad = 12)
ax3[1,1].set_ylim(-1,1)
ax3[1,1].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
ax3[1,1].set_xlabel('time [s]',fontweight = 'bold',fontsize = 24)
ax3[1,1].set_title('u1 over time',fontweight = 'bold',fontsize = 24, pad = 12)
ax3[1,1].grid(False)
ax3[1,1].tick_params(axis='both', labelsize=24)
ax3[1,1].legend(fontsize ='xx-small',loc='upper right')
ax3[1,2].set_ylabel('J_cl',rotation = 0,fontweight = 'bold',fontsize = 24,labelpad = 12)
ax3[1,2].set_ylim(0,100)
ax3[1,2].set_xlabel('time [s]',fontweight = 'bold',fontsize = 24)
ax3[1,2].set_title('closed loop cost',fontweight = 'bold',fontsize = 24, pad = 12)
ax3[1,2].grid(False)
ax3[1,2].tick_params(axis='both', labelsize=24)

suptitle3=fig3.suptitle("closed_loop-behavior (full-partitioning [ns = {0}])".format(datadict_CL_run_full['ns'][0][0]),fontweight = 'bold',fontsize = 28)
fig3.text(0.95,suptitle3.get_position()[1],"simulation-time [s]: {0}".format(round(datadict_CL_run_full['CL'][1],3)),ha = 'center', va = 'top', color = 'black', fontsize = 12)
fig3.align_labels()
fig3.tight_layout(pad = 3)










