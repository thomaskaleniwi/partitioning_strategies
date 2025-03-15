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

# constants
dt = 1 # timestep
p0 = 0.15 # initial parameter should be between 0.00 and 0.30
nx = 2 # number of states
nu = 2 # number of inputs
nd = 1 # number of parameters
# Set up do-mpc-model

model_type = 'discrete'
model = do_mpc.model.Model(model_type)


# Set model states, inputs and parameter(s)

x = model.set_variable(var_type='_x', var_name='x', shape=(nx,1))

u = model.set_variable(var_type='_u', var_name='u',shape=(nu,1))

p = model.set_variable(var_type='_tvp', var_name='p',shape=(nd,1))

# Set right-hand-side for ODE for all introduced states 

A = np.array([[1,1],[0,1]])
B = np.array([[1,0],
              [0,1]])
F = vertcat(p,p)

x_next = A@x + B@u + F@sqrt(x.T@x)

model.set_rhs('x',x_next)

# Setup model

model.setup()

# Get x_next as Casadi-Function

system = Function('system',[model.x,model.u,model.tvp],[model._rhs['x']])

# Specify bounds on states and inputs


# state constraints 

lb_x = 0 * np.ones((nx,1))
ub_x = 10 * np.ones((nx,1))

# input constraints

lb_u = np.array([[-10],[-5]]) 
ub_u = np.array([[10], [5]]) 


# Create the simulator

simulator = do_mpc.simulator.Simulator(model)
simulator.set_param(t_step = dt)

p_template = simulator.get_tvp_template()


# uncertainty is specified and different functions for different uncertainty scenarios are created

p_var = 1
pnormal = p0
p_max = (1+p_var)*p0
p_min = (1-p_var)*p0

def p_fun_wcmax(t):
    p_template['p'] = p0*(1+p_var)
    return p_template

def p_fun_wcmin(t):
    p_template['p'] = p0*(1-p_var)
    return p_template

def p_fun_0(t):
    p_template['p'] = p0
    return p_template

def interpol(a,b,fac):
    return a*fac+b*(1-fac)

def p_fun_var(t, in_seed,const=True):
    
    if const:
        np.random.seed(int(in_seed))
        rand = np.random.uniform(0,1,size = None);
        p_template['p'] = interpol(p0*(1+p_var),p0*(1-p_var),rand)
    else:
        np.random.seed(int(t//dt+in_seed))
        rand=np.random.uniform(0,1,size = None);
        p_template['p'] = interpol(p0*(1+p_var),k1_0*(1-p_var),rand)
    return p_template

simulator.set_tvp_fun(p_fun_0)
simulator.setup()

# pass the ordering, the number of partitions in the different dimensions     
# seed defines the corresponding uncertainty case
# x00 = number of partitions in the first dimension(0) for the odd prediction step -> alternating such that in the second step the second dimension(1) will be partitioned like this
# x10 = number of partitions in the second dimension(1) for the odd prediction step -> alternating such that in the second step the first dimension(0) will be partitioned like this
# x01 = number of partitions in the first dimension(0) for the even prediction step -> alternating such that in the second step the second dimension(1) will be partitioned like this
# x11 = number of partitions in the second dimension(1) for the even prediction step -> alternating such that in the second step the first dimension(0) will be partitioned like this
# ord1 = define which dimension to be partitioned according to the ordering in the odd prediction step
# ord2 = define which dimension to be partioned according to the ordering in the even prediction step    
def closed_loop_comp(seed,x00,x10,x01,x11,ord1,ord2):
    
    N = 30 #chosen same as in paper "Determinstic safety guarantees for learning-based control of monotone systems"
    N_RCIS = 2
    x_sp = np.array([[5],[2]])
    Q = np.array([[1,0],
                  [0,1]])
    R = np.array([[2,0],
                  [0,1]])
    u_bef = SX.sym('u_bef',nu, 1)
    x_ref = SX.sym('x_ref',nx, 1)
    
    stage_cost = (model.x['x']-x_ref).T@Q@(model.x['x']-x_ref) + (model.u['u']-u_bef).T@R@(model.u['u']-u_bef) 
    terminal_cost = 10*(model.x['x']-x_ref).T@Q@(model.x['x']-x_ref)

    stage_cost_fcn = Function('stage_cost',[model.x,model.u,u_bef,x_ref], [stage_cost])
    terminal_cost_fcn = Function('terminal_cost',[model.x,x_ref], [terminal_cost])

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
    
    cut1 = np.zeros((nx,1)) #-> odd case
    cut1[0] = x00 # cuts in the first dimension (0)
    cut1[1] = x10 # cuts in the second dimension (1)
    
    cut2 = np.zeros((nx,1)) #-> even case
    cut2[0] = x01 # cuts in the first dimension (0)
    cut2[1] = x11 # cuts in the second dimension (1)
    
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
# scenario 1 (dim 0 is cut firstly in the first odd step)
# depending on which count is higher the corresponding u will have to be adapted   
    if N1 == N2:
        opt_x1 = struct_symSX([
            entry('x_min_odd', shape=nx, repeat=[N1,n1]),
            entry('x_max_odd', shape=nx, repeat=[N1,n1]),
            entry('u_odd', shape=nu, repeat=[N1-1,n1]),
            entry('x_min_even', shape=nx, repeat=[N2,n2]),
            entry('x_max_even', shape=nx, repeat=[N2,n2]),
            entry('u_even', shape=nu, repeat=[N2,n2]),
            ])
        
       
    elif N1 < N2:
        opt_x1 = struct_symSX([
            entry('x_min_odd', shape=nx, repeat=[N1,n1]),
            entry('x_max_odd', shape=nx, repeat=[N1,n1]),
            entry('u_odd', shape=nu, repeat=[N1,n1]),
            entry('x_min_even', shape=nx, repeat=[N2,n2]),
            entry('x_max_even', shape=nx, repeat=[N2,n2]),
            entry('u_even', shape=nu, repeat=[N2-1,n2]),
            ])

# scenario 2 (dim 1 is cut firstly in the first odd step)
# depending on which count is higher the corresponding u will have to be adapted
    if N1 == N2:
        opt_x2 = struct_symSX([
            entry('x_min_odd', shape=nx, repeat=[N1,n2]),
            entry('x_max_odd', shape=nx, repeat=[N1,n2]),
            entry('u_odd', shape=nu, repeat=[N1-1,n2]),
            entry('x_min_even', shape=nx, repeat=[N2,n1]),
            entry('x_max_even', shape=nx, repeat=[N2,n1]),
            entry('u_even', shape=nu, repeat=[N2,n1]),
            ])
        
       
    elif N1 < N2:
        opt_x2 = struct_symSX([
            entry('x_min_odd', shape=nx, repeat=[N1,n2]),
            entry('x_max_odd', shape=nx, repeat=[N1,n2]),
            entry('u_odd', shape=nu, repeat=[N1,n2]),
            entry('x_min_even', shape=nx, repeat=[N2,n1]),
            entry('x_max_even', shape=nx, repeat=[N2,n1]),
            entry('u_even', shape=nu, repeat=[N2-1,n1]),
            ])    
        
# the corresponding 2-step RCIS must be in the right shape but no case distinction is necessary here           
    x_RCIS = struct_symSX([
       entry('x_min_odd', shape=nx ,repeat=[1,n1]),
       entry('x_max_odd', shape=nx ,repeat=[1,n1]),
       entry('u_odd', shape=nu, repeat=[1,n1]),
       entry('x_min_even', shape=nx ,repeat=[1,n2]),
       entry('x_max_even', shape=nx ,repeat=[1,n2]),
       entry('u_even', shape=nu, repeat=[1,n2])
       ])

# set the bounds on opt_x  and x_RCIS   
    lb_opt_x1 = opt_x1(0)
    ub_opt_x1 = opt_x1(np.inf)

    lb_opt_x1['x_min_odd'] = lb_x
    ub_opt_x1['x_min_odd'] = ub_x
    lb_opt_x1['x_max_odd'] = lb_x
    ub_opt_x1['x_max_odd'] = ub_x
    
    lb_opt_x1['x_min_even'] = lb_x
    ub_opt_x1['x_min_even'] = ub_x
    lb_opt_x1['x_max_even'] = lb_x
    ub_opt_x1['x_max_even'] = ub_x

    lb_opt_x1['u_odd'] = lb_u
    ub_opt_x1['u_odd'] = ub_u
    
    lb_opt_x1['u_even'] = lb_u
    ub_opt_x1['u_even'] = ub_u

###############################################################################    
    
    lb_opt_x2 = opt_x2(0)
    ub_opt_x2 = opt_x2(np.inf)

    lb_opt_x2['x_min_odd'] = lb_x
    ub_opt_x2['x_min_odd'] = ub_x
    lb_opt_x2['x_max_odd'] = lb_x
    ub_opt_x2['x_max_odd'] = ub_x
    
    lb_opt_x2['x_min_even'] = lb_x
    ub_opt_x2['x_min_even'] = ub_x
    lb_opt_x2['x_max_even'] = lb_x
    ub_opt_x2['x_max_even'] = ub_x

    lb_opt_x2['u_odd'] = lb_u
    ub_opt_x2['u_odd'] = ub_u
    
    lb_opt_x2['u_even'] = lb_u
    ub_opt_x2['u_even'] = ub_u

###############################################################################

    lb_opt_x_RCIS = x_RCIS(0)
    ub_opt_x_RCIS = x_RCIS(np.inf)

    lb_opt_x_RCIS['x_min_odd'] = lb_x
    lb_opt_x_RCIS['x_max_odd'] = lb_x
    ub_opt_x_RCIS['x_min_odd'] = ub_x
    ub_opt_x_RCIS['x_max_odd'] = ub_x
    
    lb_opt_x_RCIS['x_min_even'] = lb_x
    lb_opt_x_RCIS['x_max_even'] = lb_x
    ub_opt_x_RCIS['x_min_even'] = ub_x
    ub_opt_x_RCIS['x_max_even'] = ub_x

    lb_opt_x_RCIS['u_odd'] = lb_u
    ub_opt_x_RCIS['u_odd'] = ub_u
    
    lb_opt_x_RCIS['u_even'] = lb_u
    ub_opt_x_RCIS['u_even'] = ub_u
# define the constraint functions
# constraint_function1 is defined for partitioning the odd prediction steps 
# constraint_function2 is defined for partitioning the even prediction steps

    def constraint_function1(l,ord_dim,opt_x,i,h,lbg,ubg):
        for k in range(len(l)):
            idx=flatten(l[k])
            dim=ord_dim[-depth(l)]
            for s in idx:
                if s==idx[0] and k==0:
                    h.append(opt_x['x_min_odd',i,s,dim]-opt_x['x_min_odd',i,0,dim])
                    lbg.append(0)
                    ubg.append(0)
                else:
                    h.append(opt_x['x_min_odd',i,s,dim]-opt_x['x_min_odd',i,idx[0],dim])
                    lbg.append(0)
                    ubg.append(0)
                if s==idx[-1] and k==len(l)-1:
                    h.append(opt_x['x_max_odd',i,s,dim]-opt_x['x_max_odd',i,-1,dim])
                    lbg.append(0)
                    ubg.append(0)
                else:
                    h.append(opt_x['x_max_odd',i,s,dim]-opt_x['x_max_odd',i,idx[-1],dim])
                    lbg.append(0)
                    ubg.append(0)
            if k>=1:
                prev_last=flatten(l[k-1])[-1]
                h.append(opt_x['x_min_odd',i,idx[0],dim]-opt_x['x_max_odd',i,prev_last,dim])
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
                    h.append(opt_x['x_min_even',i,s,dim]-opt_x['x_min_even',i,0,dim])
                    lbg.append(0)
                    ubg.append(0)
                else:
                    h.append(opt_x['x_min_even',i,s,dim]-opt_x['x_min_even',i,idx[0],dim])
                    lbg.append(0)
                    ubg.append(0)
                if s==idx[-1] and k==len(l)-1:
                    h.append(opt_x['x_max_even',i,s,dim]-opt_x['x_max_even',i,-1,dim])
                    lbg.append(0)
                    ubg.append(0)
                else:
                    h.append(opt_x['x_max_even',i,s,dim]-opt_x['x_max_even',i,idx[-1],dim])
                    lbg.append(0)
                    ubg.append(0)
            if k>=1:
                prev_last=flatten(l[k-1])[-1]
                h.append(opt_x['x_min_even',i,idx[0],dim]-opt_x['x_max_even',i,prev_last,dim])
                lbg.append(0)
                ubg.append(0)
            if depth(l) >1:
                h,lbg,ubg=constraint_function2(l[k],ord_dim,opt_x,i,h,lbg,ubg)
        
        return h,lbg,ubg

# Set up the objective and the constraints of the problem
# in contrast to the previous partitioning approach, here we have a different number of subregions in each step, as the number of cuts also varies in each timestep
# therefore the cost function has to be adjusted according to if the 0th or the first dimension will be cut firstly in the first odd step (where the cutting starts)
# of course the initial constraints, the systems equation (inequality constraints), the partitioning constraints and the terminal cost will be adjusted accordingly too
# in the end we create two solvers for the two scenarios where the 0th dimension or in the other scenario the 1 dimension is cut firstly and pass J1,g1 and J2,g2 so that we solve two different optimization problems in each timestep
# doing so a way to cut in alternating dimensions is found and in addition also a different amount of subregions can be created 

# cost and constraints for scenario 1 (dim 0 is cut firstly in the first odd step)
    J1 = 0 # cost fct for normal prediction horizon N
    g1 = [] #constraint expression
    lb_g1 = [] #lower bound for constraint expression g1
    ub_g1 = [] #upper bound for constraint expression g1

# cost and constraints for scenario 2 (dim 1 is cut firstly in the first odd step)
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
    x_ref = SX.sym('x_ref', nx,1)
    x_RCIS_plus = SX.sym('x_RCIS_plus', nx, 1)
    x_RCIS_minus = SX.sym('x_RCIS_minus', nx, 1)
    
######################### Scenario 1 (dim 0 is cut in the first odd step)   
# Set initial constraints
# start with even case because indexing starts at 0    
    for s in range(n2):
        g1.append(opt_x1['x_min_even',0,s]-x_init)
        g1.append(opt_x1['x_max_even',0,s]-x_init)
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
        if i%2 == 0: #even-case -> cutting 1 dimension
            for s in range(n2):
                if i==0:
                    J1 += stage_cost_fcn(opt_x1['x_max_even', even_count,s], opt_x1['u_even', even_count,s],u_bef,x_ref)
                    J1 += stage_cost_fcn(opt_x1['x_min_even', even_count,s], opt_x1['u_even', even_count,s],u_bef,x_ref)
                else:                
                    J1 += stage_cost_fcn(opt_x1['x_max_even', even_count,s], opt_x1['u_even', even_count,s],sum(opt_x1['u_even',even_count-1,:])/len(opt_x1['u_even',even_count-1,:]),x_ref) #refers to the previous even step as number of subrergions in even and odds steps is not the same
                    J1 += stage_cost_fcn(opt_x1['x_min_even', even_count,s], opt_x1['u_even', even_count,s],sum(opt_x1['u_even',even_count-1,:])/len(opt_x1['u_even',even_count-1,:]),x_ref) #refers to the previous even step as number of subrergions in even and odds steps is not the same
                
        # inequality constraints (system equation)
                x_next_max = system(opt_x1['x_max_even',even_count,s],opt_x1['u_even',even_count,s],p_max)
                x_next_min = system(opt_x1['x_min_even',even_count,s],opt_x1['u_even',even_count,s],p_min)
                
                g1.append(opt_x1['x_max_odd', odd_count,-1]-x_next_max)
                g1.append(x_next_min - opt_x1['x_min_odd', odd_count,0])
                lb_g1.append(np.zeros((2*nx,1)))
                ub_g1.append(np.ones((2*nx,1))*inf)

            
            even_count += 1
        elif i%2 == 1: #odd-case -> cutting 0 dimension 
            for s in range(n1):
                if odd_count == 0:
                    J1 += stage_cost_fcn(opt_x1['x_max_odd', odd_count,s], opt_x1['u_odd', odd_count,s],u_bef,x_ref) 
                    J1 += stage_cost_fcn(opt_x1['x_min_odd', odd_count,s], opt_x1['u_odd', odd_count,s],u_bef,x_ref) 
                else:
                    J1 += stage_cost_fcn(opt_x1['x_max_odd', odd_count,s], opt_x1['u_odd', odd_count,s],sum(opt_x1['u_odd',odd_count-1,:])/len(opt_x1['u_odd', odd_count-1,:]),x_ref) #refers to the previous odd step as number of subrergions in even and odds steps is not the same
                    J1 += stage_cost_fcn(opt_x1['x_min_odd', odd_count,s], opt_x1['u_odd', odd_count,s],sum(opt_x1['u_odd',odd_count-1,:])/len(opt_x1['u_odd', odd_count-1,:]),x_ref) #refers to the previous odd step as number of subrergions in even and odds steps is not the same
        
        # inequality constraints (system equation)
          
                x_next_max = system(opt_x1['x_max_odd',odd_count,s],opt_x1['u_odd',odd_count,s],p_max)
                x_next_min = system(opt_x1['x_min_odd',odd_count,s],opt_x1['u_odd',odd_count,s],p_min)
                
                g1.append( opt_x1['x_max_even', even_count,-1]-x_next_max)
                g1.append(x_next_min - opt_x1['x_min_even', even_count,0])
                lb_g1.append(np.zeros((2*nx,1)))
                ub_g1.append(np.ones((2*nx,1))*inf)

            odd_count += 1

# terminal cost                
# case distinction depending on which step will be the last
    
    if N1 == N2:
        for s in range(n1):
            J1 += terminal_cost_fcn(opt_x1['x_max_odd',-1, s],x_ref)
            J1 += terminal_cost_fcn(opt_x1['x_min_odd',-1, s],x_ref)
    elif N1 < N2:
        for s in range(n2):
            J1 += terminal_cost_fcn(opt_x1['x_max_even',-1, s],x_ref)
            J1 += terminal_cost_fcn(opt_x1['x_min_even',-1, s],x_ref)

# cutting1 (first cut the 0th dimension in the first odd prediction step) 
# now we cut in alternating dimensions in different time steps
# therefore we pass a different list "ordering dimension" at each time-step        
    
    even_count1 = 1 # even_count1 = 0 would mean that first set (rather element since it is a distinct initial point) of the prediction horizon (x_max_even,0) would be partitioned/cut too.
    odd_count1 = 0 # Therefore we start indexing the even steps at 1 and after the loop we can subtract the 1 in order to get the correct number of even cutting steps 
    for i in range(1,N+1): # at i = 0 there should be no cutting since we have a distinct initial point
        # if i is even, dim 1 is cut and lis2 and ordering2 are passed
        # if i is odd, dim 0 is cut and lis1 and ordering1 are passed
        if i % 2 == 0:
            g1, lb_g1, ub_g1 = constraint_function2(lis2,ordering2,opt_x1,even_count1,g1,lb_g1,ub_g1)
        
            for s in range(n2):
                g1.append(opt_x1['x_max_even',even_count1,s]-opt_x1['x_min_even',even_count1,0])
                g1.append(opt_x1['x_max_even',even_count1,-1]-opt_x1['x_min_even',even_count1,s])
                g1.append(opt_x1['x_min_even',even_count1,s]-opt_x1['x_min_even',even_count1,0])
                g1.append(opt_x1['x_max_even',even_count1,-1]-opt_x1['x_max_even',even_count1,s])
                lb_g1.append(np.zeros((4*nx,1)))
                ub_g1.append(np.ones((4*nx,1))*inf)
            
            even_count1 += 1
        
        elif i % 2 == 1:
            g1, lb_g1, ub_g1 = constraint_function1(lis1,ordering1,opt_x1,odd_count1,g1,lb_g1,ub_g1)
            
        
            for s in range(n1):
                g1.append(opt_x1['x_max_odd',odd_count1,s]-opt_x1['x_min_odd',odd_count1,0])
                g1.append(opt_x1['x_max_odd',odd_count1,-1]-opt_x1['x_min_odd',odd_count1,s])
                g1.append(opt_x1['x_min_odd',odd_count1,s]-opt_x1['x_min_odd',odd_count1,0])
                g1.append(opt_x1['x_max_odd',odd_count1,-1]-opt_x1['x_max_odd',odd_count1,s])
                lb_g1.append(np.zeros((4*nx,1)))
                ub_g1.append(np.ones((4*nx,1))*inf)
            
            odd_count1 += 1    
    
    even_count1 -= 1
######################### Scenario 2 (dim 1 is cut in the first odd step)

# Set initial constraints
# sctart with even case because indexing starts at 0    
    for s in range(n1):
        g2.append(opt_x2['x_min_even',0,s]-x_init)
        g2.append(opt_x2['x_max_even',0,s]-x_init)
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
        if i%2 == 0: #even-case -> cutting 0th dimension
            for s in range(n1):
                if i==0:
                    J2 += stage_cost_fcn(opt_x2['x_max_even', even_count,s], opt_x2['u_even', even_count,s],u_bef,x_ref)
                    J2 += stage_cost_fcn(opt_x2['x_min_even', even_count,s], opt_x2['u_even', even_count,s],u_bef,x_ref)
                else:             
                    J2 += stage_cost_fcn(opt_x2['x_max_even', even_count,s], opt_x2['u_even', even_count,s],sum(opt_x2['u_even',even_count-1,:])/len(opt_x2['u_even',even_count-1,:]),x_ref) #refers to the previous even step  
                    J2 += stage_cost_fcn(opt_x2['x_min_even', even_count,s], opt_x2['u_even', even_count,s],sum(opt_x2['u_even',even_count-1,:])/len(opt_x2['u_even',even_count-1,:]),x_ref) #refers to the previous even step 
                
        # inequality constraints (system equation)
                x_next_max = system(opt_x2['x_max_even',even_count,s],opt_x2['u_even',even_count,s],p_max)
                x_next_min = system(opt_x2['x_min_even',even_count,s],opt_x2['u_even',even_count,s],p_min)
                
                g2.append(opt_x2['x_max_odd', odd_count,-1]-x_next_max)
                g2.append(x_next_min - opt_x2['x_min_odd', odd_count,0])
                lb_g2.append(np.zeros((2*nx,1)))
                ub_g2.append(np.ones((2*nx,1))*inf)

            
            even_count += 1
        elif i%2 == 1: #odd-case -> cutting 1 dimension 
            for s in range(n2):
                if odd_count == 0:
                    J2 += stage_cost_fcn(opt_x2['x_max_odd', odd_count,s], opt_x2['u_odd', odd_count,s],u_bef,x_ref)  
                    J2 += stage_cost_fcn(opt_x2['x_min_odd', odd_count,s], opt_x2['u_odd', odd_count,s],u_bef,x_ref)
                else:
                    J2 += stage_cost_fcn(opt_x2['x_max_odd', odd_count,s], opt_x2['u_odd', odd_count,s],sum(opt_x2['u_odd',odd_count-1,:])/len(opt_x2['u_odd', odd_count-1,:]),x_ref) #refers to the previous odd step  
                    J2 += stage_cost_fcn(opt_x2['x_min_odd', odd_count,s], opt_x2['u_odd', odd_count,s],sum(opt_x2['u_odd',odd_count-1,:])/len(opt_x2['u_odd', odd_count-1,:]),x_ref) #refers to the previous odd step  
        
        # inequality constraints (system equation)
          
                x_next_max = system(opt_x2['x_max_odd',odd_count,s],opt_x2['u_odd',odd_count,s],p_max)
                x_next_min = system(opt_x2['x_min_odd',odd_count,s],opt_x2['u_odd',odd_count,s],p_min)
                
                g2.append( opt_x2['x_max_even', even_count,-1]-x_next_max)
                g2.append(x_next_min - opt_x2['x_min_even', even_count,0])
                lb_g2.append(np.zeros((2*nx,1)))
                ub_g2.append(np.ones((2*nx,1))*inf)

            odd_count += 1
# terminal cost                
# case distinction depending on which step will be the last
    if N1 == N2:
        for s in range(n2):
            J2 += terminal_cost_fcn(opt_x2['x_max_odd',-1, s],x_ref)
            J2 += terminal_cost_fcn(opt_x2['x_min_odd',-1, s],x_ref)
    elif N1 < N2:
        for s in range(n1):
            J2 += terminal_cost_fcn(opt_x2['x_max_even',-1, s],x_ref)
            J2 += terminal_cost_fcn(opt_x2['x_min_even',-1, s],x_ref)

# cutting2 (first cut the 1 dimension in the first odd prediction step) 
# now we cut in alternating dimensions in different time steps
# therefore we pass a different list "ordering dimension" at each time-step
    even_count2 = 1  # even_count1 = 0 would mean that first set (rather element since it is a distinct initial point) of the prediction horizon (x_max_even,0) would be partitioned/cut too.
    odd_count2 = 0 # Therefore we start indexing the even steps at 1 and after the loop we can subtract the 1 in order to get the correct number of even cutting steps

    for i in range(1,N+1):
        # if i is even, dim 0 is cut and lis1 and ordering1 are passed
        # if i is odd, dim 1 is cut and lis2 and ordering2 are passed
        
        if i % 2 == 0:
            g2, lb_g2, ub_g2 = constraint_function2(lis1,ordering1,opt_x2,even_count2,g2,lb_g2,ub_g2) 
            
            for s in range(n1):
                g2.append(opt_x2['x_max_even',even_count2,s]-opt_x2['x_min_even',even_count2,0])
                g2.append(opt_x2['x_max_even',even_count2,-1]-opt_x2['x_min_even',even_count2,s])
                g2.append(opt_x2['x_min_even',even_count2,s]-opt_x2['x_min_even',even_count2,0])
                g2.append(opt_x2['x_max_even',even_count2,-1]-opt_x2['x_max_even',even_count2,s])
                lb_g2.append(np.zeros((4*nx,1)))
                ub_g2.append(np.ones((4*nx,1))*inf)
            
            even_count2 += 1
        
        elif i % 2 == 1:
            g2, lb_g2, ub_g2 = constraint_function1(lis2,ordering2,opt_x2,odd_count2,g2,lb_g2,ub_g2)  
            
            for s in range(n2):
                g2.append(opt_x2['x_max_odd',odd_count2,s]-opt_x2['x_min_odd',odd_count2,0])
                g2.append(opt_x2['x_max_odd',odd_count2,-1]-opt_x2['x_min_odd',odd_count2,s])
                g2.append(opt_x2['x_min_odd',odd_count2,s]-opt_x2['x_min_odd',odd_count2,0])
                g2.append(opt_x2['x_max_odd',odd_count2,-1]-opt_x2['x_max_odd',odd_count2,s])
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
        if i%2 == 0: #-> even case (cut dim 1)
            for s in range(n2):
                x_next_plus_RCIS = system(x_RCIS['x_max_even',even_count_RCIS,s], x_RCIS['u_even',even_count_RCIS,s],p_max)
                x_next_minus_RCIS = system(x_RCIS['x_min_even',even_count_RCIS,s], x_RCIS['u_even',even_count_RCIS,s],p_min)
                if i == N_RCIS-1: #constrain the propagation of the last hyperrectangle to lie again in the first hyperrectangle
                    g_RCIS.append(x_RCIS['x_max_even',0,-1] - x_next_plus_RCIS)
                    g_RCIS.append(x_next_minus_RCIS - x_RCIS['x_min_even',0, 0])
                else:
                    g_RCIS.append(x_RCIS['x_max_odd',odd_count_RCIS,-1] - x_next_plus_RCIS)
                    g_RCIS.append(x_next_minus_RCIS - x_RCIS['x_min_odd',odd_count_RCIS, 0])

                lb_g_RCIS.append(np.zeros((2*nx,1)))
                ub_g_RCIS.append(inf*np.ones((2*nx,1)))
            even_count_RCIS += 1
        if i%2 == 1: #-> odd case (cut dim 0)
            for s in range(n1):
                x_next_plus_RCIS = system(x_RCIS['x_max_odd',odd_count_RCIS,s], x_RCIS['u_odd',odd_count_RCIS,s],p_max)
                x_next_minus_RCIS = system(x_RCIS['x_min_odd',odd_count_RCIS,s], x_RCIS['u_odd',odd_count_RCIS,s],p_min)
                if i == N_RCIS-1: #constrain the propagation of the last hyperrectangle to lie again in the first hyperrectangle
                    g_RCIS.append(x_RCIS['x_max_even',0,-1] - x_next_plus_RCIS)
                    g_RCIS.append(x_next_minus_RCIS - x_RCIS['x_min_even',0, 0])
                else:
                    g_RCIS.append(x_RCIS['x_max_even',even_count_RCIS ,-1] - x_next_plus_RCIS)
                    g_RCIS.append(x_next_minus_RCIS - x_RCIS['x_min_even',even_count_RCIS , 0])
                lb_g_RCIS.append(np.zeros((2*nx,1)))
                ub_g_RCIS.append(inf*np.ones((2*nx,1)))
            odd_count_RCIS += 1
                
        
        
        # Cutting RCIS
        # cut in alternating dimensions depending on whether the prediction step is even or odd
        if i % 2 == 0: # cut the 1 dimension
            g_RCIS,lb_g_RCIS,ub_g_RCIS = constraint_function2(lis2,ordering2,x_RCIS,even_count_RCIS-1,g_RCIS,lb_g_RCIS,ub_g_RCIS);
            for s in range(n2):
                g_RCIS.append(x_RCIS['x_max_even',even_count_RCIS-1,s]-x_RCIS['x_min_even',even_count_RCIS-1,0])
                g_RCIS.append(x_RCIS['x_max_even',even_count_RCIS-1,-1]-x_RCIS['x_min_even',even_count_RCIS-1,s])
                g_RCIS.append(x_RCIS['x_min_even',even_count_RCIS-1,s]-x_RCIS['x_min_even',even_count_RCIS-1,0])
                g_RCIS.append(x_RCIS['x_max_even',even_count_RCIS-1,-1]-x_RCIS['x_max_even',even_count_RCIS-1,s])
                lb_g_RCIS.append(np.zeros((4*nx,1)))
                ub_g_RCIS.append(np.ones((4*nx,1))*inf)
        elif i % 2 == 1: # cut the 0th dimension
            g_RCIS,lb_g_RCIS,ub_g_RCIS = constraint_function1(lis1,ordering1,x_RCIS,odd_count_RCIS-1,g_RCIS,lb_g_RCIS,ub_g_RCIS);
            for s in range(n1):
                g_RCIS.append(x_RCIS['x_max_odd',odd_count_RCIS-1,s]-x_RCIS['x_min_odd',odd_count_RCIS-1,0])
                g_RCIS.append(x_RCIS['x_max_odd',odd_count_RCIS-1,-1]-x_RCIS['x_min_odd',odd_count_RCIS-1,s])
                g_RCIS.append(x_RCIS['x_min_odd',odd_count_RCIS-1,s]-x_RCIS['x_min_odd',odd_count_RCIS-1,0])
                g_RCIS.append(x_RCIS['x_max_odd',odd_count_RCIS-1,-1]-x_RCIS['x_max_odd',odd_count_RCIS-1,s])
                lb_g_RCIS.append(np.zeros((4*nx,1)))
                ub_g_RCIS.append(np.ones((4*nx,1))*inf)
            
    J_RCIS = -1
    even = 0
    odd = 0
    for i in range(N_RCIS):
        J_mini = -1
        if i % 2 == 0:
            for ix in range(nx):
                J_mini = J_mini*(x_RCIS['x_max_even',even,-1,ix]-x_RCIS['x_min_even',even,0,ix])
            even += 1
        elif i % 2 == 1:
            for ix in range(nx):
                J_mini = J_mini*(x_RCIS['x_max_odd',odd,-1,ix]-x_RCIS['x_min_odd',odd,0,ix])
            odd += 1
        
        J_RCIS += J_mini
        
    g_RCIS = vertcat(*g_RCIS)
    lb_g_RCIS = vertcat(*lb_g_RCIS)
    ub_g_RCIS = vertcat(*ub_g_RCIS)
    x_rcis = time.time()
    prob = {'f':J_RCIS,'x':vertcat(x_RCIS),'g':g_RCIS, 'p':vertcat(x_ref,p_plus,p_minus)}
    solver_mx_inv_set = nlpsol('solver','ipopt',prob)
   
    # now solve the optimization problem 

    x_set = np.array([[0.1,0.1]]).T
    opt_ro_initial = x_RCIS(0)
    opt_ro_initial['x_min_even'] = x_set
    opt_ro_initial['x_max_even'] = x_set
    opt_ro_initial['x_min_odd'] = x_set
    opt_ro_initial['x_max_odd'] = x_set
    results = solver_mx_inv_set(p=vertcat(x_set,p_max,p_min),x0=opt_ro_initial, lbg=lb_g_RCIS,ubg=ub_g_RCIS,lbx=lb_opt_x_RCIS,ubx=ub_opt_x_RCIS);
    y_rcis = time.time()
    print("time to compute the 2-step RCIS:" , y_rcis-x_rcis)
    # transform the optimizer results into a structured symbolic variable
    res = x_RCIS(results['x']);

    # plot the simulation results (RCIS for each timestep and the total RCIS)
    fig, ax =plt.subplots(1,N_RCIS+1,layout = 'constrained',figsize = (20,9))
    even = 0
    odd = 0
    for i in range(N_RCIS):
        if i % 2 == 0:
            for s in range(n2):
                ax[i+1].add_patch(mpl.patches.Rectangle(np.array(res['x_min_even',even,s]), np.array(res['x_max_even',even,s]-res['x_min_even',even,s])[0][0] , np.array(res['x_max_even',even,s]-res['x_min_even',even,s])[1][0], color="None",ec='red'))
                ax[i+1].text(np.array(res['x_min_even',even,s,0]+0.5*(res['x_max_even',even,s,0]-res['x_min_even',even,s,0])),np.array(res['x_min_even',even,s,1]+0.5*(res['x_max_even',even,s,1]-res['x_min_even',even,s,1])),str(s),ha = 'center', va = 'center', color = 'black', fontweight = 'bold', fontsize ='xx-small')
                ax[N_RCIS].add_patch(mpl.patches.Rectangle(np.array(res['x_min_even',even,s]), np.array(res['x_max_even',even,s]-res['x_min_even',even,s])[0][0] , np.array(res['x_max_even',even,s]-res['x_min_even',even,s])[1][0], color="None",ec='red'))
            ax[i+1].set_title("partitioning in x1",fontweight = 'bold',fontsize=24,pad = 12)
            even += 1
        elif i % 2 == 1:
            for s in range(n1):
                ax[i-1].add_patch(mpl.patches.Rectangle(np.array(res['x_min_odd',odd,s]), np.array(res['x_max_odd',odd,s]-res['x_min_odd',odd,s])[0][0] , np.array(res['x_max_odd',odd,s]-res['x_min_odd',odd,s])[1][0], color="None",ec='red'))
                ax[i-1].text(np.array(res['x_min_odd',odd,s,0]+0.5*(res['x_max_odd',odd,s,0]-res['x_min_odd',odd,s,0])),np.array(res['x_min_odd',odd,s,1]+0.5*(res['x_max_odd',odd,s,1]-res['x_min_odd',odd,s,1])),str(s),ha = 'center', va = 'center', color = 'black', fontweight = 'bold', fontsize ='xx-small')
                ax[N_RCIS].add_patch(mpl.patches.Rectangle(np.array(res['x_min_odd',odd,s]), np.array(res['x_max_odd',odd,s]-res['x_min_odd',odd,s])[0][0] , np.array(res['x_max_odd',odd,s]-res['x_min_odd',odd,s])[1][0], color="None",ec='red'))
            ax[i-1].set_title("partitioning in x0",fontweight = 'bold',fontsize=24,pad = 12)
            odd += 1
        
        ax[i].set_ylabel('x1',rotation = 0, fontweight = 'bold',fontsize=24)
        ax[i].set_xlabel('x0',rotation = 0, fontweight = 'bold',fontsize=24)
        ax[i].set_xlim([0,10])
        ax[i].set_ylim([0,10])
        ax[i].tick_params(axis='both', labelsize=24)
        ax[i].grid(False)

    ax[N_RCIS].set_ylabel('x1',rotation = 0,fontweight = 'bold',fontsize=24)
    ax[N_RCIS].set_xlabel('x0',rotation = 0,fontweight = 'bold',fontsize=24)
    ax[N_RCIS].set_xlim([0,10])
    ax[N_RCIS].set_ylim([0,10])
    ax[N_RCIS].tick_params(axis='both', labelsize=24)
    ax[N_RCIS].grid(False)
    ax[N_RCIS].set_title("{0}-step RCIS (total)".format(N_RCIS),fontweight = 'bold',fontsize = 24,pad = 12)

    suptitle = fig.suptitle("{0}-step RCIS (alternating partitioning[n1={1},n2={2}])".format(N_RCIS,n1,n2),fontweight = 'bold', fontsize=28, y = 1.1)
    fig.align_labels()
    
    # as a performance metric the total size of the RCIS is computed in the following
    # total area (RCIS) = areas of each step - intersection area (for 2 steps: total area = area1 + area2 - intersection of area1 and area2 )
    width_1 = float(res['x_max_even',0,-1,0]-res['x_min_even',0,0,0])
    height_1 = float(res['x_max_even',0,-1,1]-res['x_min_even',0,0,1])
    width_2 = float(res['x_max_odd',0,-1,0]-res['x_min_odd',0,0,0])
    height_2 = float(res['x_max_odd',0,-1,1]-res['x_min_odd',0,0,1])
    overlap_width = max(0,min(float(res['x_min_even',0,0,0]) + width_1, float(res['x_min_odd',0,0,0]) + width_2) - max(float(res['x_min_even',0,0,0]), float(res['x_min_odd',0,0,0])))
    overlap_height = max(0,min(float(res['x_min_even',0,0,1]) + height_1, float(res['x_min_odd',0,0,1]) + height_2) - max(float(res['x_min_even',0,0,1]), float(res['x_min_odd',0,0,1])))
    overlap_area = overlap_width*overlap_height

    RCIS_total = width_1*height_1 + width_2*height_2 - overlap_area

    fig.text(0.95,suptitle.get_position()[1],"total area RCIS: {0}\n computation time[s]: {1}".format(round(RCIS_total,3),round(y_rcis-x_rcis,3)),ha = 'center', va = 'top', color = 'black', fontsize = 12)
# Constraining1 for RCIS
# depending on which dimension was cut lastly we should adjust the corresponding set of the RCIS so that x(N)ExRCIS holds

    if odd_count1 > even_count1:
        g1.append(x_RCIS_plus - opt_x1['x_max_odd', -1, -1])
        g1.append(opt_x1['x_min_odd', -1, 0] - x_RCIS_minus)
        lb_g1.append(np.zeros((2 * nx, 1)))
        ub_g1.append(inf * np.ones((2 * nx, 1)))
    elif odd_count1 == even_count1:
        g1.append(x_RCIS_plus - opt_x1['x_max_even', -1, -1])
        g1.append(opt_x1['x_min_even', -1, 0] - x_RCIS_minus)
        lb_g1.append(np.zeros((2 * nx, 1)))
        ub_g1.append(inf * np.ones((2 * nx, 1)))
        
    # Concatenate constraints
    g1 = vertcat(*g1);
    lb_g1 = vertcat(*lb_g1);
    ub_g1 = vertcat(*ub_g1);

# Constraining2 for RCIS
# depending on which dimension was cut lastly we should adjust the corresponding set of the RCIS so that x(N)ExRCIS holds


    if odd_count2 > even_count2:
        g2.append(x_RCIS_plus - opt_x2['x_max_odd', -1, -1])
        g2.append(opt_x2['x_min_odd', -1, 0] - x_RCIS_minus)
        lb_g2.append(np.zeros((2 * nx, 1)))
        ub_g2.append(inf * np.ones((2 * nx, 1)))
    elif odd_count2 == even_count2:
        g2.append(x_RCIS_plus - opt_x2['x_max_even', -1, -1])
        g2.append(opt_x2['x_min_even', -1, 0] - x_RCIS_minus)
        lb_g2.append(np.zeros((2 * nx, 1)))
        ub_g2.append(inf * np.ones((2 * nx, 1)))
    
    #Concatenate constraints
    g2 = vertcat(*g2)
    lb_g2 = vertcat(*lb_g2)
    ub_g2 = vertcat(*ub_g2)


# set the problem and initialize the optimizer for the first scenario (dim 0 is cut in the first odd step (1) )

    prob1 = {'f':J1,'x':vertcat(opt_x1),'g':g1, 'p':vertcat(x_init,p_plus,p_minus,u_bef,x_ref, x_RCIS_plus, x_RCIS_minus)}
    mpc_mon_solver_cut1 = nlpsol('solver','ipopt',prob1,{'ipopt.max_iter':4000,'ipopt.resto_failure_feasibility_threshold':1e-9,'ipopt.required_infeasibility_reduction':0.99,'ipopt.linear_solver':'MA57','ipopt.ma86_u':1e-6,'ipopt.print_level':3, 'ipopt.sb': 'yes', 'print_time':1,'ipopt.ma57_automatic_scaling':'yes','ipopt.ma57_pre_alloc':10,'ipopt.ma27_meminc_factor':100,'ipopt.ma27_pivtol':1e-4,'ipopt.ma27_la_init_factor':100})

# set the problem and initialize the optimizer for the second scenario (dim 1 is cut in the first odd step (1) )
    prob2 = {'f':J2,'x':vertcat(opt_x2),'g':g2, 'p':vertcat(x_init,p_plus,p_minus,u_bef,x_ref, x_RCIS_plus, x_RCIS_minus)}
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
    
    x_0 = np.array([[1],[1]])
    simulator.reset_history()
    simulator.x0 = x_0
    uinitial = np.zeros((nu,1))
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
            if j % 2 == 0: #cut in the 0th dimension in the first (odd) prediction step (1)
                if odd_count1 > even_count1: # set the corresponding terminal constraint according to the N prediction steps
                    mpc_res1 = mpc_mon_solver_cut1(p=vertcat(x_0,pmax,pmin,u_k,x_sp,np.array(res['x_max_odd',0,-1]),np.array(res['x_min_odd',0, 0])), x0=opt_x_k1, lbg=lb_g1, ubg=ub_g1, lbx = lb_opt_x1, ubx = ub_opt_x1)
                elif odd_count1 == even_count1:
                    mpc_res1 = mpc_mon_solver_cut1(p=vertcat(x_0,pmax,pmin,u_k,x_sp,np.array(res['x_max_even',0,-1]),np.array(res['x_min_even',0, 0])), x0=opt_x_k1, lbg=lb_g1, ubg=ub_g1, lbx = lb_opt_x1, ubx = ub_opt_x1)
            #print return status (mpc_mon_solverf)
                opt.append(copy.deepcopy(mpc_res1))
            elif j % 2 == 1: #cut in the 1 dimension in the first (odd) prediction step (1)
                if odd_count2 > even_count2:
                    mpc_res2 = mpc_mon_solver_cut2(p=vertcat(x_0,pmax,pmin,u_k,x_sp,np.array(res['x_max_even',0,-1]),np.array(res['x_min_even',0, 0])), x0=opt_x_k2, lbg=lb_g2, ubg=ub_g2, lbx = lb_opt_x2, ubx = ub_opt_x2)
                elif  odd_count2 == even_count2:
                    mpc_res2 = mpc_mon_solver_cut2(p=vertcat(x_0,pmax,pmin,u_k,x_sp, np.array(res['x_max_odd',0,-1]),np.array(res['x_min_odd',0, 0])), x0=opt_x_k2, lbg=lb_g2, ubg=ub_g2, lbx = lb_opt_x2, ubx = ub_opt_x2)
                opt.append(copy.deepcopy(mpc_res2))
            
          
                
        
       
        else: # j = 0 and then j % 2 == 0 holds and our first cut will be in the 0th dimension 
            print('Run a first iteration to generate good Warmstart Values')
            if odd_count1 > even_count1: # set the corresponding terminal constraint according to the N prediction steps
                mpc_res1 = mpc_mon_solver_cut1(p=vertcat(x_0,pmax,pmin, uinitial,x_sp,np.array(res['x_max_odd',0,-1]),np.array(res['x_min_odd',0, 0])), x0=opt_x_k1, lbg=lb_g1, ubg=ub_g1, lbx = lb_opt_x1, ubx = ub_opt_x1)
            elif odd_count1 == even_count1:
                mpc_res1 = mpc_mon_solver_cut1(p=vertcat(x_0,pmax,pmin, uinitial,x_sp,np.array(res['x_max_even',0,-1]),np.array(res['x_min_even',0, 0])), x0=opt_x_k1, lbg=lb_g1, ubg=ub_g1, lbx = lb_opt_x1, ubx = ub_opt_x1)       
             
            opt_x_k1 = opt_x1(mpc_res1['x'])
            
            
            if odd_count1 > even_count1: # set the corresponding terminal constraint according to the N prediction steps
                mpc_res1 = mpc_mon_solver_cut1(p=vertcat(x_0,pmax,pmin, uinitial,x_sp,np.array(res['x_max_odd',0,-1]),np.array(res['x_min_odd',0, 0])), x0=opt_x_k1, lbg=lb_g1, ubg=ub_g1, lbx = lb_opt_x1, ubx = ub_opt_x1)
            elif odd_count1 == even_count1:
                mpc_res1 = mpc_mon_solver_cut1(p=vertcat(x_0,pmax,pmin, uinitial,x_sp,np.array(res['x_max_even',0,-1]),np.array(res['x_min_even',0, 0])), x0=opt_x_k1, lbg=lb_g1, ubg=ub_g1, lbx = lb_opt_x1, ubx = ub_opt_x1)
            
            opt.append(copy.deepcopy(mpc_res1))
        # the parameter opt_x_k must have the corresponding dimensions to be passed to the corresponding solver in the next time-step -> case distinction 
        if j % 2 == 0: # odd simulation step follows (j%2 = 1) -> use of mpc-mon_solver_cut2
            opt_x_k1 = opt_x1(mpc_res1['x']) # construction of warmstart "opt_x_k2" from previous solution necessary
            sol.append(copy.deepcopy(opt_x_k1))
            for n in range(1,N2):
                for s in range(n2):
                    opt_x_k2['x_min_odd',n-1,s] = opt_x_k1['x_min_even',n,s]
                    opt_x_k2['x_max_odd',n-1,s] = opt_x_k1['x_max_even',n,s]        
            for n in range(N1):
                for s in range(n1):
                    opt_x_k2['x_min_even',n,s] = opt_x_k1['x_min_odd',n,s]
                    opt_x_k2['x_max_even',n,s] = opt_x_k1['x_max_odd',n,s] 
            if N2 == N1:
                for s in range(n2):
                    opt_x_k2['x_min_odd',N1-1,s] = res['x_min_even',0,s]
                    opt_x_k2['x_max_odd',N1-1,s] = res['x_max_even',0,s]
                for n in range(1,N2):
                    for s in range(n2):
                        opt_x_k2['u_odd',n-1,s] = opt_x_k1['u_even',n,s]
                for n in range(N1-1):
                    for s in range(n1):
                        opt_x_k2['u_even',n,s] = opt_x_k1['u_odd',n,s]
            elif N2 > N1:
                for s in range(n1):
                    opt_x_k2['x_min_even',N2-1,s] = res['x_min_odd',0,s]
                    opt_x_k2['x_max_even',N2-1,s] = res['x_max_odd',0,s]
                for n in range(1,N2-1):
                    for s in range(n2):
                        opt_x_k2['u_odd',n-1,s] = opt_x_k1['u_even',n,s]
                for n in range(N1):
                    for s in range(n1):
                        opt_x_k2['u_even',n,s] = opt_x_k1['u_odd',n,s]
           
            u_k = opt_x_k1['u_even',0,0]
            
        elif j % 2 == 1: # even simulation step follows (j%2 = 0) -> use of mpc-mon_solver_cut1
            opt_x_k2 = opt_x2(mpc_res2['x'])
            sol.append(copy.deepcopy(opt_x_k2))
            for n in range(N1):
                for s in range(n2):
                    opt_x_k1['x_min_even',n,s] = opt_x_k2['x_min_odd',n,s]
                    opt_x_k1['x_max_even',n,s] = opt_x_k2['x_max_odd',n,s]
            for n in range(1,N2):
                for s in range(n1):
                    opt_x_k1['x_min_odd',n-1,s] = opt_x_k2['x_min_even',n,s]
                    opt_x_k1['x_max_odd',n-1,s] = opt_x_k2['x_max_even',n,s]          
            if N2 == N1:
                for s in range(n1):
                    opt_x_k1['x_min_odd',N1-1,s] = res['x_min_odd',0,s]
                    opt_x_k1['x_max_odd',N1-1,s] = res['x_max_odd',0,s]
                for n in range(1,N2):
                    for s in range(n1):
                        opt_x_k1['u_odd',n-1,s] = opt_x_k2['u_even',n,s]
                for n in range(N1-1):
                    for s in range(n2):
                        opt_x_k1['u_even',n,s] = opt_x_k2['u_odd',n,s]
            elif N2 > N1:       
                for s in range(n2):
                    opt_x_k1['x_min_even',N2-1,s] = res['x_min_even',0,s]
                    opt_x_k1['x_max_even',N2-1] = res['x_max_even',0,s]
                for n in range(1,N2-1):
                    for s in range(n1):
                        opt_x_k1['u_odd',n-1,s] = opt_x_k2['u_even',n,s]
                for n in range(N1):
                    for s in range(n2):
                        opt_x_k1['u_even',n,s] = opt_x_k2['u_odd',n,s]
            
            u_k = opt_x_k2['u_even',0,0]
           
    # simulate the system
        x_next = simulator.make_step(u_k)

    
    # Update the initial state
        x_0 = x_next
    
#        fig, ax =plt.subplots(1,2,layout = 'constrained')
#        for i in range(2):
#            if j%2 == 0: #even simulation step -> visualize cut in the first odd prediction step in  dim 0 and cut in the following even step in dim 0
#                if N1 < N2: 
#                    if i%2 == 0:
#                        for s in range(n1):
#                            ax[i].add_patch(mpl.patches.Rectangle(np.array(opt_x_k1['x_min_odd',-1,s]), np.array(opt_x_k1['x_max_odd',-1,s]-opt_x_k1['x_min_odd',-1,s])[0][0] , np.array(opt_x_k1['x_max_odd',-1,s]-opt_x_k1['x_min_odd',-1,s])[1][0], color="None",ec='grey'))
#                        ax[i].set_title("partitioning in x0")     
#                    elif i%2 == 1:
#                        for s in range(n2):
#                            ax[i].add_patch(mpl.patches.Rectangle(np.array(opt_x_k1['x_min_even',-1,s]), np.array(opt_x_k1['x_max_even',-1,s]-opt_x_k1['x_min_even',-1,s])[0][0] , np.array(opt_x_k1['x_max_even',-1,s]-opt_x_k1['x_min_even',-1,s])[1][0], color="None",ec='grey'))
#                        ax[i].set_title("partitioning in x1")
#                elif N1 == N2:
#                    if i%2 == 0:
#                        for s in range(n2):
#                            ax[i].add_patch(mpl.patches.Rectangle(np.array(opt_x_k1['x_min_even',-1,s]), np.array(opt_x_k1['x_max_even',-1,s]-opt_x_k1['x_min_even',-1,s])[0][0] , np.array(opt_x_k1['x_max_even',-1,s]-opt_x_k1['x_min_even',-1,s])[1][0], color="None",ec='grey'))
#                        ax[i].set_title("partitioning in x1")     
#                    elif i%2 == 1:
#                        for s in range(n1):
#                            ax[i].add_patch(mpl.patches.Rectangle(np.array(opt_x_k1['x_min_odd',-1,s]), np.array(opt_x_k1['x_max_odd',-1,s]-opt_x_k1['x_min_odd',-1,s])[0][0] , np.array(opt_x_k1['x_max_odd',-1,s]-opt_x_k1['x_min_odd',-1,s])[1][0], color="None",ec='grey'))
#                        ax[i].set_title("partitioning in x0")
#            elif j%2 == 1: #odd simulation step -> visualize cut in the first odd prediction step in  dim 1 and cut in following even step in dim 1
#                if N1 < N2:
#                    if i%2 == 0:
#                        for s in range(n2):
#                            ax[i].add_patch(mpl.patches.Rectangle(np.array(opt_x_k2['x_min_odd',-1,s]), np.array(opt_x_k2['x_max_odd',-1,s]-opt_x_k2['x_min_odd',-1,s])[0][0] , np.array(opt_x_k2['x_max_odd',-1,s]-opt_x_k2['x_min_odd',-1,s])[1][0], color="None",ec='grey'))
#                        ax[i].set_title("partitioning in x1")     
#                    elif i%2 == 1:
#                        for s in range(n1):
#                            ax[i].add_patch(mpl.patches.Rectangle(np.array(opt_x_k2['x_min_even',-1,s]), np.array(opt_x_k2['x_max_even',-1,s]-opt_x_k2['x_min_even',-1,s])[0][0] , np.array(opt_x_k2['x_max_even',-1,s]-opt_x_k2['x_min_even',-1,s])[1][0], color="None",ec='grey'))
#                        ax[i].set_title("partitioning in x0")
#                elif N1 == N2:
#                    if i%2 == 0:
#                        for s in range(n1):
#                            ax[i].add_patch(mpl.patches.Rectangle(np.array(opt_x_k2['x_min_even',-1,s]), np.array(opt_x_k2['x_max_even',-1,s]-opt_x_k2['x_min_even',-1,s])[0][0] , np.array(opt_x_k2['x_max_even',-1,s]-opt_x_k2['x_min_even',-1,s])[1][0], color="None",ec='grey'))
#                        ax[i].set_title("partitioning in x1")     
#                    elif i%2 == 1:
#                        for s in range(n2):
#                            ax[i].add_patch(mpl.patches.Rectangle(np.array(opt_x_k2['x_min_odd',-1,s]), np.array(opt_x_k2['x_max_odd',-1,s]-opt_x_k2['x_min_odd',-1,s])[0][0] , np.array(opt_x_k2['x_max_odd',-1,s]-opt_x_k2['x_min_odd',-1,s])[1][0], color="None",ec='grey'))
#                        ax[i].set_title("partitioning in x0")
            
#            ax[i].set_ylabel('x_1')
#            ax[i].set_xlabel('x_0')
#            ax[i].set_xlim([0,10])
#            ax[i].set_ylim([0,10])
#            ax[i].grid(False)
            

#        fig.align_labels()
    subreg.append(n1); subreg.append(n2)
    CL_time = time.time()-CL_time #time measurement for the closed-loop-run 
    mpc_mon_cut_res=copy.copy(simulator.data)
    
    datadict = {'CL': [mpc_mon_cut_res,CL_time], 'OL': [opt,sol], 'ns': [subreg], 'counter': [N1,N2],'RCIS':[RCIS_total]}
    
    return datadict

datadict_CL_run_different = closed_loop_comp(0,3,4,5,6,[1,0],[0,1])  

np.save('datadict_CL_run_different',datadict_CL_run_different)    
    
    
