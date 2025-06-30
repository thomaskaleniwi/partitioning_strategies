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
dt = 1 # timestept
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
        rand = np.random.uniform(0,1,size = None)
        p_template['p'] = interpol(p0*(1+p_var),p0*(1-p_var),rand)
    else:
        np.random.seed(int(t//dt+in_seed))
        rand=np.random.uniform(0,1,size = None)
        p_template['p'] = interpol(p0*(1+p_var),k1_0*(1-p_var),rand)
    return p_template

simulator.set_tvp_fun(p_fun_0)
simulator.setup()

# pass the ordering, the number of partitions in the different dimensions     
# seed defines the corresponding uncertainty case
# x0 = number of partitions in the first dimension(0) for the first prediction step -> alternating such that in the second step the second dimension(1) will be partitioned like this
# x1 = number of partitions in the second dimension(1) for the first prediction step -> alternating such that in the second step the first dimension(0) will be partitioned like this
# ord1 = define which dimension to be partitioned firstly in the first prediction step
def closed_loop_comp(seed,x0,x1,ord1,ord2):
    
    N = 30 #chosen same as in paper "Determinstic safety guarantees for learning-based control of monotone systems"
    N_RCIS = 2
    x_sp = np.array([[5],[2]])
    Q = np.array([[1,0],
                  [0,1]])
    R = np.array([[1,0],
                  [0,1]])
    u_bef = SX.sym('u_bef',nu, 1)
    x_ref = SX.sym('x_ref',nx, 1)
    
    stage_cost = (model.x['x']-x_ref).T@Q@(model.x['x']-x_ref)+ (model.u['u']-u_bef).T@R@(model.u['u']-u_bef) 
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
# therefore two lists and two cutting vectors are necessary
# each cutvector contains the number of cuts in the respective dimensions     
    
    cut1 = np.zeros((nx,1)) 
    cut1[0] = x0 # cuts in the first dimension (0)
    cut1[1] = x1 # cuts in the second dimension (1)
    
    cut2 = np.zeros((nx,1)) 
    cut2[0] = x0 # cuts in the first dimension (0)
    cut2[1] = x1 # cuts in the second dimension (1)
    
    ordering1 = ord1
    ordering2 = ord2
    
# define number of subregions with the number of cuts in each dimension
# formula is valid if the total number of cuts for a specific dimension is the same in each subregion  
    n1 = 1 # these are the initial values and they will be updated both in the following according to the respective cuts 
    
    for i in range(nx):
        n1*=(cut1[i]+1)
    n1 = int(n1[0])
    
    
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
      
# first assume that ns = n1 = n2
    ns = n1    #evtl assert benutzen
    assert ns == n1
    
    opt_x = struct_symSX([
        entry('x_min', shape=nx, repeat=[N+1,ns]),
        entry('x_max', shape=nx, repeat=[N+1,ns]),
        entry('u', shape=nu, repeat=[N,ns]),
    ])
    
    x_RCIS = struct_symSX([
        entry('x_min', shape=nx ,repeat=[N_RCIS,ns]),
        entry('x_max', shape=nx ,repeat=[N_RCIS,ns]),
        entry('u', shape=nu, repeat=[N_RCIS,ns])
    ])

# set the bounds on opt_x  and x_RCIS   
    lb_opt_x = opt_x(0)
    ub_opt_x = opt_x(np.inf)



    lb_opt_x['x_min'] = lb_x
    ub_opt_x['x_min'] = ub_x
    lb_opt_x['x_max'] = lb_x
    ub_opt_x['x_max'] = ub_x


    lb_opt_x['u'] = lb_u
    ub_opt_x['u'] = ub_u

###############################################################################

    lb_opt_x_RCIS = x_RCIS(0)
    ub_opt_x_RCIS = x_RCIS(np.inf)

    lb_opt_x_RCIS['x_min'] = lb_x
    lb_opt_x_RCIS['x_max'] = lb_x
    ub_opt_x_RCIS['x_min'] = ub_x
    ub_opt_x_RCIS['x_max'] = ub_x

    lb_opt_x_RCIS['u'] = lb_u
    ub_opt_x_RCIS['u'] = ub_u
# define the constraint function 

    def constraint_function(l,ord_dim,opt_x,i,h,lbg,ubg):
        for k in range(len(l)):
            idx=flatten(l[k])
            dim=ord_dim[-depth(l)]
            for s in idx:
                if s==idx[0] and k==0:
                    h.append(opt_x['x_min',i,s,dim]-opt_x['x_min',i,0,dim])
                    lbg.append(0)
                    ubg.append(0)
                else:
                    h.append(opt_x['x_min',i,s,dim]-opt_x['x_min',i,idx[0],dim])
                    lbg.append(0)
                    ubg.append(0)
                if s==idx[-1] and k==len(l)-1:
                    h.append(opt_x['x_max',i,s,dim]-opt_x['x_max',i,-1,dim])
                    lbg.append(0)
                    ubg.append(0)
                else:
                    h.append(opt_x['x_max',i,s,dim]-opt_x['x_max',i,idx[-1],dim])
                    lbg.append(0)
                    ubg.append(0)
            if k>=1:
                prev_last=flatten(l[k-1])[-1]
                h.append(opt_x['x_min',i,idx[0],dim]-opt_x['x_max',i,prev_last,dim])
                lbg.append(0)
                ubg.append(0)
            if depth(l) >1:
                h,lbg,ubg=constraint_function(l[k],ord_dim,opt_x,i,h,lbg,ubg)
        
        return h,lbg,ubg
    

# Set up the objective and the constraints of the problem


    J = 0 # cost fct for normal prediction horizon N
    g = [] #constraint expression
    lb_g = [] #lower bound for constraint expression g
    ub_g = [] #upper bound for constraint expression g
    
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
    
    
# Set initial constraints
    
    for s in range(ns):
        g.append(opt_x['x_min',0,s]-x_init)
        g.append(opt_x['x_max',0,s]-x_init)
        lb_g.append(np.zeros((2*nx,1)))
        ub_g.append(np.zeros((2*nx,1)))
        if s>0:
            g.append(opt_x['u',0,s]-opt_x['u',0,0])
            lb_g.append(np.zeros((nu,1)))
            ub_g.append(np.zeros((nu,1)))

# objective and equality constraints should be the same since ns (number of subregions) remains same            
    for i in range(N):
        # objective
        for s in range(ns):
            if i==0:
                J += stage_cost_fcn(opt_x['x_max',i,s], opt_x['u',i,s],u_bef,x_ref)
                J += stage_cost_fcn(opt_x['x_min',i,s], opt_x['u',i,s],u_bef,x_ref)
            else:
                J += stage_cost_fcn(opt_x['x_max',i,s], opt_x['u',i,s],opt_x['u',i-1,s],x_ref)
                J += stage_cost_fcn(opt_x['x_min',i,s], opt_x['u',i,s],opt_x['u',i-1,s],x_ref)
#                J += stage_cost_fcn(opt_x['x_max',i,s], opt_x['u',i,s],sum(opt_x['u',i-1,:])/len(opt_x['u',i-1,:]),x_ref)
#                J += stage_cost_fcn(opt_x['x_min',i,s], opt_x['u',i,s],sum(opt_x['u',i-1,:])/len(opt_x['u',i-1,:]),x_ref)

            # inequality constraints (system equation)
          
        for s in range(ns):
            x_next_max = system(opt_x['x_max',i,s],opt_x['u',i,s],p_max)
            x_next_min = system(opt_x['x_min',i,s],opt_x['u',i,s],p_min)

            g.append( opt_x['x_max', i+1,-1]-x_next_max)
            g.append(x_next_min - opt_x['x_min', i+1,0])
            lb_g.append(np.zeros((2*nx,1)))
            ub_g.append(np.ones((2*nx,1))*inf)

# terminal cost                

    for s in range(ns):
        J += terminal_cost_fcn(opt_x['x_max',-1, s],x_ref)
        J += terminal_cost_fcn(opt_x['x_min',-1, s],x_ref)

# actually this step is only important for the alternating partitioning approaches but it is possible to modify them "backwards" to the full partitioning approach by 
# but it is possible to modify them "backwards" to the full partitioning approach by definining lis1 and lis2 and ordering1 and ordering2 equal (as already done) 
# in that case g1,lb_g1 and ub_g1 will be the same no matter whether prediction step i is even or odd since the same arguments are passed to the constraint function 
# this problem would be also implementable with only cutting1 but the following shows the flexibility of the alternating cutting-approach and that it can be transformed backwards to the full-cutting (as previously mentioned)
# cutting1 
        
    g1 = copy.deepcopy(g)
    lb_g1 = copy.deepcopy(lb_g)
    ub_g1 = copy.deepcopy(ub_g)
    even_count1 = 0
    odd_count1 = 0
    for i in range(1,N+1): # at i = 0 there should be no cutting since we have a distinct initial point and not a set
        # if i is even, pass lis2 and ordering2
        # if i is odd, pass lis1 and ordering1
        
        if i % 2 == 0:
            g1, lb_g1, ub_g1 = constraint_function(lis2,ordering2,opt_x,i,g1,lb_g1,ub_g1)  
            even_count1 += 1
        elif i % 2 == 1:
            g1, lb_g1, ub_g1 = constraint_function(lis1,ordering1,opt_x,i,g1,lb_g1,ub_g1)  
            odd_count1 += 1
        for s in range(ns):
            g1.append(opt_x['x_max',i,s]-opt_x['x_min',i,0])
            g1.append(opt_x['x_max',i,-1]-opt_x['x_min',i,s])
            g1.append(opt_x['x_min',i,s]-opt_x['x_min',i,0])
            g1.append(opt_x['x_max',i,-1]-opt_x['x_max',i,s])
            lb_g1.append(np.zeros((4*nx,1)))
            ub_g1.append(np.ones((4*nx,1))*inf)
                

# cutting2  
       

    g2 = copy.deepcopy(g)
    lb_g2 = copy.deepcopy(lb_g)
    ub_g2 = copy.deepcopy(ub_g)
    even_count2 = 0
    odd_count2 = 0
    for i in range(1,N+1):
        # if i is even, pass lis1 and ordering1
        # if i is odd, pass lis2 and ordering2
        
        if i % 2 == 0:
            g2, lb_g2, ub_g2 = constraint_function(lis1,ordering1,opt_x,i,g2,lb_g2,ub_g2)  
            even_count2 += 1              
        elif i % 2 == 1:
            g2, lb_g2, ub_g2 = constraint_function(lis2,ordering2,opt_x,i,g2,lb_g2,ub_g2)  
            odd_count2 += 1
        for s in range(ns):
            g2.append(opt_x['x_max',i,s]-opt_x['x_min',i,0])
            g2.append(opt_x['x_max',i,-1]-opt_x['x_min',i,s])
            g2.append(opt_x['x_min',i,s]-opt_x['x_min',i,0])
            g2.append(opt_x['x_max',i,-1]-opt_x['x_max',i,s])
            lb_g2.append(np.zeros((4*nx,1)))
            ub_g2.append(np.ones((4*nx,1))*inf)
                                

# computation of 2-step RCIS # one-step is enough
    
#    N_RCIS = 2
    x_rcis = time.time()
    for i in range(N_RCIS):
        for s in range(ns):
            x_next_plus_RCIS = system(x_RCIS['x_max',i,s], x_RCIS['u',i,s],p_max)
            x_next_minus_RCIS = system(x_RCIS['x_min',i,s], x_RCIS['u',i,s],p_min)
            if i == N_RCIS-1:
                g_RCIS.append(x_RCIS['x_max',0,-1] - x_next_plus_RCIS)
                g_RCIS.append(x_next_minus_RCIS - x_RCIS['x_min',0, 0])
            else:
                g_RCIS.append(x_RCIS['x_max',i+1,-1] - x_next_plus_RCIS)
                g_RCIS.append(x_next_minus_RCIS - x_RCIS['x_min',i+1, 0])
            lb_g_RCIS.append(np.zeros((2*nx,1)))
            ub_g_RCIS.append(inf*np.ones((2*nx,1)))
        # Cutting RCIS
        # cut in alternating dimensions depending on whether we are in an even or an odd step
        if i % 2 == 0: # cut in the 0th dimension
            g_RCIS,lb_g_RCIS,ub_g_RCIS = constraint_function(lis1,ordering1,x_RCIS,i,g_RCIS,lb_g_RCIS,ub_g_RCIS)
        elif i % 2 == 1: # cut in the 1 dimension
            g_RCIS,lb_g_RCIS,ub_g_RCIS = constraint_function(lis2,ordering2,x_RCIS,i,g_RCIS,lb_g_RCIS,ub_g_RCIS)
        for s in range(ns):
            g_RCIS.append(x_RCIS['x_max',i,s]-x_RCIS['x_min',i,0])
            g_RCIS.append(x_RCIS['x_max',i,-1]-x_RCIS['x_min',i,s])
            g_RCIS.append(x_RCIS['x_min',i,s]-x_RCIS['x_min',i,0])
            g_RCIS.append(x_RCIS['x_max',i,-1]-x_RCIS['x_max',i,s])
            lb_g_RCIS.append(np.zeros((4*nx,1)))
            ub_g_RCIS.append(np.ones((4*nx,1))*inf)
            
    J_RCIS = -1;
    for i in range(1):
        J_mini = -1 #negative because maximizing then means becoming as negative as the constraints allow it
        for ix in range(nx):
            J_mini = J_mini*(x_RCIS['x_max',i,-1,ix]-x_RCIS['x_min',i,0,ix])
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
    opt_ro_initial['x_min'] = x_set
    opt_ro_initial['x_max'] = x_set
    results = solver_mx_inv_set(p=vertcat(x_set,p_max,p_min),x0=opt_ro_initial, lbg=lb_g_RCIS,ubg=ub_g_RCIS,lbx=lb_opt_x_RCIS,ubx=ub_opt_x_RCIS)
    y_rcis = time.time()
    print("time to compute the {0}-step RCIS:".format(N_RCIS) ,y_rcis-x_rcis)
    # transform the optimizer results into a structured symbolic variable
    res = x_RCIS(results['x'])

    # plot the simulation results (RCIS for each step and the total RCIS)
    fig, ax =plt.subplots(1,N_RCIS+1,layout = 'constrained',figsize = (20,9))
    for i in range(N_RCIS):
        for s in range(ns):
            ax[i].add_patch(mpl.patches.Rectangle(np.array(res['x_min',i,s]), np.array(res['x_max',i,s]-res['x_min',i,s])[0][0] , np.array(res['x_max',i,s]-res['x_min',i,s])[1][0], color="None",ec='red'))
            ax[i].text(np.array(res['x_min',i,s,0]+0.5*(res['x_max',i,s,0]-res['x_min',i,s,0])),np.array(res['x_min',i,s,1]+0.5*(res['x_max',i,s,1]-res['x_min',i,s,1])),str(s),ha = 'center', va = 'center', color = 'black', fontweight = 'bold', fontsize ='xx-small')
            ax[N_RCIS].add_patch(mpl.patches.Rectangle(np.array(res['x_min',i,s]), np.array(res['x_max',i,s]-res['x_min',i,s])[0][0] , np.array(res['x_max',i,s]-res['x_min',i,s])[1][0], color="None",ec='red'))
        ax[i].set_ylabel('x1',rotation = 0,fontweight = 'bold',fontsize = 24)
        ax[i].set_xlabel('x0',rotation = 0,fontweight = 'bold',fontsize = 24)
        ax[i].set_xlim([0,10])
        ax[i].set_ylim([0,10])
        ax[i].tick_params(axis='both', labelsize=24)
        ax[i].grid(False)
        if i%2 == 0:
            ax[i].set_title("{0}-step".format(i+1),fontweight = 'bold')
        elif i%2 == 1:
            ax[i].set_title("{0}-step".format(i+1),fontweight = 'bold')
        
    ax[N_RCIS].set_ylabel('x1',rotation = 0,fontweight = 'bold')
    ax[N_RCIS].set_xlabel('x0',rotation = 0,fontweight = 'bold')
    ax[N_RCIS].set_xlim([0,10])
    ax[N_RCIS].set_ylim([0,10])
    ax[N_RCIS].tick_params(axis='both', labelsize=24)
    ax[N_RCIS].grid(False)
    ax[N_RCIS].set_title("{0}-step RCIS (total)".format(N_RCIS),fontweight = 'bold',fontsize = 24,pad = 12)
    
    
    suptitle = fig.suptitle("{0}-step RCIS (full partitioning[ns={1}])".format(N_RCIS,ns),fontweight = 'bold', fontsize=28,y = 1.1)
    fig.align_labels()
    
    # as a performance metric the total size of the RCIS is computed in the following
    # total area (RCIS) = areas of each step - intersection area (for 2 steps: total area = area1 + area2 - intersection of area1 and area2 )
    width_1 = float(res['x_max',0,-1,0]-res['x_min',0,0,0])
    height_1 = float(res['x_max',0,-1,1]-res['x_min',0,0,1])
    width_2 = float(res['x_max',1,-1,0]-res['x_min',1,0,0])
    height_2 = float(res['x_max',1,-1,1]-res['x_min',1,0,1])
    overlap_width = max(0,min(float(res['x_min',0,0,0]) + width_1, float(res['x_min',1,0,0]) + width_2) - max(float(res['x_min',0,0,0]), float(res['x_min',1,0,0])))
    overlap_height = max(0,min(float(res['x_min',0,0,1]) + height_1, float(res['x_min',1,0,1]) + height_2) - max(float(res['x_min',0,0,1]), float(res['x_min',1,0,1])))
    overlap_area = overlap_width*overlap_height

    RCIS_total = width_1*height_1 + width_2*height_2 - overlap_area

    fig.text(0.95,suptitle.get_position()[1],"total area RCIS: {0} \n computation time[s]: {1}".format(round(RCIS_total,3),round(y_rcis-x_rcis,3)),ha = 'center', va = 'top', color = 'black', fontsize = 12)
# Constraining1 for RCIS
# adjust in the corresponding set of the RCIS so that x(N)ExRCIS holds

    
    g1.append(x_RCIS_plus - opt_x['x_max', -1, -1])
    g1.append(opt_x['x_min', -1, 0] - x_RCIS_minus)
    lb_g1.append(np.zeros((2 * nx, 1)))
    ub_g1.append(inf * np.ones((2 * nx, 1)))

    # Concatenate constraints
    g1 = vertcat(*g1)
    lb_g1 = vertcat(*lb_g1)
    ub_g1 = vertcat(*ub_g1)

# Constraining2 for RCIS
# adjust the corresponding set of the RCIS so that x(N)ExRCIS holds


    g2.append(x_RCIS_plus - opt_x['x_max', -1, -1])
    g2.append(opt_x['x_min', -1, 0] - x_RCIS_minus)
    lb_g2.append(np.zeros((2 * nx, 1)))
    ub_g2.append(inf * np.ones((2 * nx, 1)))

    #Concatenate constraints
    g2 = vertcat(*g2)
    lb_g2 = vertcat(*lb_g2)
    ub_g2 = vertcat(*ub_g2)

# instead of setting two different solvers one could also define g1 and g2 and then define the optimization
# problem with one parameter p' that defines whether one is in an even or an odd step and thus it is set to 0 or 1 such that the corresponding g is then passed to the problem

# one problem would be enough for the full-partitioning case since the  partitioning remains the same in each step 
# such that the solution of on prediction aligns already with the previous solution
# long term short: for the full-partitioning case two solvers will be used but they are the same since the problem is defined equally and the switching in the even/odd simulation steps will not have an effect

    prob1 = {'f':J,'x':vertcat(opt_x),'g':g1, 'p':vertcat(x_init,p_plus,p_minus,u_bef,x_ref, x_RCIS_plus, x_RCIS_minus)}
    mpc_mon_solver_cut1 = nlpsol('solver','ipopt',prob1,{'ipopt.max_iter':4000,'ipopt.resto_failure_feasibility_threshold':1e-9,'ipopt.required_infeasibility_reduction':0.99,'ipopt.linear_solver':'MA57','ipopt.ma86_u':1e-6,'ipopt.print_level':3, 'ipopt.sb': 'yes', 'print_time':1,'ipopt.ma57_automatic_scaling':'yes','ipopt.ma57_pre_alloc':10,'ipopt.ma27_meminc_factor':100,'ipopt.ma27_pivtol':1e-4,'ipopt.ma27_la_init_factor':100})


    prob2 = {'f':J,'x':vertcat(opt_x),'g':g2, 'p':vertcat(x_init,p_plus,p_minus,u_bef,x_ref, x_RCIS_plus, x_RCIS_minus)}
    mpc_mon_solver_cut2 = nlpsol('solver','ipopt',prob2,{'ipopt.max_iter':4000,'ipopt.resto_failure_feasibility_threshold':1e-9,'ipopt.required_infeasibility_reduction':0.99,'ipopt.linear_solver':'MA57','ipopt.ma86_u':1e-6,'ipopt.print_level':3, 'ipopt.sb': 'yes', 'print_time':1,'ipopt.ma57_automatic_scaling':'yes','ipopt.ma57_pre_alloc':10,'ipopt.ma27_meminc_factor':100,'ipopt.ma27_pivtol':1e-4,'ipopt.ma27_la_init_factor':100})



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
    opt_x_k = opt_x(1)
    
    N_sim = 20
    pmin = vertcat(p_fun_wcmin(0))
    pmax = vertcat(p_fun_wcmax(0))
    
    CL_time = time.time()
    opt = []
    sol = []
    subreg = [] 
    for j in range(N_sim):
    # solve optimization problem
    # if j is odd use solver1 and if j is even use solver 2 
        print(j)

        if j>0:
            if j % 2 == 0:
                if odd_count1 > even_count1: 
                    print(x_0)
                    mpc_res = mpc_mon_solver_cut1(p=vertcat(x_0,pmax,pmin,u_k,x_sp,np.array(res['x_max',0,-1]),np.array(res['x_min',0, 0])), x0=opt_x_k, lbg=lb_g1, ubg=ub_g1, lbx = lb_opt_x, ubx = ub_opt_x)
                elif odd_count1 == even_count1:
                    print(x_0)
                    mpc_res = mpc_mon_solver_cut1(p=vertcat(x_0,pmax,pmin,u_k,x_sp,np.array(res['x_max',0,-1]),np.array(res['x_min',0, 0])), x0=opt_x_k, lbg=lb_g1, ubg=ub_g1, lbx = lb_opt_x, ubx = ub_opt_x)
            elif j % 2 == 1:  
                if odd_count2 > even_count2:
                    print(x_0)
                    mpc_res = mpc_mon_solver_cut2(p=vertcat(x_0,pmax,pmin,u_k,x_sp,np.array(res['x_max',0,-1]),np.array(res['x_min',0, 0])), x0=opt_x_k, lbg=lb_g2, ubg=ub_g2, lbx = lb_opt_x, ubx = ub_opt_x)
                elif  odd_count2 == even_count2:
                    print(x_0)
                    mpc_res = mpc_mon_solver_cut2(p=vertcat(x_0,pmax,pmin,u_k,x_sp, np.array(res['x_max',0,-1]),np.array(res['x_min',0, 0])), x0=opt_x_k, lbg=lb_g2, ubg=ub_g2, lbx = lb_opt_x, ubx = ub_opt_x)
            
            
          
                
        
       
        else: # j = 0 and then j % 2 == 0 
            print('Run a first iteration to generate good Warmstart Values')
            if odd_count1 > even_count1: 
                mpc_res = mpc_mon_solver_cut1(p=vertcat(x_0,pmax,pmin, uinitial,x_sp,np.array(res['x_max',0,-1]),np.array(res['x_min',0, 0])), x0=opt_x_k, lbg=lb_g1, ubg=ub_g1, lbx = lb_opt_x, ubx = ub_opt_x)
            elif odd_count1 == even_count1:
                mpc_res = mpc_mon_solver_cut1(p=vertcat(x_0,pmax,pmin, uinitial,x_sp,np.array(res['x_max',0,-1]),np.array(res['x_min',0, 0])), x0=opt_x_k, lbg=lb_g1, ubg=ub_g1, lbx = lb_opt_x, ubx = ub_opt_x)         
       
                
       
            opt_x_k = opt_x(mpc_res['x'])
            
            
            if odd_count1 > even_count1: 
                mpc_res = mpc_mon_solver_cut1(p=vertcat(x_0,pmax,pmin, uinitial,x_sp,np.array(res['x_max',0,-1]),np.array(res['x_min',0, 0])), x0=opt_x_k, lbg=lb_g1, ubg=ub_g1, lbx = lb_opt_x, ubx = ub_opt_x)
            elif odd_count1 == even_count1:
                mpc_res = mpc_mon_solver_cut1(p=vertcat(x_0,pmax,pmin, uinitial,x_sp,np.array(res['x_max',0,-1]),np.array(res['x_min',0, 0])), x0=opt_x_k, lbg=lb_g1, ubg=ub_g1, lbx = lb_opt_x, ubx = ub_opt_x)
           
        
        opt_x_k = opt_x(mpc_res['x'])

        u_k = opt_x_k['u',0,0]
    # simulate the system
        x_next = simulator.make_step(u_k)

    
    # Update the initial state
        x_0 = x_next
    
#        fig, ax =plt.subplots(1,2,layout = 'constrained')
#        for i in range(2): 
#            if i == 0:
#                for s in range(ns):
#                    ax[i].add_patch(mpl.patches.Rectangle(np.array(opt_x_k['x_min',-2,s]), np.array(opt_x_k['x_max',-2,s]-opt_x_k['x_min',-2,s])[0][0] , np.array(opt_x_k['x_max',-2,s]-opt_x_k['x_min',-2,s])[1][0], color="None",ec='grey'))
#            if i == 1:
#                for s in range(ns):
#                    ax[i].add_patch(mpl.patches.Rectangle(np.array(opt_x_k['x_min',-1,s]), np.array(opt_x_k['x_max',-1,s]-opt_x_k['x_min',-1,s])[0][0] , np.array(opt_x_k['x_max',-1,s]-opt_x_k['x_min',-1,s])[1][0], color="None",ec='grey'))
            
#            ax[i].set_ylabel('x_1')
#            ax[i].set_xlabel('x_0')
#            ax[i].set_xlim([0,10])
#            ax[i].set_ylim([0,10])
#            ax[i].grid(False)
#            if i%2 == 0:
#                if j%2 == 0: #case distinction to assign the correct partitioning to the corresponding axes
#                    ax[i].set_title("partitioning in x_0")
#                elif j%2 == 1:
#                    ax[i].set_title("partitioning in x_1")
#            elif i%2 == 1:
#                if j%2 == 0:
#                    ax[i].set_title("partitioning in x_1")
#                elif j%2 == 1:
#                    ax[i].set_title("partitioning in x_0")

#        fig.align_labels()
        opt.append(copy.deepcopy(mpc_res))
        sol.append(copy.deepcopy(opt_x_k))
    subreg.append(ns)
    CL_time = time.time()-CL_time #time-measurement to evaluate the performance of the closed-loop simulation 
    mpc_mon_cut_res=copy.copy(simulator.data)
    
    datadict = {'CL': [mpc_mon_cut_res,CL_time], 'OL': [opt,sol], 'ns': [subreg],'RCIS':[RCIS_total]}
    
    return datadict

datadict_CL_run_full = closed_loop_comp(0,4,5,[0,1],[0,1])
    
np.save('datadict_CL_run_full_30.npy',datadict_CL_run_full)
    
    