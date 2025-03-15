# here we want to compute the 2-step RCIS
# the first set will be cut in the first dimension (0) and the second set will be cut in the second dimension (1)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from casadi import *
from casadi.tools import *
import control
import time as time
import do_mpc

# we assume that this code will be built in the script "partitioning in different timesteps double integrator"
# therefore it would not be necessary to define it here again

# constants
dt = 1
p0 = 0.15 # uncertainty parameter (tvp) should be between 0.00 and 0.30
nx = 2 # number of inputs 
nu = 2 # number of states
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

lb_x = 0*np.ones((nx,1))
ub_x = 10*np.ones((nx,1))

# input constraints

lb_u = np.array([[-10],
                 [-5]]) 
ub_u = np.array([[10],
                 [5]]) 

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


# now we define the optimization problem and everything that is necessary

N = 2 # number of steps as we only need a two step reachable set

# define the number of cuts in each dimension

cut1 = np.zeros((nx,1)) # cuts in the first dimension
cut1[0] = 24
cut1[1] = 0
    
cut2 = np.zeros((nx,1)) # cuts in the second dimension
cut2[0] = 0
cut2[1] = 20
    
ordering1 = [0,1]
ordering2 = [1,0]

lis1 = [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14],[15],[16],[17],[18],[19],[20],[21],[22],[23],[24]]
lis2 = [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14],[15],[16],[17],[18],[19],[20]]

n1 = 1
n2 = 1

for i in range(nx):
    n1*=(cut1[i]+1)
n1 = int(n1[0])

for i in range(nx):
    n2*=(cut2[i]+1)
n2 = int(n2[0])

# define optimzation variables and their upper and lower bounds

x_RCIS = struct_symSX([
   entry('x_min_odd', shape=nx ,repeat=[1,n1]),
   entry('x_max_odd', shape=nx ,repeat=[1,n1]),
   entry('u_odd', shape=nu, repeat=[1,n1]),
   entry('x_min_even', shape=nx ,repeat=[1,n2]),
   entry('x_max_even', shape=nx ,repeat=[1,n2]),
   entry('u_even', shape=nu, repeat=[1,n2])
   ])

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

# optimization problem

J_RCIS = 0
g_RCIS = []
lb_g_RCIS = []
ub_g_RCIS = []

x_ref = SX.sym('x_ref', nx)
p_plus = SX.sym('p_plus', nd)
p_minus = SX.sym('p_minus', nd)
x_RCIS_plus = SX.sym('x_RCIS_plus',nx,1)
x_RCIS_minus = SX.sym('x_RCIS_minus',nx,1)

N_RCIS = 2
even_count_RCIS = 0 #actually here unnecessary because a 2-step RCIS is chosen here, but it would be necessary if there would be a N(>2)-step RCIS
odd_count_RCIS = 0
x_rcis = time.time()
for i in range(N_RCIS):
    if i%2 == 0: #-> even case (cut dim 1)
        for s in range(n2):
            x_next_plus_RCIS = system(x_RCIS['x_max_even',even_count_RCIS,s], x_RCIS['u_even',even_count_RCIS,s],p_max);
            x_next_minus_RCIS = system(x_RCIS['x_min_even',even_count_RCIS,s], x_RCIS['u_even',even_count_RCIS,s],p_min);
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
            x_next_plus_RCIS = system(x_RCIS['x_max_odd',odd_count_RCIS,s], x_RCIS['u_odd',odd_count_RCIS,s],p_max);
            x_next_minus_RCIS = system(x_RCIS['x_min_odd',odd_count_RCIS,s], x_RCIS['u_odd',odd_count_RCIS,s],p_min);
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
for i in range(1):
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
x_rcis = time.time()
results = solver_mx_inv_set(p=vertcat(x_set,p_max,p_min),x0=opt_ro_initial, lbg=lb_g_RCIS,ubg=ub_g_RCIS,lbx=lb_opt_x_RCIS,ubx=ub_opt_x_RCIS);
y_rcis = time.time()
print("time to compute the {0}-step RCIS:".format(N_RCIS) ,y_rcis-x_rcis)
# transform the optimizer results into a structured symbolic variable
res = x_RCIS(results['x']);

# plot the simulation results (RCIS for each step and the total RCIS)
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
    
    ax[i].set_ylabel('x1',rotation = 0, fontweight = 'bold',fontsize = 24)
    ax[i].set_xlabel('x0',rotation = 0, fontweight = 'bold',fontsize = 24)
    ax[i].set_xlim([0,10])
    ax[i].set_ylim([0,10])
    ax[i].tick_params(axis='both', labelsize=24)
    ax[i].grid(False)

ax[N_RCIS].set_ylabel('x1',rotation = 0,fontweight = 'bold',fontsize = 24)
ax[N_RCIS].set_xlabel('x0',rotation = 0,fontweight = 'bold',fontsize = 24)
ax[N_RCIS].set_xlim([0,10])
ax[N_RCIS].set_ylim([0,10])
ax[N_RCIS].grid(False)
ax[N_RCIS].tick_params(axis='both', labelsize=24)
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










