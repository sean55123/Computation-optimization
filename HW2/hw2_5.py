from pyomo.environ import *

# ============================================================
#                           Model
# ============================================================
model = ConcreteModel()

model.K = RangeSet(0, 3) 
n_x = 3  # dimension of state
n_u = 2  # dimension of input

def x_bounds(i):
    if i == 0: return (-0.05, 0.05)  
    if i == 1: return (-5.0, 5.0)    
    if i == 2: return (-0.5, 0.5)    


def u_bounds(j):
    if j == 0: return (-10.0, 10.0)  
    if j == 1: return (-0.05, 0.05)   

model.x = Var(model.K, range(n_x), bounds=lambda m, k, i: x_bounds(i))
model.u = Var(model.K, range(n_u), bounds=lambda m, k, j: u_bounds(j))
model.y = Var(model.K, range(n_x)) 

model.eps_c = Var(model.K, within=NonNegativeReals)
model.eps_h = Var(model.K, within=NonNegativeReals)

A = [[ 0.2681, -0.00338, -0.00728],
     [ 9.703,   0.3279,  -25.44   ],
     [ 0,       0,         1      ]]
B = [[-0.00537,  0.1655],
     [ 1.297,   97.91   ],
     [ 0,       -6.637  ]]
C = [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]]

# Initial deviation from steady state
x0 = [-0.03, 0.0, 0.3]

# Setpoints for c and h are zero in the shifted coordinates
c_sp = 0.0
h_sp = 0.0


# Constraints
# 1. Initial condition
def init_rule(m, i):
    return m.x[0, i] == x0[i]
model.initcon = Constraint(range(n_x), rule=init_rule)

# 2. State update: x(k+1) = A x(k) + B u(k)
def x_state_update_rule(m, k, i):
    if k == m.K.last():  
        return Constraint.Skip
    return m.x[k+1, i] == sum(A[i][j]*m.x[k, j] for j in range(n_x)) \
                       + sum(B[i][j]*m.u[k, j] for j in range(n_u))
model.x_state_update = Constraint(model.K, range(n_x), rule=x_state_update_rule)

# 3. Output equation y(k) = C x(k) = x(k)  (since C=I)
def y_def_rule(m, k, i):
    return m.y[k, i] == sum(C[i][j]*m.x[k, j] for j in range(n_x))
model.y_def = Constraint(model.K, range(n_x), rule=y_def_rule)

# 4. Absolute value constraints for controlling c and h
def abs_c_pos_rule(m, k):
    return m.eps_c[k] >= m.y[k, 0] - c_sp
model.abs_c_pos = Constraint(model.K, rule=abs_c_pos_rule)

def abs_c_neg_rule(m, k):
    return m.eps_c[k] >= -(m.y[k, 0] - c_sp)
model.abs_c_neg = Constraint(model.K, rule=abs_c_neg_rule)

def abs_h_pos_rule(m, k):
    return m.eps_h[k] >= m.y[k, 2] - h_sp
model.abs_h_pos = Constraint(model.K, rule=abs_h_pos_rule)

def abs_h_neg_rule(m, k):
    return m.eps_h[k] >= -(m.y[k, 2] - h_sp)
model.abs_h_neg = Constraint(model.K, rule=abs_h_neg_rule)

# Objective function
def obj_rule(m):
    return sum(m.eps_c[k] + m.eps_h[k] for k in m.K)
model.obj = Objective(rule=obj_rule, sense=minimize)

# ============================================================
# Solve
# ============================================================
solver = SolverFactory('gurobi')
result = solver.solve(model, tee=True)

# ============================================================
# Output
# ============================================================
print(result)
print('Optimal objective:', value(model.obj))

for k in model.K:
    x_vals = [value(model.x[k,i]) for i in range(n_x)]
    print(f'Time {k}, x(k)={x_vals}')
for k in range(3):
    u_vals = [value(model.u[k,j]) for j in range(n_u)]
    eps_c_k = value(model.eps_c[k])
    eps_h_k = value(model.eps_h[k])
    print(f'Time {k}, u(k)={u_vals}, eps_c={eps_c_k}, eps_h={eps_h_k}')