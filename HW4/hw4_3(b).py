from pyomo.environ import *

data = {
    (0,0):  0,   (0,1): -10,  (0,2):  -5,   # Police choose 0, Terrorists choose 0,1,2
    (1,0): -10,  (1,1):   5,  (1,2):   1,
    (2,0):  -5,  (2,1):   1,  (2,2):   0,
}

# ============================================================
#                           Model
# ============================================================
model = ConcreteModel()

J = [0, 1, 2]
I = [0, 1, 2]

model.J = Set(initialize=J)
model.I = Set(initialize=I)

model.u = Var(model.I, domain=NonNegativeReals)
model.w = Var(domain=Reals)

def sum_u_rule(model):
    return sum(model.u[i] for i in model.I) == 1
model.sum_u_rule = Constraint(rule=sum_u_rule)

def w_rule(model, j):
    return model.w >= sum(data[(j, i)] * model.u[i] for i in model.I)
model.w_rule = Constraint(model.J, rule=w_rule)

model.obj = Objective(expr=model.w, sense=minimize)

# ============================================================
#                           Solver
# ============================================================
solver = SolverFactory('gurobi')
results = solver.solve(model, tee=True)

# ============================================================
#                           Output
# ============================================================
print(f"\nOptimal w = {model.w.value:.4f}")
print("Optimal Terrorists' mixed strategy (u):")
for i in model.I:
    print(f"  u[{i}] = {value(model.u[i]):.4f}")