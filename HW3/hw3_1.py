from pyomo.environ import *

# ============================================================
#                           Model
# ============================================================
model = ConcreteModel()

model.Qs = Var(domain=NonNegativeReals)
model.Qw = Var(domain=NonNegativeReals)
model.R = Var(range(1, 4), domain=NonNegativeReals)

model.objective = Objective(expr=model.Qs + model.Qw, sense=minimize)

model.constraint1 = Constraint(expr = model.Qs - model.R[1] == 30)
model.constraint2 = Constraint(expr = model.R[1] - model.R[2] == 30)
model.constraint3 = Constraint(expr = model.R[2] - model.R[3] == -123)
model.constraint4 = Constraint(expr = model.R[3] - model.Qw == -102)

# ============================================================
#                           Solver
# ============================================================
solver = SolverFactory('gurobi')
results = solver.solve(model, tee=True)

# ============================================================
#                           Output
# ============================================================
print(f"Optimal Qh: {value(model.Qs):.2f} MW")
print(f"Optimal Qc: {value(model.Qw):.2f} MW")
for i in range(1, 4):
    print(f"R[{i}]: {value(model.R[i]):.2f} MW")