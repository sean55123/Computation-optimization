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

model.x = Var(model.J, domain=NonNegativeReals)
model.u = Var(domain=Reals)

def prob_rule(model):
    return sum(model.x[i] for i in model.J) == 1
model.prob_rule = Constraint(rule=prob_rule)

def worst_case_rule(model, i):
    return sum(data[(j, i)] * model.x[j] for j in model.J) >= model.u
model.worst_case_rule = Constraint(model.I, rule=worst_case_rule)

model.obj = Objective(expr=model.u, sense=maximize)

# ============================================================
#                           Solver
# ============================================================
solver = SolverFactory('gurobi')
results = solver.solve(model, tee=True)

# ============================================================
#                           Output
# ============================================================
print(f"\nOptimal value of the game = {model.u.value:.4f}")
print("Optimal Police mixed strategy:")
for i in model.I:
    print(f"  x[{i}] = {model.x[i].value:.4f}")