from pyomo.environ import *

N = [1, 2, 3, 4, 5, 6]
E = [(1,2), (1,4), (2,3), (2,5), (3,5), (3,6), (4,5), (5,6)]
cost = {
    1: 40,
    2: 65,
    3: 43,
    4: 48,
    5: 72,
    6: 36
}

# ============================================================
#                           Model
# ============================================================
model = ConcreteModel()

# Sets for nodes, edges, and costs
model.N = Set(initialize=N)
model.E = Set(dimen=2, initialize=E)
model.cost = {i: cost[i] for i in N}

# Decision variable: 1 for installing the monitor, 0 for not
model.x = Var(model.N, domain=Binary)


# Objective function: summation of all the installation costs
model.obj = Objective(
    expr=sum(cost[i] * model.x[i] for i in N),
    sense=minimize
)

# Constraint: at least one node linked by the edge should install monitor
def coverage_rule(model, u, v):
    return model.x[u] + model.x[v] >= 1
model.coverage_constraint = Constraint(model.E, rule=coverage_rule) 

# ============================================================
#                           Solver
# ============================================================
solver = SolverFactory('gurobi')
solver.solve(model, tee=True)

# ============================================================
#                           Output
# ============================================================
print("Optimal solution cost = ", model.obj())
for i in model.N:
    if model.x[i].value > 0.9:
        print(f"Install station at node {i}")