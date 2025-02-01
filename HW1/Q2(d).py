from pyomo.environ import *

N = [1, 2, 3, 4, 5, 6]
E = [(1,2), (1,4), (2,3), (2,5), (3,5), (3,6), (4,5), (5,6)]

# ============================================================
#                           Model
# ============================================================
model = ConcreteModel()
model.N = Set(initialize=N)
model.E = Set(dimen=2, initialize=E)

# Variables: binary x for installing or not, binary y: 1 for edge being covered, 0 for left uncovered
model.x = Var(model.N, domain=Binary)
model.y = Var(model.E, domain=Binary)

# Constraints:
# 1. Maximum station installed should not over 2
def station_rule(model):
    return sum(model.x[i] for i in N) <= 2
model.station_constraint = Constraint(rule=station_rule)

# 2. Linear inequality constraint: if both ends of the edges remain unstalled (x[u] = x[v] = 0), 
# edge should be uncovered (y[u, v] = 0)
def coverage_rule(model, u, v):
    return model.y[u, v] <= (model.x[u] + model.x[v])
model.coverage_contraint = Constraint(model.E, rule=coverage_rule)

# Objective: minizing the uncoverage
model.obj = Objective(
    expr = sum(1 - model.y[e] for e in E),
    sense=minimize
)

# ============================================================
#                           Solver
# ============================================================
solver = SolverFactory("gurobi")
solver.solve(model, tee=True)

# ============================================================
#                           Output
# ============================================================
print("Number of uncovered edges = ", model.obj())
for i in model.N:
    if model.x[i].value >= 1:
        print(f"Install station at node {i}")