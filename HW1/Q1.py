from pyomo.environ import *

num_engineers = 3
num_projects  = 4

cap = {
    (1, 1): 90, (1, 2): 80, (1, 3): 10, (1, 4): 50,
    (2, 1): 60, (2, 2): 70, (2, 3): 50, (2, 4): 65,
    (3, 1): 70, (3, 2): 40, (3, 3): 80, (3, 4): 85
}

required = {1: 70, 2: 50, 3: 85, 4: 35}
max_hours = 80

# ============================================================
#                           Model
# ============================================================

model = ConcreteModel()

# Sets for engineers and projects
model.engineers = RangeSet(1, num_engineers)
model.projects = RangeSet(1, num_projects)

# Decision variables: x[i,j], hours designer i spends on project j
model.x = Var(model.engineers, model.projects, domain=NonNegativeReals)

# Objective function: maximize total score * hours
model.obj = Objective(
    expr=sum(cap[i, j] * model.x[i, j] for i in model.engineers for j in model.projects),
    sense=maximize
)

# Constraints:
# 1. Each project must meet its required hours
def project_hours_rule(model, j):
    return sum(model.x[i, j] for i in model.engineers) == required[j]
model.project_hours = Constraint(model.projects, rule=project_hours_rule)

# 2. Each designer can work at most 80 hours in total
def engineers_hours_rule(model, i):
    return sum(model.x[i, j] for j in model.projects) <= max_hours
model.engineers_hours = Constraint(model.engineers, rule=engineers_hours_rule)

# ============================================================
#                           Solver
# ============================================================
solver = SolverFactory('gurobi')
result = solver.solve(model, tee=True)

# ============================================================
#                           Output
# ============================================================
print("\nSolver Status:", result.solver.status)
print("Termination Condition:", result.solver.termination_condition)
print("\nObjective (Total Capability):", model.obj())

for i in model.engineers:
    for j in model.projects:
        print(f"Engineer {i} on Project {j}: {model.x[i, j].value:.2f} hours")