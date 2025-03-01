from pyomo.environ import *
import numpy as np

# ============================================================
#                           Model
# ============================================================
model = ConcreteModel()

model.r = Var(initialize=1, bounds=(0, None))
model.h = Var(initialize=1, bounds=(0, None))

# Overall volumn constraint
model.volume_constraint = Constraint(
    expr= np.pi * model.r**2 * model.h == 25
)

cost_side   = 150.0
cost_top    = 190.0
cost_bottom = 260.0


model.cost = Objective(
    expr = cost_side * (2 * np.pi * model.r * model.h)
         + cost_top  * (np.pi * model.r**2)
         + cost_bottom * (np.pi * model.r**2),
    sense = minimize
)

# ============================================================
#                           Solver
# ============================================================
solver = SolverFactory('ipopt')
results = solver.solve(model, tee=True)

# ============================================================
#                           Output
# ============================================================
print("Status =", results.solver.status)
print("Termination condition =", results.solver.termination_condition)
print(f"Optimal radius (r)  = {value(model.r):.4f} m")
print(f"Optimal height (h)  = {value(model.h):.4f} m")
print(f"Minimum cost        = ${value(model.cost):.2f}")