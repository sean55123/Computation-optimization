from pyomo.environ import *

model = ConcreteModel()

t = [1, 2, 3, 4, 5, 6]
model.t = RangeSet(1, 6)

d  = {1:10, 2:40, 3:20, 4:5, 5:5,  6:15}  # demand
f  = {1:50, 2:50, 3:50, 4:50,5:50, 6:50}  # fixed cost
p  = {1:1,  2:3,  3:3,  4:1, 5:1,  6:1}   # prod cost
h  = {1:2,  2:2,  3:2,  4:2, 5:2,  6:2}   # holding cost
C  = 25    # capacity

model.x = Var(model.t, domain=NonNegativeReals) # Amount of production
model.y = Var(model.t, domain=Binary) # Setup for specific period
model.I = Var(model.t, domain=NonNegativeReals) # Inventory

def inventory_balance_rule(model, t):
    if t == 1:
        return model.x[1] == d[1] + model.I[1]
    else:
        return model.I[t - 1] + model.x[t] == d[t] + model.I[t]
model.inventory_balance = Constraint(model.t, rule=inventory_balance_rule)

def production_limit_rule(model, t):
    return model.x[t] <= C*model.y[t]
model.production_limit = Constraint(model.t, rule=production_limit_rule)

def total_cost_rule(model):
    return sum(f[t]*model.y[t] + p[t]*model.x[t] + h[t]*model.I[t] for t in model.t)
model.Obj = Objective(rule=total_cost_rule, sense=minimize)

solver = SolverFactory("gurobi")
results = solver.solve(model, tee=True)


print(value(model.Obj))
for t in model.t:
    print(
        f"Period {t}: Produce {model.x[t].value:.1f}, "
        f"Inventory {model.I[t].value:.1f}, "
        f"Setup y={(model.y[t].value)}"
    )