from pyomo.environ import *
from pyomo.opt import TerminationCondition

# ============================================================
#                           Model
# ============================================================
model = ConcreteModel()

edge = [
    (1,3), (1,4), (1,8), (2,3), (2,4), (2,8), (3,4), 
    (3,5), (3,6), (3,7), (4,3), (4,5), (4,6), (4,7)
]

model.N = RangeSet(1,8)  
model.E = Set(initialize=edge)

cost_data = {(1,3):7, (1,4):8, (1,8):0, (2,3):4, (2,4):7, (2,8):0, (3,4):0, (3,5):25, (3,6):6, (3,7):17, (4,3):1e-6, (4,5):29, (4,6):8, (4,7):5}
cap_data  = {(1,3):9999, (1,4):9999, (1,8):9999,  (2,3):9999, (2,4):9999, (2,8):9999, (3,4):25, (3,5):9999, (3,6):9999, (3,7):9999, (4,3):25, (4,5):9999, (4,6):9999, (4,7):9999} 
demand_data = {1:1000, 2:1000, 3:0, 4:0, 5:-450, 6:-500, 7:-610, 8:-440}

model.cost    = Param(model.E, initialize=cost_data, default=0)
model.capacity = Param(model.E, initialize=cap_data, default=None)
model.demand   = Param(model.N, initialize=demand_data)


model.flow = Var(model.E, within=NonNegativeReals)

# Capacity constraint
def capacity_rule(model, i, j):
    if model.capacity[i, j] is None:
        return Constraint.Skip
    return model.flow[i,j] <= model.capacity[i,j]
model.CapacityConstraint = Constraint(model.E, rule=capacity_rule)

# Flow balance
def flow_balance_rule(model, i):
        return (
        sum(model.flow[i, j] for (i2, j) in model.E if i2 == i)
        - sum(model.flow[j, i2] for (j, i2) in model.E if i2 == i)
        == model.demand[i]
    )
model.FlowBalance = Constraint(model.N, rule=flow_balance_rule)

# Minimize overall cost
def obj_rule(model):
    return sum(model.cost[i,j] * model.flow[i,j] for (i,j) in model.E)
model.obj = Objective(rule=obj_rule, sense=minimize)

# ============================================================
#                           Solver
# ============================================================
solver = SolverFactory('gurobi')
solver.options['Method'] = 2 
solver.options['Crossover'] = 0  
results = solver.solve(model, tee=True)

# ============================================================
#                           Output
# ============================================================
if (results.solver.termination_condition 
        == TerminationCondition.optimal):
    print("Optimal solution found")
    for (i,j) in model.E:
        val = model.flow[i,j].value
        if val > 1e-6:
            print(f"Flow on ({i}->{j}) = {val}")
else:
    print("No optimal solution found")
