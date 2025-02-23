from pyomo.environ import *

# Constructs the graph basing on the problem
nodes = ["s", "v1", "v2", "v3", "v4", "t"]

edges = [
    ("s", "v1"),
    ("s", "v2"),
    ("v1", "v2"),
    ("v1", "v3"),
    ("v2", "v1"),
    ("v2", "v4"),
    ("v3", "v2"),
    ("v3", "t"),
    ("v4", "v3"),
    ("v4", "t")
]

capacity = {
    ("s", "v1"): 16,
    ("s", "v2"): 13,
    ("v1", "v2"): 10,
    ("v1", "v3"): 12,
    ("v2", "v1"): 4,
    ("v2", "v4"): 14,
    ("v3", "v2"): 9,
    ("v3", "t"): 20,
    ("v4", "v3"): 7,
    ("v4", "t"): 4
}

source = "s"
sink = "t"

# ============================================================
#                           Model
# ============================================================
model = ConcreteModel()
model.Nodes = Set(initialize=nodes)
model.Edges = Set(initialize=edges, dimen=2)
model.f = Var(model.Edges, domain=NonNegativeReals)

# Constraint for the capcity
def capacity_rule(model, u, v):
    return model.f[u, v] <= capacity[(u, v)]
model.capcity_constraint = Constraint(model.Edges, rule=capacity_rule)

# Constraint for overall conservation
def flow_conservation_rule(model, v):
    if v == source or v == sink:
        return Constraint.Skip  # Skip source/sink in conservation
    inflow  = sum(model.f[u, v] for (u, v_in) in model.Edges if v_in == v)
    outflow = sum(model.f[v, w] for (v_out, w) in model.Edges if v_out == v)
    return inflow == outflow
model.flow_conservation = Constraint(model.Nodes, rule=flow_conservation_rule)


def obj_rule(model):
    return sum(model.f[u, v] for (u, v) in model.Edges if u == source)
model.Obj = Objective(rule=obj_rule, sense=maximize)

# ============================================================
#                           Solver
# ============================================================
solver = SolverFactory('gurobi')
results = solver.solve(model, tee=True)

# ============================================================
#                           Output
# ============================================================
print(results)

print("Maximum flow value:", value(model.Obj))

print("Flow on each edge:")
for (u, v) in model.Edges:
    print(f"f({u}->{v}) = {value(model.f[u,v])}")