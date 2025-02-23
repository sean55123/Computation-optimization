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
model.x = Var(model.Nodes, domain=Reals)
model.y = Var(model.Edges, domain=NonNegativeReals)

def cut_rule(model, u, v):
    return model.x[u] - model.x[v] + model.y[u, v] >= 0 
model.cut_constraint = Constraint(model.Edges, rule=cut_rule)

def source_sink_rule(model):
    return -model.x["s"] + model.x["t"] >= 1
model.source_sink_constraint = Constraint(rule=source_sink_rule)
   
def obj_rule(model):
    return sum(capacity[e] * model.y[e] for e in model.Edges)
model.obj = Objective(rule=obj_rule, sense=minimize)

# ============================================================
#                           Solver
# ============================================================
solver = SolverFactory('gurobi')
results = solver.solve(model, tee=True)

# ============================================================
#                           Output
# ============================================================
print(results)

print("\nMinimum Cut Value:", model.obj())

print("\nEdges in the Min-Cut:")
for e in model.Edges:
    if value(model.y[e]) > 0.9:  # Check for values close to 1 (due to solver tolerance)
        print(f"  Edge {e}: Capacity = {capacity[e]}")

print("\nNode Partitioning:")
for n in model.Nodes:
    print(f"  Node {n}: {value(model.x[n])}")