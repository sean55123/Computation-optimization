from pyomo.environ import *
import networkx as nx
import matplotlib.pyplot as plt

# ============================================================
#                           Model
# ============================================================
model = ConcreteModel()

model.V = Set(initialize=["s1","s2","s3","x","y","t1","t2","t3"])  # nodes
model.E = Set(initialize=[
    ("s1","x"), ("s2","x"), ("s3","x"),
    ("x","y"),
    ("y","t1"), ("y","t2"), ("y","t3"),
    ("s1","t1"),
    ("x","t3")
])
model.K = Set(initialize=["1","2","3"])  # commodities

capacity_data = {
    ("s1","x"):5, ("s2","x"):3, ("s3","x"):5,
    ("x","y"):14,
    ("y","t1"):3, ("y","t2"):3, ("y","t3"):4,
    ("s1","t1"):7,
    ("x","t3"):5
}
cost_data = {
    ("s1","x"):1, ("s2","x"):2, ("s3","x"):2,
    ("x","y"):2,
    ("y","t1"):1, ("y","t2"):2, ("y","t3"):2,
    ("s1","t1"):3,
    ("x","t3"):5
}

# Demand and source/sink for each commodity
demand = {"1": 9, "2": 2, "3": 3}
source = {"1": "s1", "2": "s2", "3": "s3"}
sink   = {"1": "t1", "2": "t2", "3": "t3"}

model.capacity = Param(model.E, initialize=capacity_data)
model.cost     = Param(model.E, initialize=cost_data)

model.f = Var(model.K, model.E, within=NonNegativeReals)

# Constraints
# 1. The overall flow balance
def flow_balance_rule(model, k, n):
    inflow  = sum(model.f[k, (i, n)] for (i, n2) in model.E if n2 == n)
    outflow = sum(model.f[k, (n, j)] for (n2, j) in model.E if n2 == n)
    if n == source[k]:
        return outflow - inflow == demand[k]
    elif n == sink[k]:
        return outflow - inflow == -demand[k]
    else:
        return outflow - inflow == 0

model.flow_balance = Constraint(model.K, model.V, rule=flow_balance_rule)

# 2. Capacity requirement
def capacity_rule(model, i, j):
    return sum(model.f[k, (i, j)] for k in model.K) <= model.capacity[(i, j)]

model.capacity_constr = Constraint(model.E, rule=capacity_rule)

# Objective function minimizing expenditure cost by each edge
def objective_rule(model):
    return sum(
        model.cost[(i, j)] * model.f[k, (i, j)] for k in model.K for (i, j) in model.E
    )

model.obj = Objective(rule=objective_rule, sense=minimize)

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
print("\nObjective (Total cost):", model.obj())

flow_dict = {}

for i in model.f:
    key = (i[0], (i[1], i[2]))
    if model.f[i].value != 0:
        print(f'{model.f[i].value} amount of commedity {i[0]} have been sent via route {i[1]} to {i[2]}')
        flow_dict[key] = model.f[i].value

G = nx.DiGraph()

nodes = ["s1","s2","s3","x","y","t1","t2","t3"]
G.add_nodes_from(nodes)
edge_flows = {}
for (k, (i,j)), val in flow_dict.items():
    if val > 1e-9:
        edge_flows[(i,j)] = edge_flows.get((i,j), 0) + val

for (i,j), flow_val in edge_flows.items():
    G.add_edge(i, j, flow=flow_val)

pos = {
    "s1": (0,2),
    "s2": (0,1),
    "s3": (0,0),
    "x":  (1,1),
    "y":  (2,1),
    "t1": (3,2),
    "t2": (3,1),
    "t3": (3,0),
}

plt.figure(figsize=(8,4))
nx.draw_networkx_nodes(G, pos, node_size=400, node_color="lightblue")
nx.draw_networkx_labels(G, pos)
nx.draw_networkx_edges(
    G, pos,
    arrows=True,
    arrowstyle="-|>",
    connectionstyle="arc3,rad=0.0",
    width=2
)

edge_labels = {(u,v): f"{d['flow']:.1f}" for (u,v,d) in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.axis("off")
plt.title("Multi-Commodity Flow")
plt.show()