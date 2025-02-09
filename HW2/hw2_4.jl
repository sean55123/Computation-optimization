using JuMP
using Gurobi
using Graphs
using GraphPlot

# ============================================================
#                           Model
# ============================================================
V = ["s1", "s2", "s3", "x", "y", "t1", "t2", "t3"]
E = [("s1", "x"), ("s2", "x"), ("s3", "x"),
     ("x", "y"),
     ("y", "t1"), ("y", "t2"), ("y", "t3"),
     ("s1", "t1"),
     ("x", "t3")]
K = ["1", "2", "3"]

# Capacity and cost data for each edge
capacity = Dict(
    ("s1", "x") => 5, ("s2", "x") => 3, ("s3", "x") => 5,
    ("x", "y") => 14,
    ("y", "t1") => 3, ("y", "t2") => 3, ("y", "t3") => 4,
    ("s1", "t1") => 7,
    ("x", "t3") => 5
)

cost = Dict(
    ("s1", "x") => 1, ("s2", "x") => 2, ("s3", "x") => 2,
    ("x", "y") => 2,
    ("y", "t1") => 1, ("y", "t2") => 2, ("y", "t3") => 2,
    ("s1", "t1") => 3,
    ("x", "t3") => 5
)

# Demand, source, and sink information for each commodity
demand = Dict("1" => 9, "2" => 2, "3" => 3)
source = Dict("1" => "s1", "2" => "s2", "3" => "s3")
sink   = Dict("1" => "t1", "2" => "t2", "3" => "t3")

model = Model(Gurobi.Optimizer)

# Define nonnegative flow variables f[k,e] for each commodity k and edge e.
@variable(model, flow[k in K, e in E] >= 0)

# Constraint
# Flow balance constraints: For every commodity k and node n, enforce:
#   (outflow - inflow) == demand (at the source),
#   (outflow - inflow) == -demand (at the sink),
#   (outflow - inflow) == 0 (at intermediate nodes).
for k in K, n in V
    outflow = sum(flow[k, (n, j)] for (i, j) in E if i == n; init=0)
    inflow  = sum(flow[k, (i, n)] for (i, j) in E if j == n; init=0)
    if n == source[k]
        @constraint(model, outflow - inflow == demand[k])
    elseif n == sink[k]
        @constraint(model, outflow - inflow == -demand[k])
    else
        @constraint(model, outflow - inflow == 0)
    end
end

# Capacity constraints: the total flow (summed over commodities) on each edge
for e in E
    @constraint(model, sum(flow[k, e] for k in K) <= capacity[e])
end

# Objective: minimize the total cost (sum over all commodities and edges).
@objective(model, Min, sum(cost[e] * flow[k, e] for k in K, e in E))


# ============================================================
#                           Solver
# ============================================================
optimize!(model)

println("\nSolver Status: ", termination_status(model))
println("Objective (Total cost): ", objective_value(model))


# Collect flows along each edge (aggregating over commodities).
flow_dict = Dict{Tuple{String, String}, Float64}()

for k in K, e in E
    val = value(flow[k, e])
    if val > 1e-6  # consider a tolerance for nonzero flows
        println("$(val) amount of commodity $k has been sent via route $(e[1]) to $(e[2])")
        # Sum flows over the same edge (if multiple commodities use it).
        flow_dict[e] = get(flow_dict, e, 0.0) + val
    end
end