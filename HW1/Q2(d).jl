using JuMP
using Gurobi
show(err)
N = [1, 2, 3, 4, 5, 6]
E = [(1,2), (1,4), (2,3), (2,5), (3,5), (3,6), (4,5), (5,6)]

# ============================================================
#                           Model
# ============================================================
model = Model(Gurobi.Optimizer)

@variable(model, x[i in N], Bin)
@variable(model, y[e in E], Bin)

# Constraints:
# 1. Maximum station installed should not over 2
@constraint(model, sum(x[i] for i in N) ≤ 2)

# 2. Linear inequality constraint: if both ends of the edges remain unstalled (x[u] = x[v] = 0), 
# edge should be uncovered (y[u, v] = 0)
for (u, v) in E
    @constraint(model, y[(u, v)] ≤ (x[u] + x[v]))
end

# Objective: minizing the uncoverage
@objective(model, Min, sum(1 - y[e] for e in E))

# ============================================================
#                           Solver
# ============================================================
optimize!(model)


# ============================================================
#                           Output
# ============================================================
println("Number of uncovered edges= ", objective_value(model))
for i in N
    if value(x[i]) ≥ 1
        println("Monitor is installed at node $i")
    end
end