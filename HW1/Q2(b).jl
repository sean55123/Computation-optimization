using JuMP
using Gurobi   

N = [1, 2, 3, 4, 5, 6]
E = [(1,2), (1,4), (2,3), (2,5), (3,5), (3,6), (4,5), (5,6)]
cost = Dict(1=>40, 2=>65, 3=>43, 4=>48, 5=>72, 6=>36)

# ============================================================
#                           Model
# ============================================================
model = Model(Gurobi.Optimizer)

# Decision variable: 1 for installing the monitor, 0 for not
@variable(model, x[i in N], Bin)

# Objective function: summation of all the installation costs
@objective(model, Min, sum(cost[i]*x[i] for i in N))

# Constraint: at least one node linked by the edge should install monitor
for (u, v) in E
    @constraint(model, x[u]+x[v] ≥ 1)
end

# ============================================================
#                           Solver
# ============================================================
optimize!(model)

# ============================================================
#                           Output
# ============================================================
println("Optimal objective value: ", objective_value(model))
for i in N
    if value(x[i]) ≥ 1
        println("Monitor installed at mode $i")
    end
end