using JuMP
using Gurobi

# ============================================================
#                           Model
# ============================================================
model = Model(Gurobi.Optimizer)

@variable(model, Qs >= 0)
@variable(model, Qw >= 0)
@variable(model, R[1:3] >= 0)

@objective(model, Min, Qs + Qw)

@constraint(model, Qs - R[1] == 30)
@constraint(model, R[1] - R[2] == 30)
@constraint(model, R[2] - R[3] == -123)
@constraint(model, R[3] - Qw == -102)

# ============================================================
#                           Solver
# ============================================================
optimize!(model)

# ============================================================
#                           Output
# ============================================================
println("Optimal Qs: ", value(Qs), "MW")
println("Optimal Qw: ", value(Qw), "MW")
for i in 1:3
    println("R",i, " value: ", value(R[i]), "MW")
end