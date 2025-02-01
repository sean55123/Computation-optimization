using JuMP
using Gurobi   

num_engineerss = 3
num_projects  = 4

cap = [
    [90, 80, 10, 50],  # engineers 1
    [60, 70, 50, 65],  # engineers 2
    [70, 40, 80, 85]   # engineers 3
]
required = [70, 50, 85, 35]  # required hours for each project
max_hours = 80

# ============================================================
#                           Model
# ============================================================
model = Model(Gurobi.Optimizer)

# Decision variables x[i,j]
@variable(model, x[1:num_engineerss, 1:num_projects] >= 0)

# Objective: maximize sum of cap[i][j] * x[i,j]
@objective(model, Max, sum(cap[i][j]*x[i,j] for i in 1:num_engineerss, j in 1:num_projects))

# Constraints:
# 1) Each project j requires exactly required[j] total hours
for j in 1:num_projects
    @constraint(model, sum(x[i,j] for i in 1:num_engineerss) == required[j])
end

# 2) Each engineers i can use at most 80 hours
for i in 1:num_engineerss
    @constraint(model, sum(x[i,j] for j in 1:num_projects) <= max_hours)
end

# ============================================================
#                           Solver
# ============================================================
optimize!(model)

# ============================================================
#                           Output
# ============================================================
println("Objective value (Total Capability): ", objective_value(model))
println("Decision variables x[i,j]:")
for i in 1:num_engineerss
    for j in 1:num_projects
        println("Engineer: $i spend on project: $j = ", value(x[i,j]))
    end
end