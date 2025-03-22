from pyomo.environ import *

model = ConcreteModel()

model.I = RangeSet(1, 7)    # jobs
model.M = RangeSet(1, 3)    # machines

r_data = {1: 2, 2: 3, 3: 4, 4: 5, 5: 10, 6: 1, 7: 2} 
d_data = {1: 16, 2: 13, 3: 21, 4: 28, 5: 24, 6: 28, 7: 23} 
p_data = {(1, 1): 5, (1, 2): 7, (1, 3): 6,
          (2, 1): 3, (2, 2): 4, (2, 3): 3,
          (3, 1): 2, (3, 2): 4, (3, 3): 3,
          (4, 1): 3, (4, 2): 6, (4, 3): 4,
          (5, 1): 2, (5, 2): 4, (5, 3): 3,
          (6, 1): 1, (6, 2): 3, (6, 3): 2,
          (7, 1): 1, (7, 2): 2, (7, 3): 1}  
c_data = {(1, 1): 10, (1, 2): 6, (1, 3): 8,
          (2, 1): 8, (2, 2): 5, (2, 3): 6,
          (3, 1): 12, (3, 2): 7, (3, 3): 10,
          (4, 1): 10, (4, 2): 6, (4, 3): 8,
          (5, 1): 8, (5, 2): 5, (5, 3): 7,
          (6, 1): 12, (6, 2): 7, (6, 3): 10,
          (7, 1): 12, (7, 2): 7, (7, 3): 10}       

model.r = Param(model.I, initialize=r_data)
model.d = Param(model.I, initialize=d_data)
model.p = Param(model.I, model.M, initialize=p_data)
model.cost = Param(model.I, model.M, initialize=c_data)

# Decision variables
model.x = Var(model.I, model.M, domain=Binary)  # Assignment of job i to machine m
model.s = Var(model.I, domain=NonNegativeReals)  # Start time of job i
model.c = Var(model.I, domain=NonNegativeReals)  # Completion time of job i
model.y = Var(model.I, model.I, model.M, domain=Binary)  # Precedence variable


# 1) Each job assigned to exactly one machine
def one_machine_rule(model, i):
    return sum(model.x[i,m] for m in model.M) == 1
model.one_machine = Constraint(model.I, rule=one_machine_rule)

# 2) Completion time = start + chosen duration
def comp_time_rule(model, i):
    return model.c[i] == model.s[i] + sum(model.p[i,m]*model.x[i,m] for m in model.M)
model.comp_time = Constraint(model.I, rule=comp_time_rule)

# 3) Respect release date and due date
def release_rule(model, i):
    return model.s[i] >= model.r[i]
model.releaseC = Constraint(model.I, rule=release_rule)

def due_rule(model, i):
    return model.c[i] <= model.d[i]
model.dueC = Constraint(model.I, rule=due_rule)

# 4) No overlap constraints - linearized
bigM = 30  # A suitably large number

# If both jobs are on the same machine, one must precede the other
def job_precedence_rule(model, i, j, m):
    if i < j:  # To avoid redundant constraints
        return model.y[i,j,m] + model.y[j,i,m] >= model.x[i,m] + model.x[j,m] - 1
    return Constraint.Skip
model.job_precedence = Constraint(model.I, model.I, model.M, rule=job_precedence_rule)

# Enforce start times based on precedence
def no_overlap_rule1(model, i, j, m):
    if i != j:
        return model.s[i] >= model.c[j] - bigM * (1 - model.y[i,j,m])
    return Constraint.Skip
model.no_overlap1 = Constraint(model.I, model.I, model.M, rule=no_overlap_rule1)

def no_overlap_rule2(model, i, j, m):
    if i != j:
        return model.s[j] >= model.c[i] - bigM * (1 - model.y[j,i,m])
    return Constraint.Skip
model.no_overlap2 = Constraint(model.I, model.I, model.M, rule=no_overlap_rule2)

# Objective function
def obj_rule(model):
    return sum(model.cost[i,m]*model.x[i,m] for i in model.I for m in model.M)
model.obj = Objective(rule=obj_rule, sense=minimize)

solver = SolverFactory('gurobi')
results = solver.solve(model, tee=True)

assignments = [(i, m) for i in model.I for m in model.M if value(model.x[i,m]) > 0.5]
start_times = {i: value(model.s[i]) for i in model.I}
completion_times = {i: value(model.c[i]) for i in model.I}
total_cost = value(model.obj)

print("Total cost:", total_cost)
print("\nJob assignments:")
for i, m in assignments:
    print(f"Job {i} assigned to Machine {m}")
    print(f"  Start time: {start_times[i]}")
    print(f"  Completion time: {completion_times[i]}")
    print(f"  Duration: {p_data[(i,m)]}")