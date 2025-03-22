from pyomo.environ import *
import heapq
import time
import copy

def create_relaxed_model(duration_set=1, fixed_vars=None):
    model = ConcreteModel()
    
    model.I = RangeSet(1, 7)    # jobs
    model.M = RangeSet(1, 3)    # machines
    
    r_data = {1: 2, 2: 3, 3: 4, 4: 5, 5: 10, 6: 1, 7: 2} 
    d_data = {1: 16, 2: 13, 3: 21, 4: 28, 5: 24, 6: 28, 7: 23} 
    if duration_set == 1:
        p_data = {(1, 1): 10, (1, 2): 14, (1, 3): 12,
                  (2, 1): 6, (2, 2): 8, (2, 3): 7,
                  (3, 1): 11, (3, 2): 16, (3, 3): 13,
                  (4, 1): 6, (4, 2): 12, (4, 3): 8,
                  (5, 1): 10, (5, 2): 16, (5, 3): 12,
                  (6, 1): 7, (6, 2): 12, (6, 3): 10,
                  (7, 1): 10, (7, 2): 8, (7, 3): 10}
    elif duration_set == 2:
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

    model.x = Var(model.I, model.M, bounds=(0, 1))  # Assignment of job i to machine m
    model.s = Var(model.I, domain=NonNegativeReals)  # Start time of job i
    model.c = Var(model.I, domain=NonNegativeReals)  # Completion time of job i
    model.y = Var(model.I, model.I, model.M, bounds=(0, 1))  # Precedence variable
    
    bigM = 30
        
    # Fix variables according to branching decisions
    if fixed_vars:
        for (i, m), val in fixed_vars.items():
            model.x[i, m].fix(val)
    
    def job_assignment_rule(model, i):
        return sum(model.x[i, m] for m in model.M) == 1
    model.job_assignment = Constraint(model.I, rule=job_assignment_rule)
    
    def completion_time_rule(model, i):
        return model.c[i] == model.s[i] + sum(model.x[i, m] * model.p[i, m] for m in model.M)
    model.completion_time = Constraint(model.I, rule=completion_time_rule)
    
    def release_date_rule(model, i):
        return model.s[i] >= model.r[i]
    model.release_date = Constraint(model.I, rule=release_date_rule)
    
    def due_date_rule(model, i):
        return model.c[i] <= model.d[i]
    model.due_date = Constraint(model.I, rule=due_date_rule)
    
    def job_precedence_rule(model, i, j, m):
        if i < j:
            return model.y[i, j, m] + model.y[j, i, m] >= model.x[i, m] + model.x[j, m] - 1
        return Constraint.Skip
    model.job_precedence = Constraint(model.I, model.I, model.M, rule=job_precedence_rule)
    
    def no_overlap_rule1(model, i, j, m):
        if i != j:
            return model.s[i] >= model.c[j] - bigM * (1 - model.y[i, j, m])
        return Constraint.Skip
    model.no_overlap1 = Constraint(model.I, model.I, model.M, rule=no_overlap_rule1)
    
    def no_overlap_rule2(model, i, j, m):
        if i != j:
            return model.s[j] >= model.c[i] - bigM * (1 - model.y[j, i, m])
        return Constraint.Skip
    model.no_overlap2 = Constraint(model.I, model.I, model.M, rule=no_overlap_rule2)

    def obj_rule(model):
        return sum(model.x[i, m] * model.cost[i, m] for i in model.I for m in model.M)
    model.obj = Objective(rule=obj_rule, sense=minimize)
    
    return model

def create_integer_model(duration_set=1, fixed_vars=None):
    model = create_relaxed_model(duration_set, fixed_vars)
    
    # Convert x to binary variables
    for i in model.I:
        for m in model.M:
            if not model.x[i, m].fixed:
                model.x[i, m].domain = Binary
    
    # Convert y to binary variables
    for i in model.I:
        for j in model.I:
            for m in model.M:
                model.y[i, j, m].domain = Binary
    
    return model

def solve_model(model, log_output=False):
    solver = SolverFactory('gurobi')
    
    # Set Gurobi options
    solver_options = {
        'TimeLimit': 30,  # Maximum time in seconds
        'MIPGap': 1e-4,   # MIP gap tolerance
        'OutputFlag': 1 if log_output else 0  # Control Gurobi's output
    }
    
    results = solver.solve(model, options=solver_options, tee=log_output)
    
    # Check if solution is optimal
    if results.solver.status == SolverStatus.ok and \
       results.solver.termination_condition == TerminationCondition.optimal:
        # Get objective value
        obj_value = value(model.obj)
        
        # Get variable values
        x_values = {(i, m): value(model.x[i, m]) 
                    for i in model.I for m in model.M}
        
        return True, obj_value, x_values
    else:
        return False, None, None

def find_most_fractional(variable_values):
    most_fractional_var = None
    closest_to_half = float('inf')
    
    for var, value in variable_values.items():
        # Check if variable is fractional (not 0 or 1)
        if 0.001 < value < 0.999:  # Allow for numerical precision issues
            # Calculate how close to 0.5
            distance_to_half = abs(value - 0.5)
            if distance_to_half < closest_to_half:
                most_fractional_var = var
                closest_to_half = distance_to_half
    
    return most_fractional_var

def rounding_heuristic(x_values, duration_set):
    # Round fractional variables to nearest integer
    rounded_vars = {}
    for var, val in x_values.items():
        if val >= 0.5:
            rounded_vars[var] = 1
        else:
            rounded_vars[var] = 0
    
    # Try to fix inconsistencies in the rounding
    # Ensure each job is assigned to exactly one machine
    for i in range(1, 8):
        # Count number of machines assigned to job i
        count = sum(1 for (job, m), val in rounded_vars.items() if job == i and val == 1)
        
        if count == 0:
            # Job not assigned to any machine, find the highest fractional value
            best_m = max(((i, m) for m in range(1, 4)), 
                        key=lambda im: x_values.get(im, 0))
            rounded_vars[best_m] = 1
        elif count > 1:
            # Job assigned to multiple machines, keep only the one with highest original value
            machines = [(m, x_values.get((i, m), 0)) 
                       for (job, m), val in rounded_vars.items() 
                       if job == i and val == 1]
            best_m = max(machines, key=lambda m_val: m_val[1])[0]
            
            for m in range(1, 4):
                if m != best_m:
                    rounded_vars[(i, m)] = 0
    
    # Create model with fixed binary variables
    model = create_integer_model(duration_set, rounded_vars)
    
    # Solve the model
    feasible, obj_value, final_x_values = solve_model(model)
    
    return feasible, obj_value, final_x_values

def branch_and_bound_most_fractional(duration_set=1, time_limit=300):
    # Start time
    start_time = time.time()
    
    # Initialize
    nodes_explored = 0
    branches_created = 0
    best_obj = float('inf')
    best_solution = None
    
    # Create priority queue for best-bound-first
    # Format: (bound, -node_id, fixed_variables)
    # Negative node_id to break ties by exploring older nodes first
    queue = [(0, 0, {})]
    node_id = 1
    
    print(f"Starting branch and bound with most fractional variable branching...")
    
    while queue and time.time() - start_time < time_limit:
        # Get node with best bound
        _, _, fixed_vars = heapq.heappop(queue)
        nodes_explored += 1
        
        if nodes_explored % 10 == 0:
            print(f"Explored {nodes_explored} nodes. Current best: {best_obj:.2f}")
        
        # Create and solve the relaxed model
        model = create_relaxed_model(duration_set, fixed_vars)
        feasible, obj_value, x_values = solve_model(model)
        
        # If infeasible, skip
        if not feasible or obj_value is None:
            continue
        
        # Skip if bound is worse than best solution
        if obj_value >= best_obj:
            continue
        
        # Check if integer solution
        is_integer = True
        for val in x_values.values():
            if 0.001 < val < 0.999:  # Allow for numerical precision issues
                is_integer = False
                break
        
        if is_integer:
            # We have a feasible integer solution
            if obj_value < best_obj:
                best_obj = obj_value
                best_solution = x_values
                print(f"Found new best solution: {best_obj:.2f}")
        else:
            # Apply rounding heuristic for upper bound
            heuristic_feasible, heuristic_obj, heuristic_solution = rounding_heuristic(x_values, duration_set)
            
            if heuristic_feasible and heuristic_obj < best_obj:
                best_obj = heuristic_obj
                best_solution = heuristic_solution
                print(f"Heuristic found new best solution: {best_obj:.2f}")
            
            # Find most fractional variable to branch on
            branch_var = find_most_fractional(x_values)
            
            if branch_var:
                # Branch left (variable = 0)
                left_fixed = copy.deepcopy(fixed_vars)
                left_fixed[branch_var] = 0
                heapq.heappush(queue, (obj_value, -node_id, left_fixed))
                node_id += 1
                
                # Branch right (variable = 1)
                right_fixed = copy.deepcopy(fixed_vars)
                right_fixed[branch_var] = 1
                heapq.heappush(queue, (obj_value, -node_id, right_fixed))
                node_id += 1
                
                branches_created += 2
    
    # Print final statistics
    elapsed_time = time.time() - start_time
    print(f"\nBranch and bound completed in {elapsed_time:.2f} seconds")
    print(f"Nodes explored: {nodes_explored}")
    print(f"Best objective: {best_obj:.2f}")
    
    # Convert solution to assignment
    if best_solution:
        print("\nAssignment of jobs to machines:")
        for (i, m), val in sorted(best_solution.items()):
            if val > 0.5:
                print(f"Job {i} assigned to Machine {m}")
    
    return best_obj, best_solution, nodes_explored, branches_created

# Run the branch and bound algorithm for problem set 1
print("Solving problem set 1 with most fractional variable branching rule...")
obj1, solution1, nodes1, branches_created1 = branch_and_bound_most_fractional(duration_set=1)

print("\n" + "="*50 + "\n")

# Run the branch and bound algorithm for problem set 2
print("Solving problem set 2 with most fractional variable branching rule...")
obj2, solution2, nodes2, branches_created2 = branch_and_bound_most_fractional(duration_set=2)

print("\n" + "="*50 + "\n")
print(f"Summary:")
print(f"Problem set 1: Objective = {obj1:.2f}, Branch and bonds explored in total = {branches_created1+nodes1}")
print(f"Problem set 2: Objective = {obj2:.2f}, Branch and bonds explored in total = {branches_created2+nodes2}")