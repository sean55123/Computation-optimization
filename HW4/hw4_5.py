from pyomo.environ import *
import numpy as np

Q = np.array([[0,1,3,1],
              [1,0,0,2],
              [3,0,0,0],
              [1,2,0,0]])  

n = Q.shape[0]

# ============================================================
#                           Model
# ============================================================
model = ConcreteModel()

model.I = RangeSet(0, n-1)
model.J = RangeSet(0, n-1) 

model.X = Var(model.I, model.J, domain=Reals, bounds=(-1, 1))

# Force the summation of diagonal part to be 1
def diag_rule(model, i):
    return model.X[i, i] == 1
model.diag_constraints = Constraint(model.I, rule=diag_rule)

# Force this matrix to be symmtric -> Keep the matrix PSD
def sym_rule(model, i, j):
    return model.X[i, j] == model.X[j, i]
model.sym_constraints = Constraint(model.I, model.J, rule=sym_rule)

# Objective function: sum_{i != j} Q_{ij} (1 - X_{ij})
def obj_rule(model):
    return sum(Q[i, j] * (1 - model.X[i, j]) 
               for i in range(n) for j in range(n) if i != j)
model.obj = Objective(rule=obj_rule, sense=maximize)

# Function that adds a cut v^T X v >= 0 for a given eigenvector v
def add_psd_cut(pyomo_model, eigvec, cut_name):
    expr = 0
    for i in range(n):
        for j in range(n):
            expr += eigvec[i]*eigvec[j]*pyomo_model.X[i, j]
    
    # Constraint the objective with the cut
    c = Constraint(expr=expr >= 0)
    setattr(pyomo_model, cut_name, c)

# ============================================================
#                           Solver
# ============================================================
solver = SolverFactory('gurobi')  

max_iters = 50
tol = 1e-4

for it in range(max_iters):
    # Solve the current relaxation
    res = solver.solve(model, tee=False)
    
    # Extract solution X
    X_val = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            X_val[i, j] = value(model.X[i, j])
    
    # Check eigenvalues
    w, V = np.linalg.eigh(X_val)
    min_eig = w[0]
    print(f"Iteration {it}, min eigenvalue = {min_eig:.6f}, objective = {value(model.obj):.4f}")
    
    if min_eig >= -tol:
        print("Converged: X is PSD to within tolerance.")
        break
    
    # Otherwise, for each negative eigenvalue below -tol, add a cut
    for idx, eigval in enumerate(w):
        if eigval < -tol:
            eigvec = V[:, idx]
            cut_name = f"psd_cut_{it}_{idx}"
            add_psd_cut(model, eigvec, cut_name)

# ============================================================
#                           Output
# ============================================================
print("Final solution X:")
X_val = np.array([[value(model.X[i,j]) for j in range(n)] for i in range(n)])
print(X_val)
print("Eigenvalues of final X:", np.linalg.eigvalsh(X_val))
print("Final objective value =", value(model.obj))