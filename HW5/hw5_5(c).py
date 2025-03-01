import numpy as np

def f(x):
    x1, x2 = x
    return 3.0/(x1 + x2) + np.exp(x1) + (x1 - x2)**2

def grad_f(x):
    x1, x2 = x
    df_dx1 = -3.0/(x1 + x2)**2 + np.exp(x1) + 2*(x1 - x2)
    df_dx2 = -3.0/(x1 + x2)**2 - 2*(x1 - x2)
    return np.array([df_dx1, df_dx2])

# Constraints g_i(x) <= 0
def g1(x): return -x[0]                # x1 >= 0 => -x1 <= 0
def g2(x): return -x[1]                # x2 >= 0 => -x2 <= 0
def g3(x): return x[0]**2 + x[1]**2 - 2 # x1^2+x2^2 <= 2
def g4(x): return x[0] - x[1] - 1      # x1 - x2 <= 1

def grad_g1(x): return np.array([-1.0,  0.0])
def grad_g2(x): return np.array([ 0.0, -1.0])
def grad_g3(x):
    x1, x2 = x
    return np.array([2.0*x1, 2.0*x2])
def grad_g4(x): return np.array([1.0, -1.0])

g_funcs = [g1, g2, g3, g4]
grad_g_funcs = [grad_g1, grad_g2, grad_g3, grad_g4]

def kkt_residual(x, s, lam, t):
    # 1) Stationarity: grad f(x) + sum lam_i grad g_i(x)
    r_stat = grad_f(x)
    for i in range(4):
        r_stat += lam[i]*grad_g_funcs[i](x)
    
    # 2) Primal feasibility: g_i(x) + s_i = 0
    r_feas = np.zeros(4)
    for i in range(4):
        r_feas[i] = g_funcs[i](x) + s[i]
    
    # 3) Complementarity (central path): lam_i * s_i - 1/t = 0
    r_cent = lam*s - (1.0/t)*np.ones(4)
    
    # Concatenate into a single vector (dim 2+4+4=10)
    return np.concatenate([r_stat, r_feas, r_cent])

def kkt_jacobian(x, s, lam, t):
    Hf = hess_f(x)
    Hg = np.zeros((2,2))
    # We'll just do it properly for g3 (since it has a nonzero Hessian)
    # g3(x) = x1^2 + x2^2 - 2 => Hess(g3) = [[2,0],[0,2]]
    # g1,g2,g4 are linear => Hess=0
    Hg += lam[2]*np.array([[2.0, 0.0],[0.0, 2.0]])
    A = Hf + Hg
    
    # B) d(r_stat)/ds = partial of (grad f + sum lam_i grad g_i) wrt s
    #    = 0, because r_stat does not depend directly on s
    B = np.zeros((2,4))
    
    # C) d(r_stat)/dlam = partial wrt lam => sum_i grad g_i(x)
    #    i.e. each lam_i contributes grad g_i(x)
    C = np.column_stack([grad_g_funcs[i](x) for i in range(4)])
    
    # D) d(r_feas)/dx = partial of (g_i(x)+s_i) wrt x => grad g_i(x)
    D = np.zeros((4,2))
    for i in range(4):
        D[i,:] = grad_g_funcs[i](x)
    
    # E) d(r_feas)/ds = 4x4 identity
    E = np.eye(4)
    
    # F) d(r_feas)/dlam = 0
    F = np.zeros((4,4))
    
    # G) d(r_cent)/dx = 0
    G_ = np.zeros((4,2))
    
    # H) d(r_cent)/ds = diag(lam)
    H_ = np.diag(lam)
    
    # I) d(r_cent)/dlam = diag(s)
    I_ = np.diag(s)
    
    # Now assemble the block matrix
    top    = np.hstack([A,        B,        C])
    middle = np.hstack([D,        E,        F])
    bottom = np.hstack([G_,       H_,       I_])
    
    J = np.vstack([top, middle, bottom])
    return J

def hess_f(x):
    x1, x2 = x
    # second partials of f(x):
    d2f_dx1x1 = 6.0/(x1 + x2)**3 + np.exp(x1) + 2.0
    d2f_dx2x2 = 6.0/(x1 + x2)**3 + 2.0
    d2f_dx1x2 = 6.0/(x1 + x2)**3 - 2.0
    return np.array([
        [d2f_dx1x1, d2f_dx1x2],
        [d2f_dx1x2, d2f_dx2x2]
    ])

def primal_dual_step(x, s, lam, t):
    # Build residual and Jacobian
    r = kkt_residual(x, s, lam, t)
    J = kkt_jacobian(x, s, lam, t)
    
    # Solve for search direction
    try:
        delta = np.linalg.solve(J, -r)
    except np.linalg.LinAlgError:
        # fallback if singular
        delta = np.linalg.lstsq(J, -r, rcond=None)[0]
    
    # Extract pieces: dx in R^2, ds in R^4, dlam in R^4
    dx   = delta[0:2]
    ds   = delta[2:6]
    dlam = delta[6:10]
    return dx, ds, dlam

def line_search(x, s, lam, dx, ds, dlam, t, alpha=0.5, beta=0.3):
    step = 1.0
    # ensure positivity of s + step*ds and lam + step*dlam
    idx_s_neg   = ds < 0
    idx_lam_neg = dlam < 0
    if np.any(idx_s_neg):
        step = min(step, 0.99 * np.min(-s[idx_s_neg]/ds[idx_s_neg]))
    if np.any(idx_lam_neg):
        step = min(step, 0.99 * np.min(-lam[idx_lam_neg]/dlam[idx_lam_neg]))

    r_old = kkt_residual(x, s, lam, t)
    norm_r_old = np.linalg.norm(r_old)
    
    # reduce step until improvement
    while True:
        x_new   = x   + step*dx
        s_new   = s   + step*ds
        lam_new = lam + step*dlam
        
        # check positivity
        if np.any(s_new <= 0) or np.any(lam_new <= 0):
            step *= beta
            if step < 1e-16:
                break
            continue
        
        r_new = kkt_residual(x_new, s_new, lam_new, t)
        norm_r_new = np.linalg.norm(r_new)
        
        if norm_r_new <= (1 - alpha*step)*norm_r_old:
            # sufficient improvement
            break
        step *= beta
        if step < 1e-16:
            break
    
    return step


def primal_dual_interior_point(x0, lam0=1.0, mu=1.2, eps=1e-5, alpha=0.5, beta=0.3, max_iter=50):
    # Number of constraints
    m = 4
    
    # Start t=1
    t = 1.0
    
    # Initialize x = x0, s_i = -g_i(x0), lam
    x = np.array(x0, dtype=float)
    s = np.array([-g_funcs[i](x) for i in range(m)])  # s_i = -g_i(x)
    if np.isscalar(lam0):
        lam = lam0 * np.ones(m, dtype=float)
    else:
        lam = np.array(lam0, dtype=float)
    
    # Ensure everything is strictly positive to start
    if np.any(s <= 0) or np.any(lam <= 0):
        raise ValueError("Initial (x0, s0, lam0) not strictly feasible!")
    
    # Outer loop
    iteration = 0
    while True:
        # Solve with Newton's method for the current t
        for _ in range(max_iter):
            r = kkt_residual(x, s, lam, t)
            # Check stopping condition on residual
            if np.linalg.norm(r) < 1e-10:
                break
            
            dx, ds, dlam = primal_dual_step(x, s, lam, t)
            step = line_search(x, s, lam, dx, ds, dlam, t, alpha, beta)
            
            # Update
            x   += step*dx
            s   += step*ds
            lam += step*dlam
        
        # Check outer stopping: (m / t) < eps or some norm of r small
        if (m/t) < eps:
            break
        
        # Increase t
        t *= mu
        iteration += 1
    
    return x, s, lam

x0 = np.array([0.5, 0.5])

x_star, s_star, lam_star = primal_dual_interior_point(
    x0, lam0=1.0, mu=1.2, eps=1e-5,
    alpha=0.5, beta=0.3, max_iter=50
)

print("Primal solution x* =", x_star)
print("Slack variables s* =", s_star)
print("Dual variables lam* =", lam_star)
print("Objective value =", f(x_star))