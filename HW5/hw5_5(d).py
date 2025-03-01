import numpy as np

def f(x):
    x1, x2 = x
    return 3.0/(x1 + x2) + np.exp(x1) + (x1 - x2)**2

def grad_f(x):
    x1, x2 = x
    return np.array([
        -3.0/(x1 + x2)**2 + np.exp(x1) + 2*(x1 - x2),
        -3.0/(x1 + x2)**2 - 2*(x1 - x2)
    ])

# Constraints g_i(x) <= 0
def g1(x): return -x[0]                
def g2(x): return -x[1]                
def g3(x): return x[0]**2 + x[1]**2 - 2 
def g4(x): return x[0] - x[1] - 1      

g_funcs = [g1, g2, g3, g4]

def grad_g1(x): return np.array([-1.0,  0.0])
def grad_g2(x): return np.array([ 0.0, -1.0])
def grad_g3(x): 
    x1, x2 = x
    return np.array([2.0*x1, 2.0*x2])
def grad_g4(x): return np.array([1.0, -1.0])

grad_g_funcs = [grad_g1, grad_g2, grad_g3, grad_g4]

def hess_f(x):
    x1, x2 = x
    d2f_dx1x1 = 6.0/(x1 + x2)**3 + np.exp(x1) + 2.0
    d2f_dx2x2 = 6.0/(x1 + x2)**3 + 2.0
    d2f_dx1x2 = 6.0/(x1 + x2)**3 - 2.0
    return np.array([
        [d2f_dx1x1, d2f_dx1x2],
        [d2f_dx1x2, d2f_dx2x2]
    ])

def kkt_residual(x, s, lam, T):
    # r_stat: grad f(x) + sum lam_i grad g_i(x)
    r_stat = grad_f(x)
    for i in range(4):
        r_stat += lam[i] * grad_g_funcs[i](x)
    
    # r_feas: g_i(x) + s_i = 0
    r_feas = np.array([g_funcs[i](x) + s[i] for i in range(4)])
    
    # r_cent: lam_i s_i - T_i = 0
    r_cent = lam*s - T
    
    return np.concatenate([r_stat, r_feas, r_cent])

def kkt_jacobian(x, s, lam, T):
    # Stationarity block
    # Hess(f) + sum_i lam_i Hess(g_i). Only g3 is nonlinear
    Hf = hess_f(x)
    Hg = np.zeros((2,2))
    # g3 => Hess(g3)=[[2,0],[0,2]]
    Hg += lam[2]*np.array([[2.0,0],[0,2.0]])
    A = Hf + Hg  # 2x2
    
    # r_stat wrt s => 0
    B = np.zeros((2,4))
    
    # r_stat wrt lam => [grad g1, grad g2, grad g3, grad g4] in columns
    C = np.column_stack([grad_g_funcs[i](x) for i in range(4)])
    
    # r_feas wrt x => each row = grad g_i(x)
    D = np.array([grad_g_funcs[i](x) for i in range(4)])
    
    # r_feas wrt s => 4x4 identity
    E = np.eye(4)
    
    # r_feas wrt lam => 0
    F = np.zeros((4,4))
    
    # r_cent wrt x => 0
    G_ = np.zeros((4,2))
    
    # r_cent wrt s => diag(lam)
    H_ = np.diag(lam)
    
    # r_cent wrt lam => diag(s)
    I_ = np.diag(s)
    
    top    = np.hstack([A,        B,        C])
    middle = np.hstack([D,        E,        F])
    bottom = np.hstack([G_,       H_,       I_])
    return np.vstack([top, middle, bottom])

def primal_dual_step(x, s, lam, T):
    r = kkt_residual(x, s, lam, T)
    J = kkt_jacobian(x, s, lam, T)
    try:
        delta = np.linalg.solve(J, -r)
    except np.linalg.LinAlgError:
        delta = np.linalg.lstsq(J, -r, rcond=None)[0]
    dx   = delta[:2]
    ds   = delta[2:6]
    dlam = delta[6:10]
    return dx, ds, dlam

def line_search(x, s, lam, dx, ds, dlam, T, alpha=0.5, beta=0.3):
    step = 1.0
    # positivity constraints on s and lam
    neg_s   = ds < 0
    neg_lam = dlam < 0
    if np.any(neg_s):
        step = min(step, 0.99 * np.min(-s[neg_s]/ds[neg_s]))
    if np.any(neg_lam):
        step = min(step, 0.99 * np.min(-lam[neg_lam]/dlam[neg_lam]))
    
    r_old = kkt_residual(x, s, lam, T)
    norm_r_old = np.linalg.norm(r_old)
    
    while True:
        x_new   = x   + step*dx
        s_new   = s   + step*ds
        lam_new = lam + step*dlam
        
        # must keep s_new>0, lam_new>0
        if np.any(s_new <= 0) or np.any(lam_new <= 0):
            step *= beta
            if step < 1e-16:
                break
            continue
        
        r_new = kkt_residual(x_new, s_new, lam_new, T)
        norm_r_new = np.linalg.norm(r_new)
        # Armijo-type check
        if norm_r_new <= (1 - alpha*step)*norm_r_old:
            break
        
        step *= beta
        if step < 1e-16:
            break
    
    return step

def primal_dual_interior_point_v2(x0, lam0=1.0, mu=1.2, eps=1e-5, alpha=0.5, beta=0.3, max_iter=50):
    m = 4
    
    # Initialize x, s, lam, and T
    x = np.array(x0, dtype=float)
    s = np.array([-g_funcs[i](x) for i in range(m)])  # s_i = -g_i(x)
    if np.isscalar(lam0):
        lam = lam0*np.ones(m, dtype=float)
    else:
        lam = np.array(lam0, dtype=float)
    
    # T in R^m, starts as (1,1,1,1)^\top
    T = np.ones(m, dtype=float)
    
    # Check strict feasibility
    if np.any(s <= 0) or np.any(lam <= 0):
        raise ValueError("Initial guess is not strictly feasible.")
    
    outer_iter = 0
    while True:
        # Inner loop: Newton iterations for current T
        for _ in range(max_iter):
            r = kkt_residual(x, s, lam, T)
            if np.linalg.norm(r) < 1e-10:
                break
            
            dx, ds, dlam = primal_dual_step(x, s, lam, T)
            step = line_search(x, s, lam, dx, ds, dlam, T, alpha, beta)
            
            x   += step*dx
            s   += step*ds
            lam += step*dlam
        
        if np.max(T) > 1.0/eps:
            break
        
        T *= mu
        outer_iter += 1
    
    return x, s, lam

x0 = np.array([0.5, 0.5])
x_star, s_star, lam_star = primal_dual_interior_point_v2(
    x0, lam0=1.0, mu=1.2, eps=1e-5, alpha=0.5, beta=0.3
)
print("x* =", x_star)
print("s* =", s_star)
print("lam* =", lam_star)
print("Objective f(x*) =", f(x_star))