import numpy as np
import matplotlib.pyplot as plt

def f(x):
    """
    f(x) = 3/(x1 + x2) + exp(x1) + (x1 - x2)^2
    """
    x1, x2 = x
    return 3.0/(x1 + x2) + np.exp(x1) + (x1 - x2)**2

def grad_f(x):
    """
    Gradient of f(x).
    """
    x1, x2 = x
    # partial wrt x1
    df_dx1 = -3.0/(x1 + x2)**2 + np.exp(x1) + 2*(x1 - x2)
    # partial wrt x2
    df_dx2 = -3.0/(x1 + x2)**2 - 2*(x1 - x2)
    return np.array([df_dx1, df_dx2])

def hess_f(x):
    """
    Hessian of f(x).
    """
    x1, x2 = x
    # second partials
    d2f_dx1x1 = 6.0/(x1 + x2)**3 + np.exp(x1) + 2.0
    d2f_dx2x2 = 6.0/(x1 + x2)**3 + 2.0
    d2f_dx1x2 = 6.0/(x1 + x2)**3 - 2.0
    d2f_dx2x1 = d2f_dx1x2
    return np.array([
        [d2f_dx1x1, d2f_dx1x2],
        [d2f_dx2x1, d2f_dx2x2]
    ])

def c1(x): return x[0]
def c2(x): return x[1]
def c3(x): return 2.0 - x[0]**2 - x[1]**2
def c4(x): return 1.0 - x[0] + x[1]

def phi(x):
    return -np.log(c1(x)) - np.log(c2(x)) \
           - np.log(c3(x)) - np.log(c4(x))

def grad_phi(x):
    x1, x2 = x
    # c_i(x) values:
    cvals = np.array([c1(x), c2(x), c3(x), c4(x)])
    # grad(c_i):
    grad_c = np.array([
        [1.0, 0.0],         # grad c1
        [0.0, 1.0],         # grad c2
        [-2.0*x1, -2.0*x2], # grad c3
        [-1.0, 1.0]         # grad c4
    ])
    g = np.zeros(2)
    for i in range(4):
        g -= grad_c[i] / cvals[i]
    return g

def hess_phi(x):
    x1, x2 = x
    cvals = np.array([c1(x), c2(x), c3(x), c4(x)])
    grad_c = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [-2.0*x1, -2.0*x2],
        [-1.0,  1.0]
    ])
    # Hessians of each c_i (only c3 is nonzero):
    # c1''=0, c2''=0,
    # c3'' = [[-2, 0],[0, -2]],
    # c4''=0
    hess_c = np.zeros((4,2,2))
    hess_c[2] = np.array([[-2.0, 0.0],[0.0, -2.0]])  # for c3
    
    H = np.zeros((2,2))
    for i in range(4):
        gc = grad_c[i].reshape(2,1)   # column vector
        H -= (gc @ gc.T) / (cvals[i]**2)
        H += hess_c[i] / cvals[i]
    return H

def F_t(x, t):
    return f(x) + (1.0/t)*phi(x)

def grad_F_t(x, t):
    return grad_f(x) + (1.0/t)*grad_phi(x)

def hess_F_t(x, t):
    return hess_f(x) + (1.0/t)*hess_phi(x)

def is_feasible(x):
    x1, x2 = x
    if x1 <= 0.0: return False
    if x2 <= 0.0: return False
    if x1**2 + x2**2 >= 2.0: return False
    if x1 - x2 >= 1.0: return False
    return True

def newton_subproblem(x0, t, tol=1e-5, max_iter=50, alpha=0.2, beta=0.5):
    x = x0.copy()
    for _ in range(max_iter):
        g = grad_F_t(x, t)
        H = hess_F_t(x, t)
        # Stopping criterion on gradient
        if np.linalg.norm(g) < tol:
            break
        
        # Newton direction
        dx = -np.linalg.solve(H, g)
        
        # Backtracking line search
        step = 1.0
        while True:
            x_new = x + step*dx
            if not is_feasible(x_new):
                # reduce step if we leave the feasible region
                step *= beta
            else:
                # Armijo condition for sufficient decrease
                lhs = F_t(x_new, t)
                rhs = F_t(x, t) + alpha*step*np.dot(g, dx)
                if lhs <= rhs:
                    # acceptable step
                    break
                step *= beta
            
            if step < 1e-16:
                break
        
        x = x_new
    return x


def barrier_method(x0, mu=4.0, eps=1e-4, newton_tol=1e-5, alpha=0.2, beta=0.5):
    # Dimension of x:
    n = 2  
    # Start with t=1
    t = 1.0
    x = x0.copy()
    
    # Keep track of iterates if you want to plot them
    x_history = [x0.copy()]
    
    while True:
        # Solve the barrier subproblem for current t
        x = newton_subproblem(x, t, 
                              tol=newton_tol, 
                              alpha=alpha, beta=beta)
        x_history.append(x.copy())
        
        # Check stopping condition: n/t < eps
        if (n / t) < eps:
            break
        
        # Otherwise increase t
        t *= mu
    
    return x, x_history

# Suppose we start at (0.5, 0.5), which is strictly feasible:
x0 = np.array([0.5, 0.5])

# Solve via barrier method
x_star, path = barrier_method(x0, 
                                mu=4.0, 
                                eps=1e-4, 
                                newton_tol=1e-5, 
                                alpha=0.2, beta=0.5)

print("Optimal solution found:", x_star)

### Plot
x1 = np.linspace(0, 2, 400)
x2 = np.linspace(0, 2, 400)
X1, X2 = np.meshgrid(x1, x2)

# Define constraints
constraint3 = X1**2 + X2**2 <= 2
constraint4 = X1 - X2 <= 1

# Feasible region (x1 >= 0 and x2 >= 0 are ensured by grid limits)
feasible = constraint3 & constraint4

# Plot feasible region
plt.contourf(X1, X2, feasible, levels=[0.5, 1], colors='lightblue', alpha=0.5)

# Plot boundaries
# Circle (quarter arc)
theta = np.linspace(0, np.pi/2, 100)
circle_x1 = np.sqrt(2) * np.cos(theta)
circle_x2 = np.sqrt(2) * np.sin(theta)
plt.plot(circle_x1, circle_x2, 'k--', label=r'$x_1^2 + x_2^2 = 2$')

# Line x1 = x2 + 1
x2_line = np.linspace(0, 2, 100)
x1_line = x2_line + 1
plt.plot(x1_line, x2_line, 'k--', label=r'$x_1 = x_2 + 1$')

# Axes
plt.plot([0, 0], [0, 2], 'k--', label=r'$x_1 = 0$')
plt.plot([0, 2], [0, 0], 'k--', label=r'$x_2 = 0$')

# Set limits and labels
plt.xlim(0, 2)
plt.ylim(0, 2)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Feasible Region')

path = np.array(path)
plt.plot(path[:,0], path[:,1], 'ro-', label='Barrier iterates')
plt.plot(x_star[0], x_star[1], 'bx', markersize=10, label='Final solution')
plt.legend()
plt.grid(True)
plt.show()