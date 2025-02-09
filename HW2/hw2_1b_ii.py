import numpy as np
import matplotlib.pyplot as plt

def f(x):
    x1, x2, x3 = x[0], x[1], x[2]
    return x3*np.log(np.exp(x1/x3) + np.exp(x2/x3)) + (x3-2)**2 + np.exp(1/(x1+x2))

def grad_f(x):
    x1, x2, x3 = x[0], x[1], x[2]
    
    S = np.exp(x1) + np.exp(x2)
    
    df_dx1 = (x3*np.exp(x1) / S) - (np.exp(1/S) / (x1+x2)**2)
    df_dx2 = (x3*np.exp(x2) / S) - (np.exp(1/S) / (x1+x2)**2)
    df_dx3 = np.log(S) + 2*(x3-2)
    return np.array([df_dx1, df_dx2, df_dx3])

def hess_f(x):
    x1, x2, x3 = x[0], x[1], x[2]
    
    S = np.exp(x1) + np.exp(x2)
    H11 = (x3*np.exp(x1)*np.exp(x2)/S**2) + np.exp(1/S) 
    H22 = (x3*np.exp(x1)*np.exp(x2)/S**2) + np.exp(1/S) 
    H33 = 2
    H12 = H21 = (-x3*np.exp(x1)*np.exp(x2)/S**2) + np.exp(1/S)*((1+2*x1+x2)/(x1+x2)**4)
    H13 = H31 = np.exp(x1) / S
    H23 = H32 = np.exp(x2) / S
    
    H = ([[H11, H12, H13],
          [H21, H22, H23],
          [H31, H32, H33]])
    
    return H

def newton_backtracking(x_init, t_init=1, alpha=0.4, beta=0.5, tol=1e-5, max_iter=10000):
    x = x_init.copy()
    f_values = [f(x)]
    iter_count = 0

    for _ in range(max_iter):
        grad = grad_f(x)
        H = hess_f(x)

        if np.linalg.norm(grad) < tol:
            break

        # Ensure Hessian is positive definite
        H_mod = H.copy()
        eigvals = np.linalg.eigvalsh(H)
        min_eig = np.min(eigvals)
        if min_eig <= 0:
            H_mod += np.eye(2) * (-min_eig + 1e-6)

        # Compute the Newton direction
        try:
            delta_x = np.linalg.solve(H_mod, grad)
        except np.linalg.LinAlgError:
            print("Singular Hessian encountered.")
            break

        t = t_init

        # Backtracking line search
        while True:
            x_new = constraints(x - t * delta_x)
            f_new = f(x_new)
            if f_new <= f(x) - alpha * t * np.dot(grad, delta_x):
                break
            t *= beta

        x = x_new
        f_values.append(f_new)
        iter_count += 1

    return x, f(x), f_values, iter_count

def constraints(x):
    x1, x2, x3 = x[0], x[1], x[2]
    if x1 + x2 <= 0:
        x2 = -x1 + 1e-2
    
    if x3 <= 0:
        x3 = 1e-2
    
    return np.array([x1, x2, x3])

x_init_1 = np.array([5, 3, 3])
x_init_2 = np.array([3, 5, 3])
x_init_3 = np.array([3, 3, 5])


result_dnb_1 = newton_backtracking(x_init_1)
result_dnb_2 = newton_backtracking(x_init_2)
result_dnb_3 = newton_backtracking(x_init_3)

def print_results_dnb(result, x_init, index):
    x_min, f_min, f_values, iter_count = result
    print(f"Newton Result for initial point {index}:")
    print(f"Initial point: {x_init}")
    print(f"Local minimum point: {x_min}")
    print(f"Local minimum value: {f_min}")
    print(f"Number of iterations: {iter_count}\n")

print_results_dnb(result_dnb_1, x_init_1, 1)
print_results_dnb(result_dnb_2, x_init_2, 2)
print_results_dnb(result_dnb_3, x_init_3, 3)

plt.figure(figsize=(12, 4))

for i, result in enumerate([result_dnb_1, result_dnb_2, result_dnb_3], 1):
    plt.subplot(1, 3, i)
    plt.plot(result[2])
    plt.title(f'Newton Convergence Plot {i}')
    plt.xlabel('Iteration')
    plt.ylabel('Function Value')

plt.tight_layout()
plt.show()