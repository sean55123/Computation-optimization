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

def gradient_descent_backtracking(x_init, t_init=1, alpha=0.4, beta=0.5, tol=1e-5, max_iter=10000):
    x = x_init.copy()
    f_values = [f(x)]
    iter_count = 0

    for _ in range(max_iter):
        grad = grad_f(x)
        grad_norm = np.linalg.norm(grad)

        # Stop criteria, once the improvement is smaller than the tolerance, the optimization will be stopped.
        if grad_norm < tol:
            break

        t = t_init

        # Backtracking line search
        while True:
            x_new = constraints(x - t * grad)
            f_new = f(x_new)
            
            # Repeat the loop until reach a better point
            if f_new <= f(x) - alpha * t * grad_norm**2:
                break
            t *= beta  # Reduce step size with pre-defined reduction factor

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

result_bt_1 = gradient_descent_backtracking(x_init_1)
result_bt_2 = gradient_descent_backtracking(x_init_2)
result_bt_3 = gradient_descent_backtracking(x_init_3)

def print_results_bt(result, x_init, index):
    x_min, f_min, f_values, iter_count = result
    print(f"Backtracking Result for initial point {index}:")
    print(f"Initial point: {x_init}")
    print(f"Local minimum point: {x_min}")
    print(f"Local minimum value: {f_min}")
    print(f"Number of iterations: {iter_count}\n")

print_results_bt(result_bt_1, x_init_1, 1)
print_results_bt(result_bt_2, x_init_2, 2)
print_results_bt(result_bt_3, x_init_3, 3)

plt.figure(figsize=(12, 4))

for i, result in enumerate([result_bt_1, result_bt_2, result_bt_3], 1):
    plt.subplot(1, 3, i)
    plt.plot(result[2])
    plt.title(f'Convergence Plot {i}')
    plt.xlabel('Iteration')
    plt.ylabel('Function Value')

plt.tight_layout()
plt.show()