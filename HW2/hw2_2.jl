using LinearAlgebra
using PyPlot

# Define the function f(x)
function f(x)
    x1, x2, x3 = x[1], x[2], x[3]
    return x3 * log(exp(x1/x3) + exp(x2/x3)) + (x3 - 2)^2 + exp(1/(x1 + x2))
end

# Define the gradient of f(x)
function grad_f(x)
    x1, x2, x3 = x[1], x[2], x[3]
    S = exp(x1) + exp(x2)
    df_dx1 = (x3 * exp(x1) / S) - (exp(1/S) / (x1 + x2)^2)
    df_dx2 = (x3 * exp(x2) / S) - (exp(1/S) / (x1 + x2)^2)
    df_dx3 = log(S) + 2 * (x3 - 2)
    return [df_dx1, df_dx2, df_dx3]
end

# Define the constraint function
function constraints(x)
    x1, x2, x3 = x[1], x[2], x[3]
    if x1 + x2 <= 0
        x2 = -x1 + 1e-2
    end
    if x3 <= 0
        x3 = 1e-2
    end
    return [x1, x2, x3]
end

# BFGS with backtracking line search
function bfgs_backtracking(x_init; t_init=1.0, alpha=0.4, beta=0.5, tol=1e-5, max_iter=10000)
    x_new = 0
    f_new = 0
    x = copy(x_init)
    n = length(x)
    # Initialize the inverse Hessian approximation as the identity matrix.
    B = Matrix{Float64}(I, n, n)
    f_values = [f(x)]
    iter_count = 0

    grad = grad_f(x)

    for iter in 1:max_iter
        if norm(grad) < tol
            break
        end

        # Compute search direction.
        p = -B * grad

        # Backtracking line search.
        t = t_init
        while true
            x_new = constraints(x .+ t .* p)
            f_new = f(x_new)
            if f_new <= f(x) + alpha * t * dot(grad, p)
                break
            end
            t *= beta
        end

        s = x_new .- x
        grad_new = grad_f(x_new)
        y = grad_new .- grad

        # BFGS update.
        if dot(s, y) > 1e-10
            By = B * y
            s_y = dot(s, y)
            y_By = dot(y, By)
            B = B + (s * transpose(s)) / s_y - (By * transpose(By)) / y_By
        else
            B = Matrix{Float64}(I, n, n)
            println("Resetting B to identity matrix at iteration $iter_count due to small s^T y")
        end

        x = x_new
        grad = grad_new
        push!(f_values, f_new)
        iter_count += 1
    end

    return x, f(x), f_values, iter_count
end

# Define initial points
x_init_1 = [5.0, 3.0, 3.0]
x_init_2 = [3.0, 5.0, 3.0]
x_init_3 = [3.0, 3.0, 5.0]

# Run BFGS with backtracking from different starting points
result_bfgs_bt_1 = bfgs_backtracking(x_init_1)
result_bfgs_bt_2 = bfgs_backtracking(x_init_2)
result_bfgs_bt_3 = bfgs_backtracking(x_init_3)

# Function to print the results
function print_results_bfgs_bt(result, x_init, index)
    x_min, f_min, f_values, iter_count = result
    println("BFGS with Backtracking Result for initial point $index:")
    println("Initial point: ", x_init)
    println("Local minimum point: ", x_min)
    println("Local minimum value: ", f_min)
    println("Number of iterations: ", iter_count, "\n")
end

print_results_bfgs_bt(result_bfgs_bt_1, x_init_1, 1)
print_results_bfgs_bt(result_bfgs_bt_2, x_init_2, 2)
print_results_bfgs_bt(result_bfgs_bt_3, x_init_3, 3)

# Plotting the convergence of function values
figure(figsize=(12, 4))
results = [result_bfgs_bt_1, result_bfgs_bt_2, result_bfgs_bt_3]

for (i, result) in enumerate(results)
    subplot(1, 3, i)
    plot(result[3])
    title("BFGS Backtracking Convergence Plot $i")
    xlabel("Iteration")
    ylabel("Function Value")
end

tight_layout()
show()