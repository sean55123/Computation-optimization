using LinearAlgebra
using Plots

function f(x)
    x1, x2, x3 = x[1], x[2], x[3]
    return x3 * log(exp(x1/x3) + exp(x2/x3)) + (x3 - 2)^2 + exp(1/(x1 + x2))
end

function grad_f(x)
    x1, x2, x3 = x[1], x[2], x[3]

    S = exp(x1) + exp(x2)
    df_dx1 = (x3 * exp(x1) / S) - (exp(1/S) / (x1 + x2)^2)
    df_dx2 = (x3 * exp(x2) / S) - (exp(1/S) / (x1 + x2)^2)
    df_dx3 = log(S) + 2*(x3 - 2)
    return [df_dx1, df_dx2, df_dx3]
end

# Define the constraints function
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

# Gradient descent with backtracking line search
function gradient_descent_backtracking(x_init; t_init=1.0, alpha=0.4, beta=0.5, tol=1e-5, max_iter=10000)
    x_new = 0
    f_new = 0
    x = copy(x_init)
    f_values = [f(x)]
    iter_count = 0

    for _ in 1:max_iter
        grad = grad_f(x)
        grad_norm = norm(grad)

        # Stop if the gradient norm is below tolerance
        if grad_norm < tol
            break
        end

        t = t_init
        # Backtracking line search loop
        while true
            x_new = constraints(x .- t .* grad)
            f_new = f(x_new)
            if f_new <= f(x) - alpha * t * grad_norm^2
                break
            end
            t *= beta
        end

        x = x_new
        push!(f_values, f_new)
        iter_count += 1
    end

    return x, f(x), f_values, iter_count
end

# Define three initial points
x_init_1 = [5.0, 3.0, 3.0]
x_init_2 = [3.0, 5.0, 3.0]
x_init_3 = [3.0, 3.0, 5.0]

# Run the gradient descent for each initial point
result_bt_1 = gradient_descent_backtracking(x_init_1)
result_bt_2 = gradient_descent_backtracking(x_init_2)
result_bt_3 = gradient_descent_backtracking(x_init_3)

# Function to print results
function print_results_bt(result, x_init, index)
    x_min, f_min, f_values, iter_count = result
    println("Backtracking Result for initial point $index:")
    println("Initial point: $x_init")
    println("Local minimum point: $x_min")
    println("Local minimum value: $f_min")
    println("Number of iterations: $iter_count\n")
end

print_results_bt(result_bt_1, x_init_1, 1)
print_results_bt(result_bt_2, x_init_2, 2)
print_results_bt(result_bt_3, x_init_3, 3)

# Plot convergence curves in a 1x3 layout
p1 = plot(result_bt_1[3], title="Convergence Plot 1", xlabel="Iteration", ylabel="Function Value")
p2 = plot(result_bt_2[3], title="Convergence Plot 2", xlabel="Iteration", ylabel="Function Value")
p3 = plot(result_bt_3[3], title="Convergence Plot 3", xlabel="Iteration", ylabel="Function Value")
plot(p1, p2, p3, layout=(1, 3), size=(1200, 400))