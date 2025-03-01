import numpy as np
import matplotlib.pyplot as plt

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
plt.legend()
plt.grid(True)
plt.show()