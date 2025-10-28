import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import rosen

x = np.arange(-2, 2, .01)
y = np.arange(-1, 3, .01)
X, Y = np.meshgrid(x, y)
Z = rosen((X, Y))

min_index = np.unravel_index(np.argmin(Z), Z.shape)
x_min_brute = X[min_index]
y_min_brute = Y[min_index]
z_min_brute = Z[min_index]

print(f"Brute-force minimum approx: x={x_min_brute:.4f}, y={y_min_brute:.4f}, f={z_min_brute:.6f}")

#Gradient Descent 
def rosen_grad(x, y):
    """Gradient of the Rosenbrock function."""
    dfdx = -2*(1 - x) - 400*x*(y - x**2)
    dfdy = 200*(y - x**2)
    return np.array([dfdx, dfdy])

def gradient_descent(start, lr=0.001, tol=1e-6, max_iter=100000):
    x, y = start
    for i in range(max_iter):
        grad = rosen_grad(x, y)
        step = lr * grad
        x_new, y_new = x - step[0], y - step[1]
        if np.linalg.norm([x_new - x, y_new - y]) < tol:
            print(f"Converged after {i} iterations.")
            break
        x, y = x_new, y_new
    return x, y, rosen((x, y))
start_point = (-1.5, 2.0)
x_min_gd, y_min_gd, z_min_gd = gradient_descent(start_point, lr=0.001)

plt.figure(figsize=(7,5))
plt.pcolormesh(X, Y, Z, shading='auto', norm='log', vmin=1e-3)
plt.plot(x_min_b, y_min_b, 'ro', label='Brute-force min')
plt.plot(x_min_g, y_min_g, 'go', label='Gradient descent min')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Rosenbrock Function Minima")
plt.legend()
plt.colorbar(label="f(x, y)")
plt.show()
