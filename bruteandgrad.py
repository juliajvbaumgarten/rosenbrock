import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import rosen

print(f"Brute-force minimum approx: x={x_min_brute:.4f}, y={y_min_brute:.4f}, f={z_min_brute:.6f}")

class Result:
    def __init__(self, method, x0, x, f, iters, success, info):
        self.method = method
        self.x0 = x0
        self.x = x
        self.f = f
        self.iters = iters
        self.success = success
        self.info = info


# Brute force method 
def brute_force(xrange=(-2,2), yrange=(-1,3), step=0.01):
    x = np.arange(xrange[0], xrange[1], step)
    y = np.arange(yrange[0], yrange[1], step)
    X, Y = np.meshgrid(x, y)
    Z = rosen((X, Y))
    min_index = np.unravel_index(np.argmin(Z), Z.shape)
    x_min = X[min_index]
    y_min = Y[min_index]
    f_min = float(Z[min_index])
    return Result(
        method="brute_force",
        x0=np.array([np.nan, np.nan]),
        x=np.array([x_min, y_min]),
        f=f_min,
        iters=0,
        success=(f_min <= 1e-10),
        info={"grid_x": x, "grid_y": y, "z": Z}
    )

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

