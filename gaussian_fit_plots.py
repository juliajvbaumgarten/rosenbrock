import numpy as np
import matplotlib.pyplot as plt
from gaussian_model import gaussian
from gaussian_newton import newton_gaussian_fit
from gradient_descent_gaussian import gradient_descent_gaussian
from nelder_mead_gaussian import nelder_mead_gaussian

np.random.seed(42)
x = np.linspace(-5, 5, 100)
true_params = [2.0, 1.0, 1.5, 0.5]
y = gaussian(x, *true_params) + 0.1 * np.random.randn(len(x))
params0 = [1.0, 0.0, 1.0, 0.0]

# fit using each method
fit_newton, path_newton = newton_gaussian_fit(x, y, params0)
fit_gd, path_gd = gradient_descent_gaussian(x, y, params0, lr=1e-4)
fit_nm = nelder_mead_gaussian(x, y, params0)

# plot results
fig, axes = plt.subplots(1, 3, figsize=(12, 10))
axes = axes.ravel()

axes[0].scatter(x, y, label="data")
axes[0].plot(x, gaussian(x, *fit_newton), 'r-', label="Newton fit")
axes[0].set_title("Newton Fit"); axes[0].legend()

axes[1].scatter(x, y, label="data")
axes[1].plot(x, gaussian(x, *fit_gd), 'g-', label="Gradient Descent fit")
axes[1].set_title("Gradient Descent Fit"); axes[1].legend()

axes[2].scatter(x, y, label="data")
axes[2].plot(x, gaussian(x, *fit_nm), 'm-', label="Nelder–Mead fit")
axes[2].set_title("Nelder–Mead Fit"); axes[2].legend()

plt.tight_layout(); plt.show()

# look at parameters and compare to true ones
print("True parameters:", true_params)
print("Newton fit params:", fit_newton)
print("Gradient Descent fit:", fit_gd)
print("Nelder–Mead fit:", fit_nm)

# calculating the adjusted R^2 value
y_newton = gaussian(x, *fit_newton)
y_gd = gaussian(x, *fit_gd)
y_nm = gaussian(x, *fit_nm)

def r_squared(y, y_fit):
    ss_res = np.sum((y - y_fit)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - ss_res / ss_tot

def r_squared_adj(y, y_fit, p):
    n = len(y)
    r2 = r_squared(y, y_fit)
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

p = 4  # number of parameters
print("Adjusted R^2:")
print("Newton R^2 adjusted:", r_squared_adj(y, y_newton, p))
print("Gradient Descent R^2 adj:", r_squared_adj(y, y_gd, p))
print("Nelder–Mead R^2 adj:", r_squared_adj(y, y_nm, p))


