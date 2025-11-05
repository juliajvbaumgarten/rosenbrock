import numpy as np
import matplotlib.pyplot as plt
from gaussian_model import gaussian
from gaussian_newton import newton_gaussian_fit
from gradient_descent_gaussian import gradient_descent_gaussian
from nelder_mead_gaussian import nelder_mead_gaussian

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


