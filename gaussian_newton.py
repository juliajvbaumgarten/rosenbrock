import numpy as np
import h5py
import matplotlib.pyplot as plt

# load data
with h5py.File('./data.hdf', 'r') as f:
    xpos = f['data/xpos'][:]
    ypos = f['data/ypos'][:]

# Gaussian model and Jacobian
def gaussian(x, A, mu, sigma, C):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + C

def residuals(params, x, y):
    A, mu, sigma, C = params
    return y - gaussian(x, A, mu, sigma, C)

def jacobian(params, x):
    A, mu, sigma, C = params
    g = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    dA = -g
    dmu = -(A * g * (x - mu) / (sigma ** 2))
    dsigma = -(A * g * ((x - mu) ** 2) / (sigma ** 3))
    dC = -np.ones_like(x)
    return np.vstack((dA, dmu, dsigma, dC)).T

# Newton's method
def newton_gaussian_fit(x, y, params0, max_iter=50, tol=1e-8):
    params = np.array(params0, dtype=float)
    for _ in range(max_iter):
        r = residuals(params, x, y)
        J = jacobian(params, x)
        H = J.T @ J
        g = J.T @ r
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(H) @ g
        params -= step
        # stop sigma from going negative or zero
        if params[2] <= 0:
            params[2] = abs(params[2]) + 1e-6
        if np.linalg.norm(step) < tol:
            break
    return params

# initial guess
A0 = np.max(ypos) - np.min(ypos)
mu0 = xpos[np.argmax(ypos)]
sigma0 = (xpos.max() - xpos.min()) / 10
C0 = np.mean(ypos)
params0 = [A0, mu0, sigma0, C0]

fit_newton = newton_gaussian_fit(xpos, ypos, params0)
print("Optimal parameters (A, mu, sigma, C):", fit_newton)

# plot fit
x_sorted = np.sort(xpos)
y_fit = gaussian(x_sorted, *fit_newton)

plt.figure(figsize=(8,6))
plt.scatter(xpos, ypos, s=10, label="Data")
plt.plot(x_sorted, y_fit, 'r-', linewidth=2, label="Newton Fit")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Gaussian Fit using Newton’s Method")
plt.show()

# calculate R^2
def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

def r_squared_adj(y_true, y_pred, p):
    n = len(y_true)
    r2 = r_squared(y_true, y_pred)
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

y_fit_newton = gaussian(xpos, *fit_newton)
r2_newton = r_squared(ypos, y_fit_newton)
r2_adj_newton = r_squared_adj(ypos, y_fit_newton, p=4)  

print(f"R^2 (Newton): {r2_newton:.6f}")
print(f"Adjusted R^2 (Newton): {r2_adj_newton:.6f}")
