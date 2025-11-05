import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# load data
with h5py.File('./data.hdf', 'r') as f:
    xpos = f['data/xpos'][:]
    ypos = f['data/ypos'][:]

def gaussian(x, A, mu, sigma, C):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + C

# loss of squares
def loss(params):
    A, mu, sigma, C = params
    y_pred = gaussian(xpos, A, mu, sigma, C)
    return np.sum((ypos - y_pred)**2)

# initial guesses
A0 = np.max(ypos) - np.min(ypos)
mu0 = xpos[np.argmax(ypos)]
sigma0 = (xpos.max() - xpos.min()) / 10
C0 = np.mean(ypos)
initial_guess = [A0, mu0, sigma0, C0]

# fit with Nelder-ead
result = minimize(loss, initial_guess, method='Nelder-Mead', 
                  options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': True})

print("Optimal parameters (A, mu, sigma, C):", result.x)

# plot data and fit
x_sorted = np.sort(xpos)
y_fit = gaussian(x_sorted, *result.x)

plt.figure(figsize=(8,6))
plt.scatter(xpos, ypos, s=10, label="Data")
plt.plot(x_sorted, y_fit, 'r-', label="Gaussian Fit", linewidth=2)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Gaussian Fit using Nelder–Mead")
plt.show()

