import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize  

# Load the data
import h5py
f = h5py.File('./data.hdf', 'r')


print(f.keys())

print(f['data'].keys())

xpos = f['data/xpos'][:]
ypos = f['data/ypos'][:]


plt.scatter(xpos, ypos)
plt.show()




# cubic polynomial model
def model(x, params):
    a, b, c, d = params
    return a*x**3 + b*x**2 + c*x + d

# sum of squared errors
def loss(params):
    y_pred = model(xpos, params)
    return np.sum((ypos - y_pred)**2)

# Use Nelder-Mead
initial_guess = [1, 1, 1, 1]
result = minimize(loss, initial_guess, method='Nelder-Mead')

print("Optimal parameters:", result.x)

# Calculate predictions
y_pred = model(xpos, result.x)

# Compute R² and Adjusted R² 
ss_res = np.sum((ypos - y_pred)**2)
ss_tot = np.sum((ypos - np.mean(ypos))**2)
r2 = 1 - (ss_res / ss_tot)

n = len(ypos)       # number of data points
p = len(result.x)   # number of parameters
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print(f"R²: {r2:.4f}")
print(f"Adjusted R²: {adj_r2:.4f}")


# Plot the fit
plt.scatter(xpos, ypos, label="Data")
plt.plot(np.sort(xpos), model(np.sort(xpos), result.x), color='red', label="Fit")
plt.legend()
plt.show()
