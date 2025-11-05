import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize  

# Load the data
import h5py
f = h5py.File('./data.hdf', 'r')
xpos = f['data/xpos'][:]
ypos = f['data/ypos'][:]

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

# Plot the fit
plt.scatter(xpos, ypos, label="Data")
plt.plot(np.sort(xpos), model(np.sort(xpos), result.x), color='red', label="Fit")
plt.legend()
plt.show()
