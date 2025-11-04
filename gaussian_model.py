import numpy as np

def gaussian(x, A, mu, sigma, C):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + C

def residuals(params, x, y):
    A, mu, sigma, C = params
    return y - gaussian(x, A, mu, sigma, C)

def jacobian(params, x):
    A, mu, sigma, C = params
    g = A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    
    dA = g / (A + 1e-8)
    dmu = g * ((x - mu) / (sigma**2))
    dsigma = g * (((x - mu) ** 2) / (sigma**3))
    dC = np.ones_like(x)
    
    J = np.vstack((-dA, -dmu, -dsigma, -dC)).T  # negative because r = y - g
    return J

