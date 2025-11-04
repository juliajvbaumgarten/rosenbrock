import numpy as np
from gaussian_model import residuals
from scipy.optimize import minimize

def nelder_mead_gaussian(x, y, params0, max_iter=2000, tol=1e-8):
    res = minimize(
        lambda p: np.sum(residuals(p, x, y)**2),
        params0,
        method="Nelder-Mead",
        options={"maxiter": max_iter, "xatol": tol, "fatol": tol, "disp": False}
    )
    return res.x
