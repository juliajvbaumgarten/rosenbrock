import numpy as np
from gaussian_model import residuals, jacobian

def newton_gaussian_fit(x, y, params, max_iter=50, tol=1e-8):
    params = np.array(params, dtype=float)
    path = [params.copy()]

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
        path.append(params.copy())
        if params[2] <= 0:
            params[2] = abs(params[2]) + 1e-6
        if np.linalg.norm(step) < tol:
            break

    return params, path
