import numpy as np
from scipy.optimize import minimize

from main import Result, rosen

def nelder_mead(
    x0: np.ndarray,
    max_iter: int = 2000,
    tol: float = 1e-10
) -> Result:
    res = minimize(
        rosen, x0, method="Nelder-Mead",
        options={"maxiter": max_iter, "xatol": 1e-12, "fatol": tol, "disp": False}
    )
    return Result(
        method="nelder_mead",
        x0=np.array(x0, dtype=float),
        x=res.x,
        f=float(res.fun),
        iters=int(res.nit),
        success=bool(res.fun <= tol),
        info={"message": res.message}
    )
