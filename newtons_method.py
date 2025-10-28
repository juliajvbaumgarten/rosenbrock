import numpy as np
from scipy.optimize import minimize
from common import Result, rosen, rosen_grad, rosen_hess

def newton_method(x0, max_iter=1000, tol=1e-10):

    res = minimize(
        rosen,
        x0,
        method='Newton-CG',
        jac=rosen_grad,
        hess=rosen_hess,
        options={'maxiter': max_iter, 'xtol': tol, 'disp': False}
    )

    return Result(
        method="newton",
        x0=np.array(x0, dtype=float),
        x=res.x,
        f=float(res.fun),
        iters=int(res.nit),
        success=bool(res.fun <= tol),
        info={"message": res.message, "path": [res.x]}
    )
