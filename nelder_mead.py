# Author: Lauren Sdun, Julia Jones, Julia Baumgarten

def nelder_mead(x0: np.ndarray, max_iter: int = 2000, tol: float = 1e-10) -> Result:
    res = minimize(rosen, x0, method="Nelder-Mead",
                   options={"maxiter": max_iter, "xatol": tol, "fatol": tol, "disp": False})
    return Result("nelder_mead", x0, res.x, res.fun, res.nit, res.success, {"message": res.message})
