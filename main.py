# Author: Lauren Sdun, Julia Jones, Julia Baumgarten

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, Tuple, Dict, Any, List
from numpy.linalg import norm
from scipy.optimize import minimize

from bruteandgrad import *
from newtons_method import *
from nelder_mead import *
from main import Result, rosen

@dataclass
class Result:
    method: str
    x0: np.ndarray          # starting point (shape (2,))
    x: np.ndarray           # final point (shape (2,))
    f: float                # final function value
    iters: int              # iterations performed
    success: bool           # did we hit
    info: Dict[str, Any]    

def rosen(xy: np.ndarray) -> float:
    x, y = xy
    return (1 - x)**2 + 100.0 * (y - x**2)**2

def rosen_grad(v: np.ndarray) -> np.ndarray:
    x, y = v
    dfx = -2*(1 - x) - 400*x*(y - x**2)
    dfy = 200*(y - x**2)
    return np.array([dfx, dfy])

def rosen_hess(v: np.ndarray) -> np.ndarray:
    x, y = v
    return np.array([
        [2 - 400*(y - 3*x**2), -400*x],
        [-400*x,                200    ],
    ])


# Configuration
def run_all(corners: List[Tuple[float,float]] = [(-2,-1), (-2,3), (2,-1), (2,3)],
            gd_lrs=(1e-4, 1e-3, 1e-2),
            use_backtracking=True,
            tol=1e-10):
    results = []

    # Brute force single sweep (no path)
    bf = brute_force()
    results.append(bf)

    # Gradient descent (several step sizes) from each corner
    for x0 in corners:
        for lr in gd_lrs:
            r = gradient_descent(x0, lr=lr, max_iter=200000, tol_f=tol, use_backtracking=use_backtracking)
            results.append(r)

    # Newton
    for x0 in corners:
        r = newton_method(np.array(x0), max_iter=500, tol=tol)
        results.append(r)

    # Nelder–Mead
    for x0 in corners:
        r = nelder_mead(np.array(x0), max_iter=5000, tol=tol)
        results.append(r)

    # Print report
    print("\n=== Summary (stop criterion: f(x) < 1e-10) ===")
    for r in results:
        print(f"{r.method:28s}  x0={np.array2string(r.x0, precision=2, suppress_small=True)}  "
              f"iters={r.iters:6d}  f*={r.f: .3e}  x*={np.array2string(r.x, precision=6)}  success={r.success}")

    return results
