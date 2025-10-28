# Author: Lauren Sdun, Julia Jones, Julia Baumgarten

import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple, Dict, Any, List
from numpy.linalg import norm
from scipy.optimize import minimize

from nelder_mead import *
from newtons_method import * 
from bruteandgrad import *

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

# Configuration

X_RANGE = (-2.0, 2.0)
Y_RANGE = (-1.0, 3.0)

def random_starts(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(X_RANGE[0], X_RANGE[1], size=n)
    ys = rng.uniform(Y_RANGE[0], Y_RANGE[1], size=n)
    return [np.array([x, y]) for x, y in zip(xs, ys)]

def run_all(n_starts: int = 10, seed: int = 0):
    starts = random_starts(n_starts, seed)
    results = {"grid": [grid_search()]}
    results["gradient_descent"] = []
    results["newton"] = []
    results["nelder_mead"] = []

    for x0 in starts:
        results["gradient_descent"].append(gradient_descent(x0))
        results["newton"].append(newtons_method(x0))
        results["nelder_mead"].append(nelder_mead(x0))

    # picks best per method
    bests = []
    for m, arr in results.items():
        best = min(arr, key=lambda r: r.f)
        bests.append(best)

    # prints report
    print("\n=== Best per method (across starts) ===")
    for r in bests:
        print(f"{r.method:16s}  f*={r.f: .6e}  x*={r.x}  iters={r.iters:4d}  success={r.success}  info={r.info}")

    # inspects raw results
    return results, bests

if __name__ == "__main__":
    results, bests = run_all(n_starts=12, seed=42)

