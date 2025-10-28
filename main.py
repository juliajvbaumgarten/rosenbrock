# Author: Lauren Sdun, Julia Jones, Julia Baumgarten

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, Tuple, Dict, Any, List
from numpy.linalg import norm
from scipy.optimize import minimize

from common import Result, rosen
from bruteandgrad import brute_force, gradient_descent
from newtons_method import newton_method
from nelder_mead import nelder_mead

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


def plot_paths(results: List[Result], xlim=(-2,2), ylim=(-1,3), title="Rosenbrock iteration paths"):
    # Plot contours
    xs = np.linspace(xlim[0], xlim[1], 400)
    ys = np.linspace(ylim[0], ylim[1], 400)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    Z = (1 - X)**2 + 100.0*(Y - X**2)**2

    plt.figure(figsize=(8,6))
    cs = plt.contour(X, Y, Z, levels=np.logspace(-3, 3, 20))
    plt.clabel(cs, inline=1, fontsize=8)

    # Plot paths (no brute force)
    for r in results:
        path = r.info.get("path", None)
        if path is None:
            continue
        P = np.vstack(path)
        plt.plot(P[:,0], P[:,1], marker='o', markersize=2, linewidth=1, label=r.method)

    plt.plot([1.0], [1.0], marker='*', markersize=12, label="global min (1,1)")
    plt.xlim(*xlim); plt.ylim(*ylim)
    plt.xlabel("x"); plt.ylabel("y"); plt.title(title)
    plt.legend(loc="upper right", fontsize=8, ncols=1)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    res = run_all()
    # Only plot non-brute paths
    plot_paths([r for r in res if r.method != "brute_force"])

