from dataclasses import dataclass
from typing import Dict, Any
import numpy as np

@dataclass
class Result:
    method: str
    x0: np.ndarray
    x: np.ndarray
    f: float
    iters: int
    success: bool
    info: Dict[str, Any]

def rosen(v: np.ndarray) -> float:
    x, y = v
    return (1 - x)**2 + 100.0*(y - x**2)**2

def rosen_grad(v: np.ndarray) -> np.ndarray:
    x, y = v
    dfx = -2*(1 - x) - 400*x*(y - x**2)
    dfy = 200*(y - x**2)
    return np.array([dfx, dfy])

def rosen_hess(v: np.ndarray) -> np.ndarray:
    x, y = v
    return np.array([
        [2 - 400*(y - 3*x**2), -400*x],
        [-400*x,               200    ],
    ])
