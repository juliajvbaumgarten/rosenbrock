import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple, Dict, Any
from numpy.linalg import norm
from scipy.optimize import minimize

def rosen(xy: np.ndarray) -> float:
    x, y = xy
    return (1 - x)**2 + 100.0 * (y - x**2)**2

# Configuration

X_RANGE = (-2.0, 2.0)
Y_RANGE = (-1.0, 3.0)

@dataclass
class Result:
    method: str
    x0: np.ndarray
    x: np.ndarray
    f: float
    iters: int
    success: bool
    info: Dict[str, Any]

