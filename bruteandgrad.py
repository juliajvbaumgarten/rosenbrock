import numpy as np

from matplotlib import pyplot as plt
from scipy.optimize import rosen

x = np.arange(-2, 2, .01)
y = np.arange(-1, 3, .01)
X, Y = np.meshgrid(x, y)

z = rosen((X, Y))
plt.pcolormesh(X, Y, z, norm='log', vmin=1e-3)
c = plt.colorbar()
plt.show(block=False)


min_index = np.unravel_index(np.argmin(z), z.shape)
x_min_brute = X[min_index]
y_min_brute = Y[min_index]
z_min_brute = z[min_index]

print(f"Brute-force minimum approx: x={x_min_brute:.4f}, y={y_min_brute:.4f}, f={z_min_brute:.6f}")
