# Author: A. Nitz

import numpy

from matplotlib import pyplot as plt
from scipy.optimize import rosen

x = numpy.arange(-2, 2, .01)
y = numpy.arange(-1, 3, .01)
X, Y = numpy.meshgrid(x, y)

z = rosen((X, Y))
plt.pcolormesh(X, Y, z, norm='log', vmin=1e-3)
c = plt.colorbar()
plt.show()
