# to find the minima of the 2d rosenbrock function using Newton's method, we need 
# to calculate the gradient and the hessian. 

# calculating the gradient of the 2d rosenbrock function
def grad_rosenbrock_2d(x):
    df_dx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    df_dy = 200 * (x[1] - x[0]**2)
    return np.array([df_dx, df_dy])

# calculating the hessian (2x2 matrix of 2nd derivatives)
def hessian_rosenbrock_2d(x):
    H = np.zeros((2, 2)) # create an empty 2x2 matrix
    H[0, 0] = 2 - 400 * x[1] + 1200 * x[0]**2 # d^2f/dx^2
    H[0, 1] = -400 * x[0] # d^2f/dxdy
    H[1, 0] = -400 * x[0] # d^2f/dydx
    H[1, 1] = 200 # d^2f/dy^2
    return H
