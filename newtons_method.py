# to find the minima of the 2d rosenbrock function using Newton's method, we need 
# to calculate the gradient and the hessian. 

# calculating the gradient of the 2d rosenbrock function
def grad_rosenbrock_2d(x):
    df_dx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    df_dy = 200 * (x[1] - x[0]**2)
    return np.array([df_dx, df_dy])
