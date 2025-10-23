# defining the 2d rosenbrock function setting a = 1 and b = 100
def rosenbrock_2d(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
