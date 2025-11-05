#For polynomial model
import numpy as np
import emcee
import matplotlib.pyplot as plt
import h5py

# Load the data
f = h5py.File('./data.hdf', 'r')
xpos = f['data/xpos'][:]
ypos = f['data/ypos'][:]

# Model: cubic polynomial
def model(x, params):
    a, b, c, d = params
    return a*x**3 + b*x**2 + c*x + d

# Log-likelihood assuming Gaussian noise
def log_likelihood(params):
    y_pred = model(xpos, params)
    return -0.5 * np.sum((ypos - y_pred)**2)

#  uniform priors to keep parameters reasonable
def log_prior(params):
    a, b, c, d = params
    if -10 < a < 10 and -10 < b < 10 and -10 < c < 10 and -10 < d < 10:
        return 0.0  
    return -np.inf  #

def log_posterior(params):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params)
ndim = 4        # number of parameters (a, b, c, d)
nwalkers = 500  # ensemble size
nsteps = 1000   # number of iterations

# Initialize walkers in a small Gaussian ball around a random starting point
initial = np.random.randn(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)

print("Running MCMC...")
sampler.run_mcmc(initial, nsteps, progress=True)
print("Done.")
for i in range(100, nsteps+1, 100):
    samples = sampler.get_chain(discard=0, thin=1, flat=False)[:i,:,:]
    flat_samples = samples.reshape(-1, ndim)
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.flatten()
    labels = ['a', 'b', 'c', 'd']
    
    for j in range(ndim):
        axs[j].hist(flat_samples[:, j], bins=30, color='skyblue', edgecolor='k')
        axs[j].set_title(f"Parameter {labels[j]} after {i} steps")
    
    plt.tight_layout()
    plt.show()

