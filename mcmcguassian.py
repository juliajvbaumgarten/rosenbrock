import numpy as np
import emcee
from matplotlib import pyplot as plt
import h5py

# Load the data
f = h5py.File('./data.hdf', 'r')
xpos = f['data/xpos'][:]
ypos = f['data/ypos'][:]

# Gaussian model

def gaussian(x, A, mu, sigma, C):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + C


# Log-prior
def log_prior(params):
    A, mu, sigma, C = params
    if 0 < A < 10 and np.min(xpos) < mu < np.max(xpos) and 0 < sigma < 10 and -10 < C < 10:
        return 0.0  # flat prior in allowed ranges
    return -np.inf


# Log-likelihood (Gaussian noise, unit variance)
def log_likelihood(params):
    A, mu, sigma, C = params
    model = gaussian(xpos, A, mu, sigma, C)
    return -0.5 * np.sum((ypos - model) ** 2) 


# Log-posterior
def log_posterior(params):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params)


# MCMC parameters
ndim = 4       # number of parameters: A, mu, sigma, C
nwalkers = 500
nsteps = 1000


# Initialize walkers near a rough guess
# Rough guesses: A=1, mu=mean(x), sigma=1, C=0
initial_guess = np.array([1, np.mean(xpos), 1, 0])
initial = initial_guess + 0.01 * np.random.randn(nwalkers, ndim)


# Create the sampler and run
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)

print("Running MCMC...")
sampler.run_mcmc(initial, nsteps, progress=True)
print("Done.")


# Plot histograms every 100 steps
for i in range(100, nsteps+1, 100):
    samples = sampler.get_chain(discard=0, thin=1, flat=False)[:i,:,:]
    flat_samples = samples.reshape(-1, ndim)
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.flatten()
    labels = ['A', 'mu', 'sigma', 'C']
    
    for j in range(ndim):
        axs[j].hist(flat_samples[:, j], bins=30, color='skyblue', edgecolor='k')
        axs[j].set_title(f"Parameter {labels[j]} after {i} steps")
    
    plt.tight_layout()
    plt.show()
