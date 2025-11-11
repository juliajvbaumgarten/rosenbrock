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

# After MCMC is done
samples = sampler.get_chain(discard=100, thin=10, flat=True)
labels = ['a', 'b', 'c', 'd']

# Corner plot
fig = corner.corner(samples, labels=labels, truths=None, show_titles=True)
plt.show()

def direct_autocorr(chain, max_lag=None):
    """ This directly calculates the autocorrelation time / length
    using numpy functions. Note there are some heuristic steps.
    """
    # We'll assume that the first half of the chain is "burn-in"
    chain = chain[len(chain)//2:]
    
    n = len(chain)
    if max_lag is None:
        max_lag = n // 10 # practical limit for estimation


    y = chain - np.mean(chain)
    # mode='full' gives 2*n-1 points. The second half (starting at n-1) is positive lags.
    c = np.correlate(y, y, mode='full')
    c = c[n-1 : n-1+max_lag] # Take positive lags up to max_lag
    rho = c / c[0] # Normalize so rho[0] is 1.0

    negative_indices = np.where(rho < 0)[0]
    if len(negative_indices) > 0:
        cutoff = negative_indices[0]
    else:
        cutoff = len(rho)

    tau_estimate = 1.0 + 2.0 * np.sum(rho[1:cutoff])
    return tau_estimate, rho

chain = sampler.get_chain(discard=100, thin=1, flat=False)
nwalkers, nsteps, ndim = chain.shape


walker_idx = 0  # choose a specific walker
for j, label in enumerate(['a', 'b', 'c', 'd']):
    param_chain = chain[walker_idx, :, j]
    tau_emcee = emcee.autocorr.integrated_time(param_chain, quiet=True)[0]
    tau_direct, rho = direct_autocorr(param_chain)
    print(f"{label}: emcee ACL = {tau_emcee:.2f}, direct ACL = {tau_direct:.2f}")



phis = [0.3, 0.6, 0.9]
for phi in phis:
    chain_phi = generate_simulated_chain(10000, phi)
    theory_acl = (1 + phi) / (1 - phi)
    tau_est, _ = direct_autocorr(chain_phi)
    print(f"phi={phi}: Theoretical={theory_acl:.2f}, Estimated={tau_est:.2f}")

   
acl_emcee_all = []
for j, label in enumerate(['a', 'b', 'c', 'd']):
    taus = []
    for w in range(nwalkers):
        param_chain = chain[w, :, j]
        tau = emcee.autocorr.integrated_time(param_chain, quiet=True)[0]
        taus.append(tau)
    acl_emcee_all.append(taus)
    print(f"{label}: mean ACL = {np.mean(taus):.2f}, std = {np.std(taus):.2f}")

fig, axs = plt.subplots(4, 1, figsize=(10, 8))
labels = ['a', 'b', 'c', 'd']
for j in range(ndim):
    axs[j].plot(chain[:, :, j], alpha=0.3)
    axs[j].set_ylabel(labels[j])
axs[-1].set_xlabel("Step")
plt.tight_layout()
plt.show()

burnin = 300
chain_post = chain[:, burnin:, :]
acl_guess = 100  # assume 1/10th chain length if ACL ≈ chain length

within_var = []
final_samples = []

for w in range(nwalkers):
    sampled = chain_post[w, ::acl_guess, :]  # one sample every ACL
    var_within = np.var(sampled, axis=0)
    within_var.append(var_within)
    final_samples.append(chain_post[w, -1, :])  # final sample

within_var = np.array(within_var)
final_samples = np.array(final_samples)

print("Average within-chain variance:", np.mean(within_var, axis=0))
print("Variance across final samples:", np.var(final_samples, axis=0))
