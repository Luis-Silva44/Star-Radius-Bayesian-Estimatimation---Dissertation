# %%
import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
from IPython.display import display, Math
# %%

np.random.seed(82)

# Choose the "true" parameters.
m_true = -0.9594
b_true = 4.294

# Generate some synthetic data from the model.
N = 50
x = np.sort(10 * np.random.rand(N))
yerr = 0.1 * np.random.rand(N)
y = m_true * x + b_true
y += yerr 

plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
x0 = np.linspace(0, 10, 500)
plt.plot(x0, m_true * x0 + b_true, "k", alpha=0.3, lw=3)
plt.xlim(0, 10)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
# %%
def log_likelihood(theta, x, y, yerr):
    m, b = theta
    model = m * x + b
    sigma2 = yerr**2
    return (-0.5 * np.sum((y - model) ** 2 / sigma2) + np.sum(np.log(2 * np.pi * sigma2))) 

def log_prior(theta):
    m, b = theta
    m_prior = -0.5 * (m - 1.5)**2 / 0.005**2 
    log_m_prior = -0.5 * ((m - 1.5) / 0.005) ** 2 - np.log(0.005 * np.sqrt(2 * np.pi))  # Added normalization
    if -5.0 < m < 5.0 and 0.0 < b < 10.0:
        return log_m_prior
    return -np.inf

def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)

pos = (1.2, 4.3) + 1e-4 * np.random.randn(32, 2)
nwalkers, ndim = pos.shape

sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability, args=(x, y, yerr)
)
state = sampler.run_mcmc(pos, 500, progress=True)

# %% 
fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["m", "b"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")
flat_samples = sampler.get_chain(flat=True)

fig = corner.corner(
    flat_samples, labels=labels, truths=[m_true, b_true])
# %%
inds = np.random.randint(len(flat_samples), size=100)
for ind in inds:
    sample = flat_samples[ind]
    plt.plot(x0, np.dot(np.vander(x0, 2), sample[:2]), "C1", alpha=0.1)
plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
plt.plot(x0, m_true * x0 + b_true, "k", label="truth")
plt.legend(fontsize=14)
plt.xlim(0, 10)
plt.xlabel("x")
plt.ylabel("y")
# %%
for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    display(Math(txt))

# %%

m_values = np.linspace(-5,5,20)
b_true = 4.294
prior_list = []
posterior_list = []
likelihood_list = []

for m in m_values:
    theta = (m, b_true)
    likelihood_list.append(log_likelihood(theta, x, y, yerr))
    prior_list.append(log_prior(theta))
    posterior_list.append(log_probability(theta, x, y, yerr))
# %%
print(likelihood_list)
plt.plot(m_values, (prior_list))
plt.xlim(-3,5)
plt.show()
plt.plot(m_values, (likelihood_list))
plt.xlim(-3,5)
plt.show()
plt.plot(m_values, (posterior_list))
plt.xlim(-3,5)
plt.show()
# %%
