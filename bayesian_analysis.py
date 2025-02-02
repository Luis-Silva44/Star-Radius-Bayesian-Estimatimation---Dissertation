# %% 
from gaia_module import gaia_values
from SED_fitting import get_flux_values
from SED_flux import SED_flux_bands
# %%
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.constants import R_sun
import emcee
import scipy.stats as stats
import multiprocessing
import corner
#  %% 
star_name = 'WASP-84'
Ebv = 0.020
table_value = 0.828

Teff_mean = 5221
Teff_std = 72

log_g_mean = 4.28
log_g_std = 0.13

metallicity_mean = 0.13 
metallicity_std = 0.05

_, parallax, _= gaia_values(star_name)
unit_change = 1 * u.parsec
distance1 = (1 / parallax.value) * unit_change
distance = distance1.to(R_sun)
distance_value = distance.value.nominal_value
distance_std = distance.value.std_dev

filter_wavelen, flux_values_Jy = get_flux_values(star_name)
flux_values = flux_values_Jy.value
flux_vals = [m.nominal_value for m in flux_values]
flux_unc = [m.std_dev for m in flux_values]
#%%
def log_likelihood(params, obs_flux, obs_flux_unc, filter_wavelen):
    try:
        Teff, log_g, metallicity, distance, radius = params
        #print(Teff, log_g, metallicity, distance, radius)
        filter_wavelen, model_flux = SED_flux_bands(filter_wavelen, Teff, metallicity, log_g, Ebv)
        model_flux = model_flux * (radius/distance)**2
        c = np.log(2 * np.pi * obs_flux_unc**2)
        return -0.5 * (c + ((model_flux.value - obs_flux)) ** 2 / obs_flux_unc**2).sum()
    except:
        return -np.inf
    
def log_likelihood2(params, obs_flux, obs_flux_unc, filter_wavelen):
    try:
        Teff, log_g, metallicity, distance, radius = params
        #print(Teff, log_g, metallicity, distance, radius)
        filter_wavelen, model_flux = SED_flux_bands(filter_wavelen, Teff, metallicity, log_g, Ebv)
        model_flux = model_flux * (radius/distance)**2
        return ((model_flux.value - obs_flux) ** 2 / obs_flux_unc**2).sum()
    except:
        return -np.inf
    
def uniform_prior(theta):
    Teff, log_g, metallicity, radius, distance = theta

    if (3500 < Teff < 8000) and (0.0 < log_g < 5.0) and (-2.5 < metallicity < 0.5) and (.1 < radius < 1000) and (1.0 < distance < 1e300):
        return 0.0
    return -np.inf

def log_prior(theta):
    Teff, log_g, metallicity, radius, distance = theta
    
    if not (3500 < Teff < 8000): return -np.inf
    if not (0.0 < log_g < 5.0): return -np.inf
    if not (-2.5 < metallicity < 0.5): return -np.inf
    if not (0.1 < radius < 1000): return -np.inf

    prior = 0
    prior += stats.norm(Teff_mean, Teff_std).logpdf(Teff)
    prior += stats.norm(log_g_mean, log_g_std).logpdf(log_g)
    prior += stats.norm(metallicity_mean, metallicity_std).logpdf(metallicity)
    prior += stats.norm(distance_value, distance_std).logpdf(distance) 
    return prior

def log_posterior(params, obs_flux, obs_flux_unc, filter_wavelen):
    prior = log_prior(params)
    prior2 = uniform_prior(params)
    if np.isinf(prior2):
        return -np.inf
    
    return prior2 + log_likelihood2(params, obs_flux, obs_flux_unc, filter_wavelen)

# Setup for emcee
nwalkers = 10  # Number of walkers
ndim = 5 
nsteps = 100  # Number of MCMC steps

p0 = np.random.randn(nwalkers, ndim)
p0[:, 0] = np.random.normal(Teff_mean,Teff_std,nwalkers)
p0[:, 1] = np.random.normal(log_g_mean,log_g_std,nwalkers)
p0[:, 2] = np.random.normal(metallicity_mean,metallicity_std,nwalkers)
p0[:, 4] = np.random.normal(distance_value,distance_std,nwalkers)
p0[:, 3] = 1.0 + 0.5 * np.random.randn(nwalkers)  

# %% 
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(flux_vals, flux_unc, filter_wavelen), pool=multiprocessing.Pool(12))
state = sampler.run_mcmc(p0, nsteps, progress=True)
# %% 
samples = sampler.get_chain()

fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
labels = ["Teff", "log_g", "metallicity", "radius", "distance"]
for i in range(ndim):
    axes[i].plot(samples[:, :, i], color="k", alpha=0.3)
    axes[i].set_ylabel(labels[i])
axes[-1].set_xlabel("Step number")
plt.show()


flat_samples = sampler.get_chain(discard = 40, thin=1, flat=True)
for i in range(ndim):
    print(f"{labels[i]}: {np.mean(flat_samples[:, i])} ± {np.std(flat_samples[:, i])}")

fig = corner.corner(flat_samples, labels=labels, truths=[Teff_mean, log_g_mean, metallicity_mean, table_value, distance_value])
# %%
