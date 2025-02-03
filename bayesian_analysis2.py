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
import multiprocessing
import corner
from IPython.display import display, Math

# %%

star_name = 'WASP-8'
Ebv = 0.085
table_value = 1.045

Teff_mean = 5690
Teff_unc = 36
log_g_mean = 4.42
log_g_unc = 0.15
metallicity_mean = 0.29
metallicity_unc = 0.03


_, parallax, _= gaia_values(star_name)
unit_change = 1 * u.parsec
exp_distance = (1 / parallax.value) * unit_change

filter_wavelen, flux_values = get_flux_values(star_name)
obs_flux = np.array([m.value.nominal_value for m in flux_values])
obs_flux_unc = np.array([m.value.std_dev for m in flux_values])

#_, model_flux = SED_flux_bands(filter_wavelen, Teff_mean, metallicity_mean, log_g_mean, Ebv)
#model_flux = np.array(model_flux.value)
# %%

def likelihood(params, obs_flux, obs_flux_unc, filter_wavelen):
    try:
        Teff, metallicity, log_g, distance, radius = params
        distance2 = (distance * u.pc).to(R_sun)
        _, model_flux = SED_flux_bands(filter_wavelen, Teff, metallicity, log_g, Ebv)
        
        model_flux_scaled = model_flux.value * (radius / distance2.value)**2
        
        c = np.log(2 * np.pi * obs_flux_unc**2)
        return -0.5 * np.sum(c + ((obs_flux - model_flux_scaled)**2 / obs_flux_unc**2))
    except:
        return -np.inf

def prior(params):
    Teff, metallicity, log_g, distance, radius = params

    if not 0.1 < radius < 10.0:
        return -np.inf
    
    Teff_prior = -0.5 * ((Teff - Teff_mean) / Teff_unc)**2
    metallicity_prior = -0.5 * ((metallicity - metallicity_mean) / metallicity_unc)**2
    log_g_prior = -0.5 * ((log_g - log_g_mean) / log_g_unc)**2
    distance_prior = -0.5 * ((distance - exp_distance.value.nominal_value) / exp_distance.value.std_dev)**2
    return Teff_prior + metallicity_prior + log_g_prior + distance_prior

def posterior(params, obs_flux, obs_flux_unc, model_flux):
    log_prior = prior(params)
    if not np.isfinite(log_prior):
        return -np.inf
    return log_prior + likelihood(params, obs_flux, obs_flux_unc, model_flux)

# %%

#pos = (Teff_mean, metallicity_mean, log_g_mean, exp_distance.value.nominal_value,1.0) + 1e-4 * np.random.randn(10, 5)
pos = np.array([Teff_mean + np.random.randn(10),
                metallicity_mean + 1e-4 * np.random.randn(10),
                log_g_mean + 1e-3 * np.random.randn(10),
                exp_distance.value.nominal_value + 1e-1 * np.random.randn(10),
                1.0 + 1e-2 * np.random.randn(10)]).T

nwalkers, ndim = pos.shape
# %% 
sampler = emcee.EnsembleSampler(nwalkers, ndim, posterior, args=(obs_flux, obs_flux_unc, filter_wavelen), pool = multiprocessing.Pool(16), moves=emcee.moves.DEMove())
sampler.run_mcmc(pos, 700, progress=True)
# %%
print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))
print("Mean autocorrelation time:", np.mean(sampler.get_autocorr_time()))
# %% 
fig, axes = plt.subplots(5, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["Teff", "Metallicity", "Log_g", "Distance", "Radius"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")
# %%
flat_samples = sampler.get_chain(discard=200, flat=True)
fig = corner.corner(
    flat_samples, labels=labels)
# %%

for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    display(Math(txt))
















































