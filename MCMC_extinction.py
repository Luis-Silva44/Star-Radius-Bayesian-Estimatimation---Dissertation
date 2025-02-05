# %%
from gaia_module import gaia_values
from SED_fitting import get_flux_values
from SED_flux import *
# %%
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.constants import R_sun
import emcee as emcee
import multiprocessing
import corner
from IPython.display import display, Math

# %%
star_name = 'HD 136352'

Teff = 5664
log_g = 4.39
metallicity = -0.34

expected_Ebv = 0.010
expected_radius = 1.054

_, parallax, _= gaia_values(star_name)
unit_change = 1 * u.parsec
exp_distance = (1 / parallax.value) * unit_change

filter_wavelen, flux_values = get_flux_values(star_name)
obs_flux = np.array([m.value.nominal_value for m in flux_values])
obs_flux_unc = np.array([m.value.std_dev for m in flux_values])

SED_wavelen, SED_flux = SED_interpolator(Teff,metallicity,log_g)
nearest_index = []
for i in range(len(filter_wavelen)):
    nearest_index.append(find_nearest_index(SED_wavelen, filter_wavelen[i]))

model_flux = np.array([SED_flux[i].value for i in nearest_index])
# %%

def likelihood(params, obs_flux, obs_flux_unc, model_flux, filter_wavelen):
    distance, Ebv, radius = params
    distance2 = (distance * u.pc).to(R_sun)
    filter_wavelen_ = filter_wavelen.astype(np.float64)
    flux_attenuated = flux_extinction(filter_wavelen_, model_flux, Ebv)
    model_flux_scaled = flux_attenuated * (radius / distance2.value)**2

    c = np.log(2 * np.pi * obs_flux_unc**2)
    return -0.5 * np.sum(c + ((obs_flux - model_flux_scaled)**2 / obs_flux_unc**2))

def prior(params):
    distance, Ebv, radius = params

    if not (0.1 < radius < 10.0) and (0.0 < Ebv < 0.50):
        return -np.inf
    
    distance_prior = -0.5 * ((distance - exp_distance.value.nominal_value) / exp_distance.value.std_dev)**2
    return distance_prior

def posterior(params, obs_flux, obs_flux_unc, model_flux, filter_wavelen):
    log_prior = prior(params)
    if not np.isfinite(log_prior):
        return -np.inf
    
    return log_prior + likelihood(params, obs_flux, obs_flux_unc, model_flux, filter_wavelen)
# %%
pos = np.array([exp_distance.value.nominal_value + exp_distance.value.std_dev * np.random.randn(10), 
                np.random.normal(0.1, 0.005, size=10),
                np.random.normal(1.0, 0.5, size=10)]).T

nwalkers, ndim = pos.shape

sampler = emcee.EnsembleSampler(nwalkers, ndim, posterior, args=(obs_flux, obs_flux_unc, model_flux, filter_wavelen), pool = multiprocessing.Pool(16), moves=emcee.moves.DEMove())
state = sampler.run_mcmc(pos, 2500, progress=True)

# %%# %% 

print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))
#print("Mean autocorrelation time:", np.mean(sampler.get_autocorr_time()))

# %% 
fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
samples = sampler.get_chain(discard=2000)
labels = ["Distance", "E(B-V)", "Radius"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")
# %%
flat_samples = sampler.get_chain(discard=2000, flat=True)
fig = corner.corner(
    flat_samples, labels=labels)

# %% 
expected_values = [exp_distance.value.nominal_value, expected_Ebv, expected_radius]
for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    display(Math(txt))
    print('Error in ', labels[i], '=', abs(mcmc[i] - expected_values[i]) / expected_values[i] * 100)

# %%
