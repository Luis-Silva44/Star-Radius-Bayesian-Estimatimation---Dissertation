# %% 
from gaia_module import gaia_values
from SED_fitting import get_flux_values
from SED_flux import *
from graphs_visualization import computed_real_comparison
# %%
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.constants import R_sun
import astropy.units as u
import emcee as emcee
import multiprocessing
import corner
from IPython.display import display, Math
import time
# %%
def likelihood(params, obs_flux, obs_flux_unc, log_g, metallicity, Ebv, filter_wavelen):
        distance, temperature, radius = params

        distance2 = (distance * u.pc).to(R_sun)
        _, model_flux = SED_flux_bands(filter_wavelen, temperature, metallicity, log_g, Ebv)
        model_flux_scaled = model_flux.value * (radius / distance2.value)**2

        c = np.log(2 * np.pi * obs_flux_unc**2)
        return -0.5 * np.sum(c + ((obs_flux - model_flux_scaled)**2 / obs_flux_unc**2))

def prior(params, exp_distance, exp_temp, temp_unc):
        distance, temperature, radius = params

        if not (0.1 < radius < 10.0) or not (3500 < temperature < 10000):
            return -np.inf
        
        temperature_prior = -0.5 * ((exp_temp - temperature) / temp_unc)**2
        distance_prior = -0.5 * ((distance - exp_distance.value.nominal_value) / exp_distance.value.std_dev)**2
        return distance_prior + temperature_prior

def posterior(params, obs_flux, obs_flux_unc, log_g, metallicity, Ebv, filter_wavelen, exp_distance, exp_temp, temp_unc):
        log_prior = prior(params, exp_distance, exp_temp, temp_unc)
        if not np.isfinite(log_prior):
            return -np.inf
        return log_prior + likelihood(params, obs_flux, obs_flux_unc, log_g, metallicity, Ebv, filter_wavelen)
# %%
def basic_MCMC_temperature(exp_values, nwalkers, obs_flux, obs_flux_unc, log_g, metallicity, Ebv, filter_wavelen):
    exp_distance, exp_temp, expected_radius, temp_unc = exp_values 

    pos = np.array([exp_distance.value.nominal_value + exp_distance.value.std_dev * np.random.randn(nwalkers), 
                np.random.normal(5000, 1500, size=nwalkers),
                np.random.normal(1.0, 0.5, size=nwalkers)]).T

    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(nwalkers, ndim, posterior, args=(obs_flux, obs_flux_unc, log_g, metallicity, Ebv, filter_wavelen, exp_distance, exp_temp, temp_unc), pool = multiprocessing.Pool(16), moves=emcee.moves.DEMove())
    state = sampler.run_mcmc(pos, 1200, progress=True)

    labels = ["Distance", "Temperature", "Radius"]
    fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain(discard = 500)

    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")

    flat_samples = sampler.get_chain(discard = 500, flat=True)
    fig = corner.corner(flat_samples, labels=labels)
    plt.show()

    expected_values = [exp_distance.value.nominal_value, exp_temp, expected_radius.value]
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], labels[i])
        display(Math(txt))
        print('Error in ', labels[i], '=', abs(mcmc[1] - expected_values[i]) / expected_values[i] * 100)
        if i == 2:
            return (mcmc[1] * R_sun).to(R_sun), sampler
# %%
star_name = 'HD 49674'
exp_temp = 5662
temp_unc = 28
table_value = (1.022 * R_sun).to(R_sun)

Ebv = 0.028
log_g = 4.42
metallicity = 0.3

_, parallax, _= gaia_values(star_name)
unit_change = 1 * u.parsec
exp_distance = (1 / parallax.value) * unit_change

filter_wavelen, flux_values = get_flux_values(star_name)
obs_flux = np.array([m.value.nominal_value for m in flux_values])
obs_flux_unc = np.array([m.value.std_dev for m in flux_values])

exp_values = (exp_distance, exp_temp, table_value, temp_unc)
# %% 
comp_radius, sampler = basic_MCMC_temperature(exp_values, 12, obs_flux, obs_flux_unc, log_g, metallicity, Ebv, filter_wavelen)

# %%
