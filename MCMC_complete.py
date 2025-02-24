# %% 
from gaia_module import gaia_values
from SED_fitting import get_flux_values
from SED_flux import *
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
def likelihood(params, obs_flux, obs_flux_unc, filter_wavelen):
        distance, temperature, log_g, metallicity, Ebv, radius = params
        distance2 = (distance * u.pc).to(R_sun)
        SED_wavelen, model_flux = SED_interpolator(temperature, metallicity, log_g)
        nearest_index = []
        for i in range(len(filter_wavelen)):
            nearest_index.append(find_nearest_index(SED_wavelen, filter_wavelen[i]))

        model_flux = np.array([model_flux[i].value for i in nearest_index]) 

        filter_wavelen_ = filter_wavelen.astype(np.float64)
        flux_attenuated = flux_extinction(filter_wavelen_, model_flux, Ebv)
        model_flux_scaled = flux_attenuated * (radius / distance2.value)**2

        c = np.log(2 * np.pi * obs_flux_unc**2)
        return -0.5 * np.sum(c + ((obs_flux - model_flux_scaled)**2 / obs_flux_unc**2))

def prior(params, exp_distance, exp_temperature, temp_unc, exp_log, log_unc, exp_met, met_unc):
        distance, temperature, log_g, metallicty, Ebv, radius = params

        if not (0.1 < radius < 10.0) or not (0.0 < Ebv < 0.5) or not(3500 < temperature < 10000) or not (-2.5 < metallicty < 0.5) or not (0 < log_g < 5.0):
            return -np.inf
        
        distance_prior = -0.5 * ((distance - exp_distance.value.nominal_value) / exp_distance.value.std_dev)**2
        temperature_prior = -0.5*((temperature - exp_temperature) / temp_unc)**2
        log_g_prior = -0.5 *((log_g - exp_log) / log_unc)**2
        met_prior = -0.5 * ((metallicty - exp_met) / met_unc)**2
        return distance_prior + temperature_prior + log_g_prior + met_prior

def posterior(params, obs_flux, obs_flux_unc, exp_distance, exp_temperature, temp_unc, exp_log, log_unc, exp_met, met_unc, filter_wavelen):
        log_prior = prior(params, exp_distance, exp_temperature, temp_unc, exp_log, log_unc, exp_met, met_unc)
        if not np.isfinite(log_prior):
            return -np.inf
    
        return log_prior + likelihood(params, obs_flux, obs_flux_unc, filter_wavelen)

# %% 
def complete_MCMC(exp_values, nwalkers, obs_flux, obs_flux_unc, filter_wavelen):
    exp_distance, exp_temperature, temp_unc, exp_log, log_unc, exp_met, met_unc, exp_Ebv, exp_radius = exp_values 

    pos = np.array([exp_distance.value.nominal_value + exp_distance.value.std_dev * np.random.randn(nwalkers), 
                    exp_temperature + temp_unc * np.random.randn(nwalkers),
                    exp_log + log_unc * np.random.randn(nwalkers),
                    exp_met + met_unc * np.random.randn(nwalkers),
                    np.abs(np.random.normal(0.05, 0.05, size=nwalkers)),
                    np.random.normal(1.0, 0.5, size=nwalkers)]).T

    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(nwalkers, ndim, posterior, args=(obs_flux, obs_flux_unc, exp_distance, exp_temperature, temp_unc, exp_log, log_unc, exp_met, met_unc, filter_wavelen), pool = multiprocessing.Pool(12))
    state = sampler.run_mcmc(pos, 1500, progress=True)

    labels = ["Distance", "Temperature", "Log g", "Metallicity", "E(B-V)", "Radius"]
    fig, axes = plt.subplots(6, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()

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

    expected_values = [exp_distance.value.nominal_value, exp_temperature, exp_log, exp_met, exp_Ebv, exp_radius.value]
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], labels[i])
        display(Math(txt))
        print('Error in ', labels[i], '=', abs(mcmc[1] - expected_values[i]) / expected_values[i] * 100)
        if i == 5:
            return (mcmc[1] * R_sun).to(R_sun), sampler
# %%
star_name = 'WASP-84'
expected_Ebv = 0.020
table_value = (0.828 * R_sun).to(R_sun)

exp_Teff = 5221
Teff_unc = 72
log_g = 4.28
log_unc = 0.13
metallicity = 0.05
met_unc = 0.05

_, parallax, _= gaia_values(star_name)
unit_change = 1 * u.parsec
exp_distance = (1 / parallax.value) * unit_change

filter_wavelen, flux_values = get_flux_values(star_name)
obs_flux = np.array([m.value.nominal_value for m in flux_values])
obs_flux_unc = np.array([m.value.std_dev for m in flux_values])


exp_values = (exp_distance, exp_Teff, Teff_unc, log_g, log_unc, metallicity, met_unc, expected_Ebv, table_value)

_, sampler1 = complete_MCMC(exp_values, 12, obs_flux, obs_flux_unc, filter_wavelen)

# %%
star_name = 'HD128582'
expected_Ebv = 0.008
table_value = (1.63 * R_sun).to(R_sun)

exp_Teff = 6168
Teff_unc = 29
log_g = 4.17
log_unc = 0.08
metallicity = 0.098
met_unc = 0.09

_, parallax, _= gaia_values(star_name)
unit_change = 1 * u.parsec
exp_distance = (1 / parallax.value) * unit_change

filter_wavelen, flux_values = get_flux_values(star_name)
obs_flux = np.array([m.value.nominal_value for m in flux_values])
obs_flux_unc = np.array([m.value.std_dev for m in flux_values])


exp_values = (exp_distance, exp_Teff, Teff_unc, log_g, log_unc, metallicity, met_unc, expected_Ebv, table_value)

_, sampler2 = complete_MCMC(exp_values, 12, obs_flux, obs_flux_unc, filter_wavelen)

# %% 
star_name = 'HD 49674'
expected_Ebv = 0.028
table_value = (1.022 * R_sun).to(R_sun)

exp_Teff = 5662
Teff_unc = 72
log_g = 4.42
log_unc = 0.05
metallicity = 0.3
met_unc = 0.02

_, parallax, _= gaia_values(star_name)
unit_change = 1 * u.parsec
exp_distance = (1 / parallax.value) * unit_change

filter_wavelen, flux_values = get_flux_values(star_name)
obs_flux = np.array([m.value.nominal_value for m in flux_values])
obs_flux_unc = np.array([m.value.std_dev for m in flux_values])


exp_values = (exp_distance, exp_Teff, Teff_unc, log_g, log_unc, metallicity, met_unc, expected_Ebv, table_value)

_, sampler3 = complete_MCMC(exp_values, 12, obs_flux, obs_flux_unc, filter_wavelen)


# %%
