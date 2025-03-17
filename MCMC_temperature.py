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
import scipy.stats as stats
# %%
def likelihood(params, obs_flux, obs_flux_unc, filter_wavelen, Ebv):
        distance, temperature, log_g, metallicity, radius = params
        distance2 = (distance * u.pc).to(R_sun)

        SED_wavelen, model_flux = SED_interpolator(temperature, metallicity, log_g)
        
        filter_bands = {'GBP':0.532, 'G': 0.673, 'GRP':0.797, 
                    'J':1.25, 'H':1.65, 'K':2.15,
                    'W1':3.4, 'W2':4.6, 'W3':12, 'W4':22} 
        
        selected_bands = [name for name, wav in filter_bands.items() if wav in filter_wavelen.value]
        SED_values = []
        for i in selected_bands:
            SED_values.append(convolution(model_flux, SED_wavelen, i))

        model_flux = np.array(SED_values)
        filter_wavelen = filter_wavelen.astype(np.float64)
        flux_attenuated = flux_extinction(filter_wavelen, model_flux, Ebv)

        model_flux_scaled = flux_attenuated * (radius / distance2.value)**2

        #c = np.log(2 * np.pi * obs_flux_unc**2)
        
        #return (-0.5 * np.sum(c + ((obs_flux - model_flux_scaled)**2 / obs_flux_unc**2)))
    
        return np.sum(stats.norm.logpdf(obs_flux, loc = model_flux_scaled, scale = 2 * obs_flux_unc))

def prior(params, exp_distance, exp_temperature, temp_unc, exp_log, log_unc, exp_met, met_unc):
        distance, temperature, log_g, metallicity, radius = params

        if not (0.1 < radius < 10.0) or not(3500 < temperature < 10000) or not (-2.5 < metallicity < 0.5) or not (0 < log_g < 5.0):
            return -np.inf
        
        distance_prior = stats.norm.logpdf(distance, loc = exp_distance.value.nominal_value, scale = exp_distance.value.std_dev)
        temperature_prior = stats.norm.logpdf(temperature, loc = exp_temperature, scale = temp_unc)
        log_g_prior = stats.norm.logpdf(log_g, loc = exp_log, scale = log_unc)
        met_prior = stats.norm.logpdf(metallicity, loc = exp_met, scale = met_unc)
        
        return (distance_prior + temperature_prior + log_g_prior + met_prior) 

def posterior(params, obs_flux, obs_flux_unc, exp_distance, exp_temperature, temp_unc, exp_log, log_unc, exp_met, met_unc, filter_wavelen, Ebv):
        log_prior = prior(params, exp_distance, exp_temperature, temp_unc, exp_log, log_unc, exp_met, met_unc)
        if not np.isfinite(log_prior):
            return -np.inf
    
        return log_prior + likelihood(params, obs_flux, obs_flux_unc, filter_wavelen, Ebv)

# %%
def basic_MCMC(exp_values, nwalkers, obs_flux, obs_flux_unc, filter_wavelen, Ebv):
    exp_distance, exp_temperature, temp_unc, exp_log, log_unc, exp_met, met_unc, exp_radius = exp_values 

    pos = np.array([exp_distance.value.nominal_value + exp_distance.value.std_dev * np.random.randn(nwalkers), 
                    exp_temperature + temp_unc * np.random.randn(nwalkers),
                    exp_log + log_unc * np.random.randn(nwalkers),
                    exp_met + met_unc * np.random.randn(nwalkers),
                    np.random.normal(1.0, 0.5, size=nwalkers)]).T

    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(nwalkers, ndim, posterior, args=(obs_flux, obs_flux_unc, exp_distance, exp_temperature, temp_unc, exp_log, log_unc, exp_met, met_unc, filter_wavelen, Ebv), pool = multiprocessing.Pool(12))
    state = sampler.run_mcmc(pos, 800, progress=True)

    labels = ["Distance", "Temperature", "Log g", "Metallicity", "Radius"]
    fig, axes = plt.subplots(5, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()

    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")

    flat_samples = sampler.get_chain(discard = 300, flat=True)
    fig = corner.corner(flat_samples, labels=labels)
    plt.show()

    expected_values = [exp_distance.value.nominal_value, exp_temperature, exp_log, exp_met, exp_radius.value]
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], labels[i])
        display(Math(txt))
        print('Error in ', labels[i], '=', abs(mcmc[1] - expected_values[i]) / expected_values[i] * 100)
        if i == 4:
            return (mcmc[1] * R_sun).to(R_sun), sampler
# %%
star_name = 'WASP-84'
Ebv = 0.020
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

exp_values = (exp_distance, exp_Teff, Teff_unc, log_g, log_unc, metallicity, met_unc, table_value)

_, sampler1 = basic_MCMC(exp_values, 12, obs_flux, obs_flux_unc, filter_wavelen, Ebv)

# %%
star_name = 'HD128582'
Ebv = 0.008
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

exp_values = (exp_distance, exp_Teff, Teff_unc, log_g, log_unc, metallicity, met_unc, table_value)

_, sampler2 = basic_MCMC(exp_values, 12, obs_flux, obs_flux_unc, filter_wavelen, Ebv)

# %%
