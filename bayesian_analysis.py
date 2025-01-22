# %% 
from SED_fitting import * 
from gaia_module import * 
from auxiliary_functions import * 
# %%
def get_flux(star_name, Teff, mettalicity, log_g, Ebv, unit):
    _, parallax, _= gaia_values(star_name)
    unit_change = 1 * u.parsec
    distance = (1 / parallax.value) * unit_change
    filter_wavelen, photometry_flux_Jy = get_flux_values(star_name)
    wavelen, SED_flux_Jy = SED_flux_bands(filter_wavelen, Teff, mettalicity, log_g, Ebv)
    
    photometry_flux = flux_unit_change(photometry_flux_Jy, unit)
    SED_flux = flux_unit_change(SED_flux_Jy, unit)

    photometry_flux_vals = []
    photometry_flux_unc = []
    for i in range(len(wavelen)):
        photometry_flux_vals.append(photometry_flux[i].value.nominal_value)
        photometry_flux_unc.append(photometry_flux[i].value.std_dev)

    photometry_flux_vals = np.array(photometry_flux_vals)
    photometry_flux_unc = np.array(photometry_flux_unc)
    photometry_flux_unc = np.where(photometry_flux_unc == 0, 1e-10, photometry_flux_unc)

    return photometry_flux_vals, photometry_flux_unc, SED_flux.value, distance.value.nominal_value
# %%

star_name = 'DE Boo'	
Teff = 5289
logg = 4.27
mettalicity = 0.03
table_value = (0.863 * R_sun).to(R_sun)
Ebv = 0.04

#photometry_flux_vals, photometry_flux_unc, SED_flux,distance = get_flux(star_name, Teff, mettalicity, log_g, Ebv, 'Jy')
# %%

import numpy as np
import emcee
import matplotlib.pyplot as plt

# Define the log-likelihood function
def log_likelihood(radius, photometry_flux_vals, photometry_flux_unc, SED_flux, distance):
    if radius <= 0:
        return -np.inf  
    
    scaled_SED_flux = SED_flux * (radius**2 / distance**2)
    chi_squared = np.sum(((photometry_flux_vals - scaled_SED_flux) / photometry_flux_unc) ** 2)
    log_likelihood_value = -0.5 * chi_squared
    return log_likelihood_value

# Define the log-prior function
def log_prior(radius):
 
    if 0.1 < radius < 10.0:  
        return 0.0  # log(1) = 0
    else:
        return -np.inf 

# Define the log-posterior function
def log_posterior(radius, photometry_flux_vals, photometry_flux_unc, SED_flux, distance):
    log_prior_value = log_prior(radius)
    if log_prior_value == -np.inf:
        return -np.inf  
    log_likelihood_value = log_likelihood(radius, photometry_flux_vals, photometry_flux_unc, SED_flux, distance)
    return log_prior_value + log_likelihood_value


def run_mcmc(photometry_flux_vals, photometry_flux_unc, SED_flux, distance, n_walkers=32, n_steps=5000):

    initial_radius = 1.0  
    initial_positions = initial_radius + 0.1 * np.random.randn(n_walkers, 1) 
    sampler = emcee.EnsembleSampler(
        n_walkers, 
        1,  # Single parameter
        log_posterior, 
        args=(photometry_flux_vals, photometry_flux_unc, SED_flux, distance)
    )
    sampler.run_mcmc(initial_positions, n_steps, progress=True)
    samples = sampler.get_chain(flat=True)
    return samples


photometry_flux_vals, photometry_flux_unc, SED_flux, distance = get_flux(star_name, Teff, mettalicity, log_g, Ebv, 'Jy')

samples = run_mcmc(photometry_flux_vals, photometry_flux_unc, SED_flux, distance)
radius_median = np.median(samples)
print('Stellar Radius:', radius_median, 'R_sun')

plt.hist(samples, bins=50, density=True, alpha=0.7, color="blue")
plt.xlabel("Stellar Radius (R_sun)")
plt.ylabel("Probability Density")
plt.title("Posterior Distribution of Stellar Radius")
plt.legend()
plt.grid()
plt.show()
