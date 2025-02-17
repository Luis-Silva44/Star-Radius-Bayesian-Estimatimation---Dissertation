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
from scipy.stats import norm

# %%

def likelihood(params, obs_flux, obs_flux_unc, model_flux, filter_wavelen):
        distance, Ebv, radius = params
        distance2 = (distance * u.pc).to(R_sun)
        filter_wavelen_ = filter_wavelen.astype(np.float64)
        flux_attenuated = flux_extinction(filter_wavelen_, model_flux, Ebv)
        model_flux_scaled = flux_attenuated * (radius / distance2.value)**2

        c = np.log(2 * np.pi * obs_flux_unc**2)
        return -0.5 * np.sum(c + ((obs_flux - model_flux_scaled)**2 / obs_flux_unc**2))

def prior(params, exp_distance):
        distance, Ebv, radius = params

        if not (0.1 < radius < 10.0) or not (0.0 < Ebv < 0.5):
            return -np.inf
        
        distance_prior = -0.5 * ((distance - exp_distance.value.nominal_value) / exp_distance.value.std_dev)**2
        return distance_prior

def posterior(params, obs_flux, obs_flux_unc, model_flux, filter_wavelen, exp_distance):
        log_prior = prior(params, exp_distance)
        if not np.isfinite(log_prior):
            return -np.inf
    
        return log_prior + likelihood(params, obs_flux, obs_flux_unc, model_flux, filter_wavelen)
# %% 
def basic_MCMC(exp_values, nwalkers, obs_flux, obs_flux_unc, model_flux, filter_wavelen):
    exp_distance, expected_Ebv, expected_radius = exp_values 

    pos = np.array([exp_distance.value.nominal_value + exp_distance.value.std_dev * np.random.randn(nwalkers), 
                np.abs(np.random.normal(0.5, 0.2, size=nwalkers)),
                np.random.normal(1.0, 0.5, size=nwalkers)]).T

    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(nwalkers, ndim, posterior, args=(obs_flux, obs_flux_unc, model_flux, filter_wavelen, exp_distance), pool = multiprocessing.Pool(16))
    state = sampler.run_mcmc(pos, 2500, progress=True)

    labels = ["Distance", "E(B-V)", "Radius"]
    fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain(discard=1250)

    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")

    flat_samples = sampler.get_chain(discard=2000, flat=True)
    fig = corner.corner(flat_samples, labels=labels)
    plt.show()

    expected_values = [exp_distance.value.nominal_value, expected_Ebv, expected_radius.value]
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], labels[i])
        display(Math(txt))
        print('Error in ', labels[i], '=', abs(mcmc[1] - expected_values[i]) / expected_values[i] * 100)
        if i == 2:
            return (mcmc[1] * R_sun).to(R_sun)

# %%
def star_set_tester_MCMC(star_list):
    time_start = time.time()

    problem_stars = []
    computed_radius = []
    table_value_radius = []

    for i in range(len(star_list)):
        star_name = star_list['Star'][i]
        Teff = star_list['Teff'][i]
        metallicity = star_list['Fe/H'][i]
        log_g = star_list['logg'][i]
        expected_Ebv = star_list['E(B-V)'][i]
        table_value = star_list['Radius'][i]

        table_value = table_value * R_sun
        table_value = table_value.to(R_sun)

        print('Star being tested:', star_name)

        try:
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
            exp_values = (exp_distance, expected_Ebv, table_value)

            comp_radius = basic_MCMC(exp_values, 10, obs_flux, obs_flux_unc, model_flux, filter_wavelen)
            computed_radius.append(comp_radius)
            table_value_radius.append(table_value)
        except Exception as e:
            problem_stars.append(star_name)
            print('Issue with star', star_name)
            print('--------')

    time_end = time.time()
    print('Program took', time_end - time_start, 'seconds to run for', len(star_list),'stars')
    print(len(problem_stars), 'stars had issues with computing radius')

    return problem_stars, computed_radius, table_value_radius
# %% 

star_data = pd.read_csv('~/tese/testdata/list_stars.txt', sep="\t", header=0, skiprows=[1])
star_test_subset = star_data.head()

# %%
problem_stars, computed_list, table_list = star_set_tester_MCMC(star_data)
# %%
computed_radii = np.array([i.value for i in computed_list])
table_values = np.array([i.value for i in table_list])
computed_real_comparison(computed_radii, table_values)

#  %% 

star_name = 'TOI-2350'
expected_Ebv = 0.005
table_value = (1.052 * R_sun).to(R_sun)

Teff = 5481
log_g = 0.32
metallicity = 4.45

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
exp_values = (exp_distance, expected_Ebv, table_value)

basic_MCMC(exp_values, 10, obs_flux, obs_flux_unc, model_flux, filter_wavelen)

# %%
