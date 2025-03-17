# %% 
import astropy.units as u
import numpy as np
from astroquery.vizier import Vizier
from uncertainties import ufloat
from uncertainties import unumpy
import matplotlib.pyplot as plt
from astropy.constants import R_sun

from auxiliary_functions import *
from SED_flux import SED_bands, flux_extinction
import scipy.stats as stats
import emcee as emcee
import multiprocessing
import corner
from IPython.display import display, Math
# %%

def flux_to_mag(flux, band):
    flux_zero_points =  {'GBP':4.11e-12,
                         'G':2.5e-12,
                         'GRP':1.24e-12,
                         'J':3.1293e-13,
                         'H':1.133e-13,
                         'K':4.283e-14,
                         'W1':8.180e-15,
                         'W2':2.415e-15,
                         'W3':6.515e-17,
                         'W4':5.090e-18
                         }
    
    flux_constant = flux_zero_points.get(band)

    if flux_constant is None:
        raise ValueError(f'No zero point flux value found for {band} band')
    return -2.5 * unumpy.log10(flux / flux_constant)
# %%   

def gaia_values(star_name):
    gaia_id, star_name = retrieve_gaia_id(star_name)
    gaia_catalog = "I/355/gaiadr3"  # Gaia DR3 catalog
    gaia_data = Vizier.query_constraints(columns = ['**'], catalog=gaia_catalog, Source=str(gaia_id))

    G_flux = ufloat(gaia_data[0]['FG'], gaia_data[0]['e_FG']) * 1.346109e-20
    GBP_flux = ufloat(gaia_data[0]['FBP'], gaia_data[0]['e_FBP']) * 3.009167E-20
    GRP_flux = ufloat(gaia_data[0]['FRP'], gaia_data[0]['e_FRP']) * 1.638483E-20
    gaia_parallax = ufloat(gaia_data[0]['Plx'], gaia_data[0]['e_Plx'])
    
    G_mag = flux_to_mag(G_flux, 'G')
    GBP_mag = flux_to_mag(GBP_flux, 'GBP')
    GRP_mag = flux_to_mag(GRP_flux, 'GRP')
    return gaia_parallax, [GBP_mag, G_mag, GRP_mag]

def wise_values(star_name):
    Vizier.ROW_LIMIT = -1
    gaia_id, star_name = retrieve_gaia_id(star_name)
    wise_catalog = 'II/311/wise'
    ra, dec = vizier_coords(star_name)
    coords = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')
    wise_data = Vizier.query_region(coords, radius=10* u.arcsec , catalog=wise_catalog)

    if wise_data:
        W1_mag = ufloat(wise_data[0][0]['W1mag'], wise_data[0][0]['e_W1mag'])
        W2_mag = ufloat(wise_data[0][0]['W2mag'], wise_data[0][0]['e_W2mag'])
        W3_mag = ufloat(wise_data[0][0]['W3mag'], wise_data[0][0]['e_W3mag'])
        
    return [W1_mag, W2_mag, W3_mag]

def two_mass_values(star_name):
    Vizier.ROW_LIMIT = -1
    gaia_id, star_name = retrieve_gaia_id(star_name)
    gaia_catalog = "I/355/gaiadr3"
    gaia_data = Vizier.query_constraints(catalog=gaia_catalog, Source=str(gaia_id))
    two_mass_iden = str(gaia_data[0]['_2MASS'][0])
    res = Vizier.query_object(star_name, catalog='II/246/out')

    flag = 0
    for i in res[0]:
        if i['_2MASS'] == two_mass_iden: 
            break 
        flag += 1

    two_mass_data = res[0][flag]
    if two_mass_data:
        J_mag = ufloat(two_mass_data['Jmag'], two_mass_data['e_Jmag'])
        H_mag = ufloat(two_mass_data['Hmag'], two_mass_data['e_Hmag'])
        K_mag = ufloat(two_mass_data['Kmag'], two_mass_data['e_Kmag'])

    return [J_mag, H_mag, K_mag]


# %%
def mag_list(star_name):
    parallax, gaia_mag = gaia_values(star_name)
    two_mass_mag = two_mass_values(star_name)
    wise_mag = wise_values(star_name)
    mag_list = gaia_mag + two_mass_mag + wise_mag
    mag_values = np.array([m.nominal_value for m in mag_list])
    mag_unc = np.array([m.std_dev for m in mag_list])
    
    return mag_values, mag_unc, parallax

# %%
def SED_mags(filter_wavelen, Teff, log_g, metallicity, Ebv):
    SED_flux = SED_bands(filter_wavelen, Teff, metallicity, log_g, Ebv)
    SED_attenuated = flux_extinction(filter_wavelen, SED_flux, Ebv)
    SED_flux = SED_attenuated.to(u.watt * u.cm ** -2 * u.um ** -1, equivalencies=u.spectral_density(filter_wavelen))
    filter_check = ['GBP', 'G', 'GRP', 'J', 'H', 'K', 'W1', 'W2', 'W3']
    SED_mags = []
    flag = 0
    for band in filter_check:
        SED_mags.append(flux_to_mag(SED_flux[flag].value, band))
        flag += 1 
    return np.array(SED_mags)

# %%
filter_wavelen = band_wavelen(['GBP', 'G', 'GRP', 'J', 'H', 'K', 'W1', 'W2', 'W3'])
star_name = 'WASP-84'

Teff = 5221
teff_unc = 72
log_g = 4.28
log_unc = 0.13
metallicity = 0.05
metallicity_unc = 0.05
Ebv = 0.020

table_value = (0.828 * R_sun).to(R_sun)
# %%
mag_values, mag_unc, parallax = mag_list(star_name)
unit_change = 1 * u.parsec
exp_distance = (1 / parallax) * unit_change

exp_values = (exp_distance, Teff, teff_unc, log_g, log_unc, metallicity, metallicity_unc, table_value)
# %% 
def likelihood(params, mag_values, mag_unc, filter_wavelen, Ebv):
        distance, temperature, log_g, metallicity, radius = params
        distance2 = (distance * u.pc).to(R_sun)

        model_mags = SED_mags(filter_wavelen, temperature, log_g, metallicity, Ebv)
        
        model_mags_scaled = model_mags + 5 * np.log10(radius / distance2.value)
    
        return np.sum(stats.norm.logpdf(mag_values, loc = model_mags_scaled, scale = mag_unc))

def prior(params, exp_distance, exp_temperature, temp_unc, exp_log, log_unc, exp_met, met_unc):
        distance, temperature, log_g, metallicity, radius = params

        if not (0.1 < radius < 10.0) or not(3500 < temperature < 10000) or not (-2.5 < metallicity < 0.5) or not (0 < log_g < 5.0):
            return -np.inf
        
        distance_prior = stats.norm.logpdf(distance, loc = exp_distance.value.nominal_value, scale = exp_distance.value.std_dev)
        temperature_prior = stats.norm.logpdf(temperature, loc = exp_temperature, scale = temp_unc)
        log_g_prior = stats.norm.logpdf(log_g, loc = exp_log, scale = log_unc)
        met_prior = stats.norm.logpdf(metallicity, loc = exp_met, scale = met_unc)
        
        return (distance_prior + temperature_prior + log_g_prior + met_prior) 

def posterior(params, mag_values, mag_unc, exp_distance, exp_temperature, temp_unc, exp_log, log_unc, exp_met, met_unc, filter_wavelen, Ebv):
        log_prior = prior(params, exp_distance, exp_temperature, temp_unc, exp_log, log_unc, exp_met, met_unc)
        if not np.isfinite(log_prior):
            return -np.inf
    
        return log_prior + likelihood(params, mag_values, mag_unc, filter_wavelen, Ebv)

def test_MCMC(exp_values, nwalkers, mag_values, mag_unc, filter_wavelen, Ebv):
    exp_distance, exp_temperature, temp_unc, exp_log, log_unc, exp_met, met_unc, exp_radius = exp_values 

    pos = np.array([exp_distance.value.nominal_value + exp_distance.value.std_dev * np.random.randn(nwalkers), 
                    exp_temperature + temp_unc * np.random.randn(nwalkers),
                    exp_log + log_unc * np.random.randn(nwalkers),
                    exp_met + met_unc * np.random.randn(nwalkers),
                    np.random.normal(1.0, 0.5, size=nwalkers)]).T   

    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(nwalkers, ndim, posterior, args=(mag_values, mag_unc, exp_distance, exp_temperature, temp_unc, exp_log, log_unc, exp_met, met_unc, filter_wavelen, Ebv), pool = multiprocessing.Pool(12))
    state = sampler.run_mcmc(pos, 200, progress=True)

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

    flat_samples = sampler.get_chain(flat=True)
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
test_MCMC(exp_values, 12, mag_values, mag_unc, filter_wavelen, Ebv)
# %%
teff_values = np.linspace(4001, 6001, 20)

prior_list = []
likelihood_list = []
posterior_list = []

for i in teff_values:
    params = (exp_distance.value.nominal_value, i, log_g, metallicity, table_value.value)
    prior_value = prior(params, exp_distance, Teff, teff_unc, log_g, log_unc, metallicity, metallicity_unc)
    likelihood_value = likelihood(params, mag_values, mag_unc, filter_wavelen, Ebv)
    post_value = posterior(params, mag_values, mag_unc, exp_distance, Teff, teff_unc, log_g, log_unc, metallicity, metallicity_unc, filter_wavelen, Ebv)
    prior_list.append(prior_value)
    likelihood_list.append(likelihood_value)
    posterior_list.append(post_value)
# %%
plt.plot(teff_values, (prior_list))
plt.title('Prior probability distribution')
plt.xlabel('Temperature')
plt.ylabel('Probability')
plt.show()
plt.plot(teff_values, (likelihood_list))
plt.title('Likelihood probability distribution')
plt.xlabel('Temperature')
plt.ylabel('Probability')
plt.show()
plt.plot(teff_values, (posterior_list))
plt.title('Posterior probability distribution')
plt.xlabel('Temperature')
plt.ylabel('Probability')
plt.show()
# %%
