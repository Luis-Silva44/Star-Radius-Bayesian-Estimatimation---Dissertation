# %% Imports
from auxiliary_functions import *
from gaia_module import gaia_values
from wise_module import * 
from two_mass_module import * 
from SED_flux import *
from graphs_visualization import * 

import astropy.units as u
from astropy.constants import R_sun
import numpy as np
from scipy.optimize import minimize

# %% 
def get_flux_values(star_name):
    gaia_flux, _ = gaia_values(star_name)
    two_mass_flux = two_mass_values(star_name)
    wise_flux = wise_values(star_name)

    flux_values = np.concatenate((gaia_flux, two_mass_flux, wise_flux))
    filter_wavelen = band_wavelen()

    flux_values_Jy = flux_values.to(u.Jy, equivalencies=u.spectral_density(filter_wavelen))
    return flux_values_Jy

# %%

def SED_fitting(star_name, Teff, mettalicity, log_g, Ebv, unit):
    _, parallax = gaia_values(star_name)
    unit_change = 1 * u.parsec
    distance = (1 / parallax.value) * unit_change
    photometry_flux_Jy = get_flux_values(star_name)
    wavelen, SED_flux_Jy = SED_flux_bands(Teff, mettalicity, log_g, Ebv)
    
    photometry_flux = flux_unit_change(photometry_flux_Jy, unit)
    SED_flux = flux_unit_change(SED_flux_Jy, unit)

    photometry_flux_vals = []
    photometry_flux_unc = []
    for i in range(len(wavelen)):
        photometry_flux_vals.append(photometry_flux[i].value.nominal_value)
        photometry_flux_unc.append(photometry_flux[i].value.std_dev)

    photometry_flux_vals = np.array(photometry_flux_vals)
    photometry_flux_unc = np.array(photometry_flux_unc) 


    def minimization_function(radius, distance, SED_flux, photometry_flux_vals, photometry_flux_unc):
        SED_flux = SED_flux * radius**2 / (distance.value.nominal_value * u.parsec).to(R_sun) ** 2
        chi_squared = np.sum(((SED_flux.value - photometry_flux_vals)/ photometry_flux_unc) ** 2)
        return chi_squared
    
    minimization_result = minimize(minimization_function, x0=1.0, args=(distance,SED_flux,photometry_flux_vals, photometry_flux_unc), method='Nelder-Mead')
    minimization_radius= minimization_result.x[0]

    minimization_radius = (minimization_radius * R_sun).to(R_sun)

    SED_wavelen, SED_att = SED_attenuated(Teff,mettalicity,log_g,Ebv)
    SED_att = flux_unit_change(SED_att, unit)
    SED_fitting_plot(SED_wavelen, SED_att, photometry_flux_vals, photometry_flux_unc, minimization_radius.value, distance)

    return minimization_radius, minimization_result

# %% Test cell

star_name = 'WASP-84'	
Teff = 5221 
logg = 4.28
mettalicity = 0.05
table_value = (0.828 * R_sun).to(R_sun)
Ebv = 0.020
#distance = 100.4

radius, _ = SED_fitting(star_name, Teff, mettalicity, logg, Ebv, 'Jy')
error = abs(radius - table_value) / table_value * 100
print('Computed radius is:', radius)
print('Table value of radius:', table_value)
print('Error:', error, '%')
# %%
