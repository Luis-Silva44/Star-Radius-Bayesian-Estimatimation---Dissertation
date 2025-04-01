# %% Imports
from auxiliary_functions import *
from gaia_module import *
from wise_module import * 
from two_mass_module import * 
from SED_flux import *
from graphs_visualization import * 

import astropy.units as u
import time
from astropy.constants import R_sun
import numpy as np
from scipy.optimize import minimize

# %% 
def get_flux_values(star_name):
    gaia_flux, _, gaia_check = gaia_values(star_name)
    two_mass_flux, two_mass_check= two_mass_values(star_name)
    wise_flux, wise_check = wise_values(star_name)

    flux_values = np.concatenate((gaia_flux, two_mass_flux, wise_flux))

    filter_check = gaia_check + two_mass_check + wise_check
    filter_wavelen = band_wavelen(filter_check)
    flux_values_Jy = flux_values.to(u.Jy, equivalencies=u.spectral_density(filter_wavelen))

    return filter_wavelen, flux_values_Jy

# %%

def SED_fitting(star_name, Teff, mettalicity, log_g, Ebv, unit):
    _, parallax, _ = gaia_values(star_name)
    unit_change = 1 * u.parsec
    distance = (1 / parallax.value) * unit_change
    filter_wavelen, photometry_flux_Jy = get_flux_values(star_name)
    #SED_flux_Jy = SED_bands(filter_wavelen, Teff, mettalicity, log_g, Ebv)
    _, SED_flux_Jy = SED_flux_bands(filter_wavelen, Teff, mettalicity, log_g, Ebv)
    photometry_flux = flux_unit_change(photometry_flux_Jy, unit)
    SED_flux = flux_unit_change(SED_flux_Jy, unit)

    photometry_flux_vals = []
    photometry_flux_unc = []
    for i in range(len(filter_wavelen)):
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
    SED_flux = SED_flux * minimization_radius**2 / (distance.value.nominal_value * u.parsec).to(R_sun) ** 2

    SED_wavelen, SED_att = SED_attenuated(Teff,mettalicity,log_g,Ebv)
    SED_att = flux_unit_change(SED_att, unit)
    
    SED_fitting_plot(SED_wavelen, SED_att, SED_flux, filter_wavelen, photometry_flux_vals, photometry_flux_unc, minimization_radius.value, distance)
    
    return minimization_radius, minimization_result

# %% 
def single_star_tester(star_name, Teff, mettalicity, logg, Ebv, table_value, unit):
    radius, _ = SED_fitting(star_name, Teff, mettalicity, logg, Ebv, unit)
    error = abs(radius - table_value) / table_value * 100
    print('Computed radius is:', radius)
    print('Table value of radius:', table_value)
    print('Error:', error, '%')
    return radius

# %% 
def star_set_tester(star_list, unit):
    time_start = time.time()

    problem_stars = []
    computed_radius = []
    table_value_radius = []

    for i in range(len(star_list)):
        star_name = star_list['Star'][i]
        Teff = star_list['Teff'][i]
        mettalicity = star_list['Fe/H'][i]
        log_g = star_list['logg'][i]
        Ebv = float(star_list['E(B-V)'][i])
        table_value = star_list['Radius'][i]

        table_value = table_value * R_sun
        table_value = table_value.to(R_sun)

        print('Star being tested:', star_name)

        try:
            radius, _ = SED_fitting(star_name, Teff, mettalicity, log_g, Ebv, unit)
            computed_radius.append(radius)
            table_value_radius.append(table_value)
                
            print('Value of radius computed:', radius)
            print('Table value of radius:', table_value)
            print('Error in value computed:', abs(table_value - radius) / table_value * 100)
            print('--------')

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

# Uncomment to run for list of stars (approx. 45 seconds)
#problem_list, computed_list, table_list = star_set_tester(star_data, 'SI')

# %% 
#computed_radii = np.array([i.value for i in computed_list])
#table_values = np.array([i.value for i in table_list])
#computed_real_comparison(computed_radii, table_values)

# %%
#print(problem_list)

'''['GJ176', # temperature too low, outside of model range
''' 

'''
Star that simbad value of J doesn't match: 
55 Cnc
HD 136352
HD 219134
HD128582
'''
# %% 
star_name =  'WASP-84'
Teff =  5221
logg = 4.28
mettalicity = 0.05
table_value = (0.828 * R_sun).to(R_sun)
Ebv = 0.020
single_star_tester(star_name, Teff, mettalicity, logg, Ebv, table_value, 'SI')

# %%
star_name = 'HD128582'
Teff = 6168
logg = 4.17
mettalicity = 0.098
table_value = (1.63 * R_sun).to(R_sun)
Ebv = 0.008
single_star_tester(star_name, Teff, mettalicity, logg, Ebv, table_value, 'SI')
# %%
