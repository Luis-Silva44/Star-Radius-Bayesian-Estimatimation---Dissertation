# %% Imports
from auxiliary_functions import * 
from transmission_test import * 

from scipy.interpolate import LinearNDInterpolator
import pysynphot as S
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from uncertainties import ufloat
from extinction import apply, ccm89

# %% Function that creates 8 SED using Kurucz and Castelli models for a cube of parameters
def create_SEDs(Teff_vals, mettalicity_vals, logg_vals):
    SED_data = []

    for teff in Teff_vals:
        for mett in mettalicity_vals:
            for logg in logg_vals:
                try:
                    sed_values = S.Icat('ck04models',teff,mett,logg)
                    SED_data.append(((teff,mett,logg),sed_values))
                except Exception as e:
                    print(f"Error getting SED values for Teff={teff}, log_g={logg} and mettalicity={mett}")
                    
    return SED_data

# %% Settign up the model grid 
mettalicity_grid = np.array([-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.2, 0.5])
logg_grid = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
a = list(range(3000, 13001, 250))
b = list(range(14000, 50001, 1000)) 
Teff_grid = a + b
Teff_grid = np.array(Teff_grid)

# %% Create the vertices of a 3D cube of parameters as limits of interpolation 
def SED_high_and_low(Teff,mettalicity,logg):
    Teff_low = max([t for t in Teff_grid if t <= Teff])
    Teff_high = min([t for t in Teff_grid if t >= Teff])
    mettalicity_low = max([m for m in mettalicity_grid if m <= mettalicity])
    mettalicity_high = min([m for m in mettalicity_grid if m >= mettalicity])
    logg_low = max([l for l in logg_grid if l <= logg])
    logg_high = min([l for l in logg_grid if l >= logg])

    Teff_values = [Teff_low, Teff_high]
    mettalicity_values =  [mettalicity_low, mettalicity_high]
    logg_values = [logg_low, logg_high]

    SED_data = create_SEDs(Teff_values,mettalicity_values,logg_values)
    return SED_data

# %% Interpolator of the SED flux with the parameter values that we want
def SED_interpolator(Teff,mettalicity,logg):
    SED_data = SED_high_and_low(Teff,mettalicity,logg)
    SED_wavelen = SED_data[0][1].wave * u.angstrom

    fluxes = []
    points = []

    for (parameters, SED_values) in SED_data: 
        fluxes.append(SED_values.flux)
        points.append(parameters)
    
    fluxes = np.array(fluxes)
    points = np.array(points)
    
    interpolated_fluxes = []

    for i in range(len(SED_wavelen)):
        flux_interpolator = LinearNDInterpolator(points, fluxes[:,i])
        interpolated_fluxes.append(flux_interpolator(Teff,mettalicity,logg))

    interpolated_fluxes = np.array(interpolated_fluxes) * u.erg / u.cm**2 / u.s / u.angstrom

    SED_wavelen = SED_wavelen.to(u.um)
    model_flux_Jy = interpolated_fluxes.to(u.Jy, u.spectral_density(SED_wavelen))   
    return SED_wavelen, model_flux_Jy

# %% Function that applies extinction to the SED 
def flux_extinction(wavelen, flux, Ebv):
    flux_ext = apply(ccm89(wavelen.to(u.angstrom), Ebv*3.1, 3.1), flux)
    return flux_ext

def SED_attenuated(Teff, mettalicity, logg, Ebv):
    wavelen, flux = SED_interpolator(Teff,mettalicity,logg)
    wavelen = wavelen.astype(np.float64)
    flux_attenuated = flux_extinction(wavelen, flux, Ebv)
    return wavelen, flux_attenuated

# %% Create a list of SED flux values with the size and wavelengths of the filter list
'''def SED_flux_bands(filter_wavelen, Teff, mettalicity, log_g, Ebv):
    SED_wavelen, SED_fluxes_Jy = SED_attenuated(Teff, mettalicity, log_g, Ebv)
    nearest_index = []
    for i in range(len(filter_wavelen)):
        nearest_index.append(find_nearest_index(SED_wavelen, filter_wavelen[i]))

    model_flux_values_Jy = np.array([SED_fluxes_Jy[i].value for i in nearest_index]) * u.Jy

    return filter_wavelen, model_flux_values_Jy'''
# %% 
def SED_bands(filter_wavelen, Teff, metallicity, log_g, Ebv):
    SED_wavelen, SED_fluxes_Jy = SED_attenuated(Teff, metallicity, log_g, Ebv)
    
    filter_bands = {'GBP':0.532, 'G': 0.673, 'GRP':0.797, 
                    'J':1.25, 'H':1.65, 'K':2.15,
                    'W1':3.4, 'W2':4.6, 'W3':12, 'W4':22} 
    
    selected_bands = [name for name, wav in filter_bands.items() if wav in filter_wavelen.value]
    SED_values = []
    for i in selected_bands:
        SED_values.append(convolution(SED_fluxes_Jy, SED_wavelen, i))

    return np.array(SED_values) * u.Jy
# %%  Testing the function
'''star_name = '55 Cnc'
Teff = 5353
metallicity = 0.3
log_g = 4.3
Ebv = 0.043
#SED_flux_bands(Teff, mettalicity, log_g, Ebv)^
filter_wavelen = band_wavelen(None)

SED_bands(filter_wavelen, Teff, metallicity, log_g, Ebv)'''
