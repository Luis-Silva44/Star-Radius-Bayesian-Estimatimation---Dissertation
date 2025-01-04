# %% Imports

from get_flux_values import *
from scipy.interpolate import LinearNDInterpolator
import pysynphot as S
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from uncertainties import ufloat
from flux_extinction import flux_extinction

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
    Teff_high = min([t for t in Teff_grid if t > Teff])
    mettalicity_low = max([m for m in mettalicity_grid if m <= mettalicity])
    mettalicity_high = min([m for m in mettalicity_grid if m > mettalicity])
    logg_low = max([l for l in logg_grid if l <= logg])
    logg_high = min([l for l in logg_grid if l > logg])

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
def SED_attenuated(Teff, mettalicity, logg, Ebv):
    wavelen, flux = SED_interpolator(Teff,mettalicity,logg)
    wavelen = wavelen.astype(np.float64)
    flux_attenuated = flux_extinction(wavelen, flux, Ebv)
    return wavelen, flux_attenuated

# %% Function to get the wavelength of the peak of transmission of each band
def band_wavelen():
    filter_bands = {'GBP':0.532, 'G': 0.673, 'GRP':0.797, 
                    'J':1.25, 'H':1.65, 'K':2.15,
                    'W1':3.4, 'W2':4.6, 'W3':12, 'W4':22}

    wavelen = np.array([d for d in filter_bands.values()]) * u.um
    return wavelen 
# %% Simple function to get the closest index to a certain value. Used to get the closest wavelengths in the models
def find_nearest_index(array, value):
        index = (np.abs(array - value)).argmin()
        return index

# %% Create a list of SED flux values with the size and wavelengths of the filter list
def SED_flux_bands(Teff, mettalicity, log_g, Ebv, unit):
    SED_wavelen, SED_fluxes_Jy = SED_attenuated(Teff, mettalicity, log_g, Ebv)
    wavelen = band_wavelen()
    nearest_index = []
    for i in range(len(wavelen)):
        nearest_index.append(find_nearest_index(SED_wavelen, wavelen[i]))

    model_flux_values_Jy = np.array([SED_fluxes_Jy[i].value for i in nearest_index]) * u.Jy

    return wavelen, model_flux_values_Jy

# %%  Testing the function
#"star_name = 1019003226022657920
#Teff = 5581 
#mettalicity = 0.33
#log_g = 4.33
#Ebv = 0.3 
#SED_flux_bands(Teff, mettalicity, log_g, Ebv)
