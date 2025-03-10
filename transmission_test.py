# %% 
from SED_flux import * 
# %%
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import astropy.units as u
import scipy.integrate as integrate

# %%
def transmission_function(filter):

    filter_path = {'W1': 'WISE_WISE.W1.dat',
                   'W2': 'WISE_WISE.W2.dat', 
                   'W3': 'WISE_WISE.W3.dat',
                   'W4': 'WISE_WISE.W3.dat', 
                   'G': 'GAIA_GAIA3.G.dat',
                   'GBP': 'GAIA_GAIA3.Gbp.dat',
                   'GRP': 'GAIA_GAIA3.Grp.dat',
                   'J': '2MASS_2MASS.J.dat',
                   'H': '2MASS_2MASS.H.dat',
                   'K': '2MASS_2MASS.Ks.dat'}

    transmission_file = filter_path.get(filter)

    file_path = '/home/luis/tese/transmission_functions/' +  transmission_file
    transmission_function = pd.read_csv(file_path, delimiter=' ', names=['Wavelength', 'Transmission'])

    transmission_wavelength = np.array(transmission_function['Wavelength'])
    transmission = np.array(transmission_function['Transmission'])

    return (transmission_wavelength * u.angstrom).to(u.um), transmission
# %%

def convolution(model_flux, SED_wavelen, filter):
    transmission_wave, transmission = transmission_function(filter)
    model_flux = np.interp(transmission_wave, SED_wavelen, model_flux)

    convolution_value = integrate.simpson(y = model_flux * transmission, x = transmission_wave)
    normalization = integrate.simpson(y = transmission, x = transmission_wave)  

    return convolution_value / normalization

# %%
