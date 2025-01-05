# %% 
from gaia_module import gaia_values
from wise_module import * 
from two_mass_module import * 
from SED_flux import *

import uncertainties as u
import numpy as np

# %% Function that allows easy unit change

def flux_unit_change(value,unit):
    if unit == 'Jy':
        return value.to(u.Jy)
    
    elif unit == 'cgs':
        cgs_flux_units =  u.erg / u.cm**2 / u.s / u.Hz
        return value.to(cgs_flux_units)
    
    elif unit == 'SI':
        SI_flux_units = u.watt / u.m**2 / u.Hz
        return value.to(SI_flux_units)
    
    else:
        raise ValueError('Unit not recognized by programm')
    
# %% 

def get_flux_values(star_name):
    gaia_flux, _ = gaia_values(star_name)
    two_mass_flux = two_mass_values(star_name)
    wise_flux = wise_values(star_name)

    flux_values = np.concatenate(gaia_flux, two_mass_flux, wise_flux)
    return flux_values
# %% 

#retrieve_gaia_id(704967037090946688)

two_mass_values(704967037090946688)