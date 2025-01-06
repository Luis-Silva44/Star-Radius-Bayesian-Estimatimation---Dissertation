# %% 
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier
from uncertainties import ufloat
import numpy as np

from auxiliary_functions import *
 # %% 

# Function to get the photometry values and errors and turn them into fluxes
def two_mass_values(star_name):
    gaia_id = retrieve_gaia_id(star_name)
    two_mass_catalog = 'II/246/out'
    ra, dec = vizier_coords(star_name)
    coords = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')

    two_mass_data = Vizier.query_region(coords, radius=10*u.arcsec, catalog=two_mass_catalog)
    
    if two_mass_data:
        J_mag = ufloat(two_mass_data[0]['Jmag'], two_mass_data[0]['e_Jmag'])
        H_mag = ufloat(two_mass_data[0]['Hmag'], two_mass_data[0]['e_Hmag'])
        K_mag = ufloat(two_mass_data[0]['Kmag'], two_mass_data[0]['e_Kmag'])

        J_flux = mag_to_flux(J_mag,'J')
        H_flux = mag_to_flux(H_mag,'H')
        K_flux = mag_to_flux(K_mag,'K')

        unit = 1 * u.watt / u.um / u.cm**2
        two_mass_flux = np.array([J_flux, H_flux, K_flux]) * unit

        return two_mass_flux
    else: 
        raise ValueError('No 2MASS data found')
    
# %%
two_mass_values('55 Cnc')