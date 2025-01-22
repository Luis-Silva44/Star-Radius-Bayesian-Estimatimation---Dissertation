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
        J_flux = mag_to_flux(J_mag,'J')
        H_flux = mag_to_flux(H_mag,'H')
        K_flux = mag_to_flux(K_mag,'K')

        unit = 1 * u.watt / u.um / u.cm**2
        two_mass_flux = np.array([J_flux, H_flux, K_flux]) * unit
        two_mass_check = ['J', 'H', 'K']

    else: 
        unit = 1 * u.watt / u.um / u.cm**2
        two_mass_flux = [] * unit
        two_mass_check = []
        print('No 2MASS data found')
    
    return two_mass_flux, two_mass_check
# %% 

#two_mass_values('TOI-5696')