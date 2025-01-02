# %% 
from gaia_module import retrieve_gaia_id
from two_mass_module import mag_to_flux, vizier_coords
import numpy as np
import astropy.units as u
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
from uncertainties import ufloat

# %% 
# Function to get the photometry values and errors and turn them into fluxes
def wise_values(star_name):
    wise_catalog = 'II/311/wise'
    ra, dec = vizier_coords(star_name)
    coords = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')

    wise_data = Vizier.query_region(coords, radius=10* u.arcsec , catalog=wise_catalog)

    if wise_values:
        W1_mag = ufloat(wise_data[0]['W1mag'], wise_data[0]['e_W1mag'])
        W2_mag = ufloat(wise_data[0]['W2mag'], wise_data[0]['e_W2mag'])
        W3_mag = ufloat(wise_data[0]['W3mag'], wise_data[0]['e_W3mag'])
        W4_mag = ufloat(wise_data[0]['W4mag'], wise_data[0]['e_W4mag'])

        W1_flux = mag_to_flux(W1_mag,'W1')
        W2_flux= mag_to_flux(W2_mag,'W2')
        W3_flux= mag_to_flux(W3_mag,'W3')
        W4_flux= mag_to_flux(W4_mag,'W4')

        unit = 1 * u.watt / u.um / u.cm**2
        wise_flux = np.array([W1_flux, W2_flux, W3_flux, W4_flux]) * unit

        return wise_flux
    else:
        raise ValueError('No WISE data found')