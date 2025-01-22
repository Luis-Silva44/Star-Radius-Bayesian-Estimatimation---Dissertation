# %% Imports
import astropy.units as u
import numpy as np
from astroquery.vizier import Vizier
from uncertainties import ufloat

from auxiliary_functions import *

# %% 
# Function to search the gaia data release 3 and give us the values of flux, flux errors and parallax
def gaia_values(star_name):
    gaia_id, star_name = retrieve_gaia_id(star_name)
    gaia_catalog = "I/355/gaiadr3"  # Gaia DR3 catalog
    gaia_data = Vizier.query_constraints(catalog=gaia_catalog, Source=str(gaia_id))

    if gaia_data:
    # Get the flux values and errors in each of the gaia bands, and the parallax
        G_flux = ufloat(gaia_data[0]['FG'], gaia_data[0]['e_FG'])
        GBP_flux = ufloat(gaia_data[0]['FBP'], gaia_data[0]['e_FBP'])
        GRP_flux = ufloat(gaia_data[0]['FRP'], gaia_data[0]['e_FRP'])
        gaia_parallax = ufloat(gaia_data[0]['Plx'], gaia_data[0]['e_Plx'])

        # transform the fluxes into normal flux density units according to the gaia article 
        unit = 1 * u.watt / u.m**2 / u.nm
        G_flux = G_flux * 1.346109e-21 
        GBP_flux = GBP_flux * 3.009167E-21
        GRP_flux = GRP_flux * 1.638483E-21 
        gaia_flux = np.array([GBP_flux, G_flux, GRP_flux]) * unit

        gaia_check = ['GBP', 'G', 'GRP']

        unit = 1 * u.mas
        return gaia_flux.to(u.watt / u.um / u.cm**2), (gaia_parallax * unit).to(u.arcsec), gaia_check

    else:
        raise ValueError('No Gaia ID found')


 #%%  Simple test
