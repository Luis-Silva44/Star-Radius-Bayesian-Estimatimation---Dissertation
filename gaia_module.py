# %% Imports
import astropy.units as u
import numpy as np
from astroquery.vizier import Vizier
from astroquery.simbad import Simbad
from uncertainties import ufloat
from SED_flux import band_wavelen

# Function that allows easy unit changes in the programm. Easy to adapt or add new unit systems
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
# Function that allows the programm to either look for star name or gaia id. Only requirement 
# is that gaia_id is an int, and the name a string
def retrieve_gaia_id(star_name):
    if type(star_name) == int:
        gaia_id = star_name
    elif type(star_name) == str:
        result_table = Simbad.query_objectids(star_name)
        for x in result_table:
            if 'Gaia DR3' in x['ID']:
                gaia_id = str(x['ID']).replace('Gaia DR3 ', '')
    return gaia_id

# %% 
# Function to search the gaia data release 3 and give us the values of flux, flux errors and parallax
def gaia_values(star_name):
    gaia_id = retrieve_gaia_id(star_name)
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

        unit = 1 * u.arcsec
        return gaia_flux.to(u.watt / u.um / u.cm**2), gaia_parallax * unit

    else:
        raise ValueError('No Gaia ID found')


 #%%  Simple test