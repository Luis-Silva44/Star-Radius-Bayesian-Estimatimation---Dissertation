#%% 
import astropy.units as u
from astroquery.vizier import Vizier
import numpy as np
from uncertainties import ufloat
import matplotlib.pyplot as plt
from astroquery.simbad import Simbad

 

#STILL NEEDED: REMOVE WARNING FROM COORDINATES
# %% 

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
def vizier_coords(gaia_id):
    Vizier.ROW_LIMIT = 1  
    gaia_catalog = "I/355/gaiadr3"  # Gaia DR3 catalog

    gaia_values = Vizier.query_constraints(catalog=gaia_catalog, Source=str(gaia_id))
    
    if gaia_values:
        ra, dec = gaia_values[0]['RA_ICRS'][0], gaia_values[0]['DE_ICRS'][0]
        return ra, dec
    else:
        raise ValueError("No Gaia id found.")

#%% 
def wise_values(gaia_id):
    wise_catalog = 'II/311/wise'
    ra, dec = vizier_coords(str(gaia_id))
    wise_values = Vizier.query_region(f"{ra} {dec}", radius=10* u.arcsec , catalog=wise_catalog)
    if wise_values:
        return wise_values
    else:
        raise ValueError('No WISE data found')

def two_mass_values(gaia_id):
    two_mass_catalog = 'II/246/out'
    ra, dec = vizier_coords(str(gaia_id))

    two_mass_values = Vizier.query_region(f"{ra} {dec}", radius=10*u.arcsec, catalog=two_mass_catalog)
    
    if two_mass_values:
        return two_mass_values
    else: 
        raise ValueError('No 2MASS data found')

def gaia_values(gaia_id):
    gaia_catalog = "I/355/gaiadr3"  # Gaia DR3 catalog
    gaia_values = Vizier.query_constraints(catalog=gaia_catalog, Source=str(gaia_id))
    return gaia_values

def mag_to_flux(mag,band): # in W cm-2 micrometer-1
    flux_zero_points =  {'J':3.1293e-13,
                         'H':1.133e-13,
                         'K':4.283e-14,
                         'W1':8.180e-15,
                         'W2':2.415e-15
                         }
    flux_constant = flux_zero_points.get(band)

    if flux_constant is None:
        raise ValueError(f'No zero point flux value found for {band} band')
    return flux_constant * 10**(-mag * 0.4)

#%%
def get_flux_values(star_name):
    
    gaia_id = retrieve_gaia_id(star_name)

    filter_bands = {'GBP':0.532, 'G': 0.673, 'GRP':0.797, 
                    'J':1.25, 'H':1.65, 'K':2.15,
                    'W1':3.4, 'W2':4.6}

    filter_wavelen = np.array([d for d in filter_bands.values()]) * u.um

    wise_data = wise_values(gaia_id)
    two_mass_data = two_mass_values(gaia_id)
    gaia_data = gaia_values(gaia_id)
    
    W1_mag = ufloat(wise_data[0]['W1mag'], wise_data[0]['e_W1mag'])
    W2_mag = ufloat(wise_data[0]['W2mag'], wise_data[0]['e_W2mag'])
    J_mag = ufloat(two_mass_data[0]['Jmag'], two_mass_data[0]['e_Jmag'])
    H_mag = ufloat(two_mass_data[0]['Hmag'], two_mass_data[0]['e_Hmag'])
    K_mag = ufloat(two_mass_data[0]['Kmag'], two_mass_data[0]['e_Kmag'])

    W1_flux = mag_to_flux(W1_mag,'W1')
    W2_flux= mag_to_flux(W2_mag,'W2')
    J_flux = mag_to_flux(J_mag,'J')
    H_flux = mag_to_flux(H_mag,'H')
    K_flux = mag_to_flux(K_mag,'K')
  
    G_flux = ufloat(gaia_data[0]['FG'], gaia_data[0]['e_FG'])
    GBP_flux = ufloat(gaia_data[0]['FBP'], gaia_data[0]['e_FBP'])
    GRP_flux = ufloat(gaia_data[0]['FRP'], gaia_data[0]['e_FRP'])
    
    unit = 1 * u.watt / u.m**2 / u.nm
    G_flux = G_flux * 1.346109e-21 * unit
    GBP_flux = GBP_flux * 3.009167E-21 * unit
    GRP_flux = GRP_flux * 1.638483E-21 * unit

    G_flux = G_flux.to(u.watt / u.um / u.cm**2)
    GBP_flux = GBP_flux.to(u.watt / u.um / u.cm**2)
    GRP_flux = GRP_flux.to(u.watt / u.um / u.cm**2)

    unit = 1 * u.watt / u.um / u.cm**2
    flux_values = np.array([GBP_flux.value, G_flux.value, GRP_flux.value, J_flux,H_flux,K_flux,W1_flux,W2_flux]) * unit
    flux_values_Jy = flux_values.to(u.Jy, equivalencies=u.spectral_density(filter_wavelen))

    return filter_wavelen, flux_values_Jy

# %% 
#wavelen, flux = get_flux_values(1019003226022657920)
#flux