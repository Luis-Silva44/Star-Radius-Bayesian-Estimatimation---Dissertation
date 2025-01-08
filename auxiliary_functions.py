# %% Imports 

import numpy as np
import astropy.units as u
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
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
    
# %% 
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
def band_wavelen():
    filter_bands = {'GBP':0.532, 'G': 0.673, 'GRP':0.797, 
                    'J':1.25, 'H':1.65, 'K':2.15,
                    'W1':3.4, 'W2':4.6, 'W3':12, 'W4':22}

    wavelen = np.array([d for d in filter_bands.values()]) * u.um
    return wavelen

# %%
def mag_to_flux(mag,band): # in W cm-2 micrometer-1
    flux_zero_points =  {'J':3.1293e-13,
                         'H':1.133e-13,
                         'K':4.283e-14,
                         'W1':8.180e-15,
                         'W2':2.415e-15,
                         'W3':6.515e-17,
                         'W4':5.090e-18
                         }
    flux_constant = flux_zero_points.get(band)

    if flux_constant is None:
        raise ValueError(f'No zero point flux value found for {band} band')
    return flux_constant * 10**(-mag * 0.4)

# %% 
def vizier_coords(star_name):
    gaia_id = retrieve_gaia_id(star_name)
    gaia_catalog = "I/355/gaiadr3"  # Gaia DR3 catalog
    gaia_values = Vizier.query_constraints(catalog=gaia_catalog, Source=str(gaia_id))
    
    if gaia_values:
        ra, dec = gaia_values[0]['RA_ICRS'][0], gaia_values[0]['DE_ICRS'][0]
        return ra, dec
    else:
        raise ValueError("No Gaia id found.")
# %%
def find_nearest_index(array, value):
        index = (np.abs(array - value)).argmin()
        return index

# %% 
def star_set_mean_tester(star_list, unit):
    time_start = time.time()

    problem_stars = []
    computed_radius = []
    table_value_radius = []

    for i in range(len(star_list)):
        star_name = star_list['Star'][i]
        Teff = star_list['Teff'][i]
        mettalicity = star_list['Fe/H'][i]
        log_g = star_list['logg'][i]
        Ebv = float(star_list['E(B-V)'][i])
        table_value = star_list['Radius'][i]

        table_value = table_value * R_sun
        table_value = table_value.to(R_sun)

        print('Star being tested:', star_name)

        try:
            radius, _ = SED_fitting(star_name, Teff, mettalicity, log_g, Ebv, unit)
            computed_radius.append(radius)
            table_value_radius.append(table_value)
                
            print('Value of radius computed:', radius)
            print('Table value of radius:', table_value)
            print('Error in value computed:', abs(table_value - radius) / table_value * 100)
            print('--------')

        except Exception as e:
            problem_stars.append(star_name)
            print('Issue with star', star_name)
            print('--------')

    time_end = time.time()
    print('Program took', time_end - time_start, 'seconds to run for', len(star_list),'stars')
    print(len(problem_stars), 'stars had issues with computing radius')

    return problem_stars, computed_radius, table_value_radius