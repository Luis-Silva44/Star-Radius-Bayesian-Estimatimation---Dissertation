# %% 

from gaia_module import retrieve_gaia_id
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier
from uncertainties import ufloat

 # %% 

# Function used to turn from the apparent magnitudes to the fluxes from the zero point fluxes in vega system. This values are taken directly from the respective 2MASS and WISE articles
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

# Function to get the sky coordinates of the star from the gaia iD/star name
def vizier_coords(star_name):
    gaia_id = retrieve_gaia_id(star_name)
    gaia_catalog = "I/355/gaiadr3"  # Gaia DR3 catalog
    gaia_values = Vizier.query_constraints(catalog=gaia_catalog, Source=str(gaia_id))
    
    if gaia_values:
        ra, dec = gaia_values[0]['RA_ICRS'][0], gaia_values[0]['DE_ICRS'][0]
        return ra, dec
    else:
        raise ValueError("No Gaia id found.")

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