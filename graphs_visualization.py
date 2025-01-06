# %%
from SED_fitting import * 
from auxiliary_functions import * 

import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.constants import R_sun
from uncertainties import umath
import pandas as pd
# %%

def SED_fitting_plot(SED_wavelen, SED_flux, photometry_flux_values, photometry_flux_unc, minimization_radius, distance):
    wavelen = band_wavelen()
    minimization_flux = SED_flux * minimization_radius **2 / distance.to(R_sun) ** 2
    minimization_flux = np.array([flux.value.nominal_value for flux in minimization_flux])

    plt.title('Fitting the modeled flux to the values of observed flux (extinction fixed)')
    plt.xlabel('Wavelength (μm)')
    plt.ylabel(f'Flux ({SED_flux.unit})')
    plt.plot(SED_wavelen, minimization_flux)
    plt.errorbar(wavelen, photometry_flux_values, yerr = photometry_flux_unc, fmt='o')
    plt.xlim(0,23)
    plt.grid()
    plt.legend(['Model flux','Observed flux'])
    plt.show()

# %% 
def create_dataframe(photometry_flux, SED_flux, distance):
    wavelen = band_wavelen()
    
    angular_diameter = []
    for i in range(len(photometry_flux)):
        ang_diam = 2 * umath.sqrt(photometry_flux[i].value / SED_flux[i].value)
        angular_diameter.append(ang_diam)

    unit = 1 * u.rad
    angular_diameter = angular_diameter * unit

    stellar_radius = distance.to(R_sun) * angular_diameter.value / 2 #THIS IS SINE OF A VERY SMALL ANGLE
    
    flux_table = pd.DataFrame({
    'Filter Wavelength': wavelen,
    'Observed flux': photometry_flux,
    'Surface flux (model)': SED_flux,
    'Angular Diameter': angular_diameter,
    'Stellar radius': stellar_radius})

    column_units = {'Filter Wavelength': wavelen.unit,
                    'Observed flux': photometry_flux.unit,
                    'Surface flux (model)': SED_flux.unit,
                    'Angular Diameter': angular_diameter.unit,
                    'Stellar radius':stellar_radius.unit}
    
    flux_table.rename(columns={col: f"{col} ({unit})" for col, unit in column_units.items()}, inplace=True)
    return flux_table

# Test
#star_name = 'WASP-84'	
#Teff = 5221 
#logg = 4.28
#mettalicity = 0.05
#table_value = (0.828 * R_sun).to(R_sun)
#Ebv = 0.020
#distance = 100.4 * u.pc

#photometry_flux = get_flux_values(star_name)
#wavelen, SED_flux = SED_flux_bands(Teff, mettalicity, logg, Ebv)
#create_dataframe(photometry_flux, SED_flux, distance)
# %%


