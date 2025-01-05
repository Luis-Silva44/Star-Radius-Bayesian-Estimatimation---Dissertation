# %%
from SED_fitting import * 
from SED_flux import band_wavelen
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.constants import R_sun

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