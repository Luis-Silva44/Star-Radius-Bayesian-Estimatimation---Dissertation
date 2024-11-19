# %%
from get_flux_values import *
from scipy.interpolate import LinearNDInterpolator
import pysynphot as S
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from uncertainties import ufloat
from flux_extinction import flux_extinction

# %% 
def create_SEDs(Teff_vals, mettalicity_vals, logg_vals):
    SED_data = []

    for teff in Teff_vals:
        for mett in mettalicity_vals:
            for logg in logg_vals:
                try:
                    sed_values = S.Icat('ck04models',teff,mett,logg)
                    SED_data.append(((teff,mett,logg),sed_values))
                except Exception as e:
                    print(f"Error getting SED values for Teff={teff}, log_g={logg} and mettalicity={mett}")
                    
    return SED_data
# %% 
mettalicity_grid = np.array([-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.2, 0.5])
logg_grid = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
a = list(range(3000, 13001, 250))
b = list(range(14000, 50001, 1000)) 
Teff_grid = a + b
Teff_grid = np.array(Teff_grid)

def SED_high_and_low(Teff,mettalicity,logg):
    Teff_low = max([t for t in Teff_grid if t <= Teff])
    Teff_high = min([t for t in Teff_grid if t > Teff])
    mettalicity_low = max([m for m in mettalicity_grid if m <= mettalicity])
    mettalicity_high = min([m for m in mettalicity_grid if m > mettalicity])
    logg_low = max([l for l in logg_grid if l <= logg])
    logg_high = min([l for l in logg_grid if l > logg])

    Teff_values = [Teff_low, Teff_high]
    mettalicity_values =  [mettalicity_low, mettalicity_high]
    logg_values = [logg_low, logg_high]

    SED_data = create_SEDs(Teff_values,mettalicity_values,logg_values)
    return SED_data

def SED_interpolator(Teff,mettalicity,logg):
    SED_data = SED_high_and_low(Teff,mettalicity,logg)
    SED_wavelen = SED_data[0][1].wave * u.angstrom

    fluxes = []
    points = []

    for (parameters, SED_values) in SED_data: 
        fluxes.append(SED_values.flux)
        points.append(parameters)
    
    fluxes = np.array(fluxes)
    points = np.array(points)
    
    interpolated_fluxes = []

    for i in range(len(SED_wavelen)):
        flux_interpolator = LinearNDInterpolator(points, fluxes[:,i])
        interpolated_fluxes.append(flux_interpolator(Teff,mettalicity,logg))

    interpolated_fluxes = np.array(interpolated_fluxes) * u.erg / u.cm**2 / u.s / u.angstrom

    SED_wavelen = SED_wavelen.to(u.um)
    model_flux_Jy = interpolated_fluxes.to(u.Jy, u.spectral_density(SED_wavelen))

    return SED_wavelen, model_flux_Jy


def SED_attenuated(Teff,mettalicity,logg, Ebv):
    wavelen, flux = SED_interpolator(Teff,mettalicity,logg)
    wavelen = wavelen.astype(np.float64)
    flux_attenuated = flux_extinction(wavelen, flux, Ebv)
    return wavelen, flux_attenuated



# %% 
def SED_plot(gaia_id, Teff, mettalicity, log_g, Ebv, unit):
    SED_wavelen, model_flux_Jy= SED_interpolator(Teff,mettalicity,log_g)
    _, flux_attenuated = SED_attenuated(Teff,mettalicity,log_g, Ebv)
    model_flux = flux_unit_change(model_flux_Jy, unit)
    flux_attenuated = flux_unit_change(flux_attenuated, unit)

    plt.plot(SED_wavelen, model_flux)
    plt.plot(SED_wavelen, flux_attenuated)
    plt.xlim(0,5)
    plt.xlabel('Wavelength (μm)')
    plt.ylabel('Flux (unit)')
    plt.grid()
    plt.title('SED of stellar model')
    plt.show()

    band_wavelen, flux_values_Jy = get_flux_values(gaia_id)
    flux_values = flux_unit_change(flux_values_Jy, unit)
    flux_val = []
    flux_unc = []
    for i in range(len(band_wavelen)):
        flux_val.append(flux_values[i].value.nominal_value)
        flux_unc.append(flux_values[i].value.std_dev)

    plt.errorbar(band_wavelen, flux_val, yerr=flux_unc, fmt='o',ecolor='orange')
    plt.xlim(0,5)
    plt.xlabel('Wavelength (μm)')
    plt.ylabel('Flux (unit)')
    plt.grid()
    plt.title('SED observed')
    plt.show()

# %% 
gaia_id = 1019003226022657920
Teff = 5581 
mettalicity = 0.33
log_g = 4.33
Ebv = 0.3

SED_plot(gaia_id, Teff, mettalicity, log_g, Ebv,'Jy')
