#%%
from get_flux_values import * 
from SED_fitting import * 
import pandas as pd
import uncertainties.umath as umath

# %% 
def find_nearest_index(array, value):
        index = (np.abs(array - value)).argmin()
        return index

def get_angular_diameter(gaia_id, Teff, mettalicity, log_g, Ebv):
    wavelen, obs_flux_values_Jy = get_flux_values(gaia_id)
    SED_wavelen, SED_fluxes_Jy = SED_attenuated(Teff, mettalicity, log_g, Ebv)

    nearest_index = []
    for i in range(len(wavelen)):
        nearest_index.append(find_nearest_index(SED_wavelen, wavelen[i]))

    model_flux_values_Jy = np.array([SED_fluxes_Jy[i].value for i in nearest_index]) * u.Jy

    angular_diameter = []
    for i in range(len(obs_flux_values_Jy)):
        ang_diam = 2 * umath.sqrt(obs_flux_values_Jy[i].value / model_flux_values_Jy[i].value)
        angular_diameter.append(ang_diam)

    unit = 1 * u.rad
    angular_diameter = angular_diameter * unit
    angular_diameter_arcsec = angular_diameter.to(u.arcsec)
    return wavelen, obs_flux_values_Jy, model_flux_values_Jy, angular_diameter_arcsec

# %% 
def create_dataframe(gaia_id, Teff, mettalicity, log_g, Ebv, distance, unit):
    wavelen, obs_flux_values_Jy, model_flux_values_Jy, ang_diam = get_angular_diameter(gaia_id, Teff, mettalicity, log_g,Ebv)
    R_Sun = 6.957e8 * u.m
    distance = distance.to(R_Sun)
    ang_diam = ang_diam.to(u.rad)
    obs_flux_values = flux_unit_change(obs_flux_values_Jy, unit)
    model_flux_values = flux_unit_change(model_flux_values_Jy, unit)

    stellar_radius = distance.to(R_Sun) * ang_diam.value / 2 #THIS IS SINE OF A VERY SMALL ANGLE

    flux_table = pd.DataFrame({
    'Filter Wavelength': wavelen,
    'Observed flux': obs_flux_values,
    'Surface flux (model)': model_flux_values,
    'Angular Diameter': ang_diam,
    'Stellar radius':stellar_radius})

    column_units = {'Filter Wavelength': wavelen.unit,
                    'Observed flux': obs_flux_values.unit,
                    'Surface flux (model)': model_flux_values.unit,
                    'Angular Diameter': ang_diam.unit,
                    'Stellar radius':stellar_radius.unit}
    
    flux_table.rename(columns={col: f"{col} ({unit})" for col, unit in column_units.items()}, inplace=True)

    print(flux_table)
    print('---')
    mean_stellar_radius = np.mean(stellar_radius)

    return mean_stellar_radius
# %% 
R_Sun = 6.957e8 * u.m
gaia_id = 3788394461991295488
Teff = 6265
mettalicity = 0.15
log_g = 4.38
Ebv = 0.039
distance = 359.4 * u.pc
SWEET_cat_value = 1.468 * R_Sun
SWEET_cat_value = SWEET_cat_value.to(R_Sun)

SED_plot(gaia_id, Teff, mettalicity, log_g, Ebv, 'SI')
stellar_rad = create_dataframe(gaia_id, Teff, mettalicity, log_g, Ebv, distance, 'Jy')

print('Mean value:', stellar_rad)
print('SWEET-cat value:', SWEET_cat_value)
print('Erro relativamente ao tabelado:', abs(SWEET_cat_value - stellar_rad) / SWEET_cat_value * 100,'%')

# %% 

# NEXT TO ADD
## Search by star name OR gaia id 
## Read file of test stars