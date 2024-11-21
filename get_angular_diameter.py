#%%
from get_flux_values import * 
from SED_fitting import * 
import pandas as pd
import uncertainties.umath as umath

# %% 
def find_nearest_index(array, value):
        index = (np.abs(array - value)).argmin()
        return index

def get_angular_diameter(star_name, Teff, mettalicity, log_g, Ebv):
    wavelen, obs_flux_values_Jy = get_flux_values(star_name)
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
def create_dataframe(star_name, Teff, mettalicity, log_g, Ebv, distance, unit):
    wavelen, obs_flux_values_Jy, model_flux_values_Jy, ang_diam = get_angular_diameter(star_name, Teff, mettalicity, log_g,Ebv)
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
star_name = 'GJ176'
Teff = 3355
mettalicity = -0.01
log_g = 4.79
Ebv = 0.015
distance = 9.5 * u.pc
table_value = 0.394 * R_Sun
table_value = table_value.to(R_Sun)

#SED_plot(star_name, Teff, mettalicity, log_g, Ebv, 'SI')
#stellar_rad = create_dataframe(star_name, Teff, mettalicity, log_g, Ebv, distance, 'Jy')

#rint('Mean value:', stellar_rad)
#print('SWEET-cat value:', table_value)
#print('Erro relativamente ao tabelado:', abs(table_value - stellar_rad) / table_value * 100,'%')


# %% 
def parameter_unloader(star_list, unit):
    for i in range(len(star_list)):
        star_name = star_list['Star'][i]
        Teff = star_list['Teff'][i]
        mettalicity = star_list['Fe/H'][i]
        log_g = star_list['logg'][i]
        Ebv = float(star_list['E(B-V)'][i])
        distance = float(star_list['Distance'][i]) * u.pc
        table_value = star_list['Radius'][i]

        print('Star being tested:', star_name)
        SED_plot(star_name, Teff, mettalicity, log_g, Ebv, unit)
        stellar_rad = create_dataframe(star_name, Teff, mettalicity, log_g, Ebv, distance, unit)

        print('Mean value of radius computed:', stellar_rad)
        print('Table value of radius:', table_value)
        print('--------')

# %% 

star_data = pd.read_csv('~/git_project/testdata/list_stars.txt', sep="\t", header=0, skiprows=[1])
#star_test_subset = star_data.iloc[]

## PROBLEM: subset has a different i than cycle 
#parameter_unloader(star_test_subset, 'Jy')

