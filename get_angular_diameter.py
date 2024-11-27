#%%
from get_flux_values import * 
from SED_fitting import * 
import pandas as pd
import uncertainties.umath as umath
import time
from uncertainties import nominal_value
from astropy.constants import R_sun

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
def mean_flux_graph(wavelen, stellar_radius,table_value):
    
    mean_stellar_radius = np.mean(stellar_radius)

    stellar_radius_vals = []
    stellar_radius_unc = []
    for i in range(len(wavelen)):
        stellar_radius_vals.append(stellar_radius[i].value.nominal_value)
        stellar_radius_unc.append(stellar_radius[i].value.std_dev)
    
    fig, ax = plt.subplots(1, 1, sharex=True)
    plt.title('Radius values for each band \n and comparison with mean value and table value')
    values = ax.errorbar(wavelen, stellar_radius_vals, yerr=stellar_radius_unc, fmt='o',ecolor='orange')
    values.set_label('Values of radius computed in each band')
    value_table = ax.axhline(table_value.value)
    value_table.set_label('Table value of radius')
    percent_difference = ax.axhline(table_value.value + 0.05*table_value.value, color='r', linestyle='--')
    percent_difference.set_label('5 percent error of table value')
    ax.axhline(table_value.value - 0.05*table_value.value, color='r', linestyle='--')
    mean_radius = ax.axhline(mean_stellar_radius.value.nominal_value, color='g')
    mean_radius.set_label('Mean stellar radius value')
    ax.legend(framealpha=0.1)
    plt.grid()
    plt.xlabel('Wavelength of each band')
    plt.ylabel('Star Radius')
    plt.show()

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

    #print(flux_table)
    mean_stellar_radius = np.mean(stellar_radius)

    return mean_stellar_radius

# %% 
def star_set_tester(star_list, unit, show_plot):
    time_start = time.time()
    problem_stars = []

    for i in range(len(star_list)):
        star_name = star_list['Star'][i]
        Teff = star_list['Teff'][i]
        mettalicity = star_list['Fe/H'][i]
        log_g = star_list['logg'][i]
        Ebv = float(star_list['E(B-V)'][i])
        distance = float(star_list['Distance'][i]) * u.pc
        table_value = star_list['Radius'][i]

        R_Sun = 6.957e8 * u.m
        table_value = table_value * R_Sun
        table_value = table_value.to(R_Sun)

        print('Star being tested:', star_name)
        if show_plot == 'yes':
            try:
                wavelen, _, _, ang_diam = get_angular_diameter(star_name, Teff, mettalicity, log_g,Ebv)
                ang_diam = ang_diam.to(u.rad)
                stellar_radius = distance.to(R_Sun) * ang_diam.value / 2

                SED_plot(star_name, Teff, mettalicity, log_g, Ebv, unit)
                mean_flux_graph(wavelen, stellar_radius,table_value) 
                stellar_rad = create_dataframe(star_name, Teff, mettalicity, log_g, Ebv, distance, unit)
                
                print('Mean value of radius computed:', stellar_rad)
                print('Table value of radius:', table_value)
                print('--------')

            except Exception as e:
                problem_stars.append(star_name)
                print('Issue with star', star_name)
                print('--------')

        elif show_plot == 'no':
            try:
                stellar_rad = create_dataframe(star_name, Teff, mettalicity, log_g, Ebv, distance, unit)

                print('Mean value of radius computed:', stellar_rad)
                print('Table value of radius:', table_value)
                print('--------')

            except Exception as e:
                problem_stars.append(star_name)
                print('Issue with star', star_name)
                print('--------')

    time_end = time.time()
    print('Program took', time_end - time_start, 'seconds to run for', len(star_list),'stars')
    print(len(problem_stars), 'stars had issues with computing radius')

    return problem_stars


def single_star_tester(star_name, Teff, mettalicity, log_g, Ebv, distance, table_value, unit, show_plot):
    distance = distance * u.pc
    R_Sun = 6.957e8 * u.m
    table_value = table_value * R_Sun
    table_value = table_value.to(R_Sun)

    if show_plot == 'yes':
        wavelen, _, _, ang_diam = get_angular_diameter(star_name, Teff, mettalicity, log_g,Ebv)
        ang_diam = ang_diam.to(u.rad)
        stellar_radius = distance.to(R_Sun) * ang_diam.value / 2

        SED_plot(star_name, Teff, mettalicity, log_g, Ebv, unit)
        mean_flux_graph(wavelen, stellar_radius,table_value)
        stellar_rad = create_dataframe(star_name, Teff, mettalicity, log_g, Ebv, distance, unit)
        print('Mean value of radius computed:', stellar_rad)
        print('Table value of radius:', table_value)
        
    elif show_plot == 'no':
        stellar_rad = create_dataframe(star_name, Teff, mettalicity, log_g, Ebv, distance, unit)
        print('Mean value of radius computed:', stellar_rad)
        print('Table value of radius:', table_value)

# %% 

star_data = pd.read_csv('~/tese/testdata/list_stars.txt', sep="\t", header=0, skiprows=[1])
star_test_subset = star_data.head()
star_test = star_data.iloc[0:1]

problem_list = star_set_tester(star_test, 'SI', show_plot='yes')
# %% 
#print(problem_list)
## PROBLEM: subset has a different i than cycle 
# %% 

single_star_tester('WASP-84', 5221, 0.05, 4.28, 0.020, 100.4, 0.828, 'SI', 'no')

# %% 
star_name = 'WASP-84'
Teff = 5221
mettalicity = 0.05
log_g = 4.28
Ebv = 0.020

wavelen, obs_flux_values_Jy, model_flux_values_Jy, ang_diam = get_angular_diameter(star_name, Teff, mettalicity, log_g, Ebv)

obs_flux_values = flux_unit_change(obs_flux_values_Jy, 'SI')
model_flux_values = flux_unit_change(model_flux_values_Jy, 'SI')

#model_flux_values = model_flux_values * radius**2 / distance**2

obs_flux_vals = []
obs_flux_unc = []
for i in range(len(wavelen)):
    obs_flux_vals.append(obs_flux_values[i].value.nominal_value)
    obs_flux_unc.append(obs_flux_values[i].value.std_dev)

obs_flux_vals = np.array(obs_flux_vals) * model_flux_values.unit
obs_flux_unc = np.array(obs_flux_unc) * model_flux_values.unit

plt.plot(wavelen, model_flux_values,'o')
plt.errorbar(wavelen, obs_flux_vals, yerr = obs_flux_unc, fmt='o')
plt.show()

# %% 
print(obs_flux_vals)
print(obs_flux_unc)
print(model_flux_values)

def minimization_function(radius, distance, obs_flux_vals, obs_flux_unc, model_flux_values): 
    distance = distance * u.pc
    model_flux_values = model_flux_values * radius**2 / distance.to(R_sun) ** 2
    chi_squared = 0
    for i in range(len(model_flux_values)):
        chi_squared += ((model_flux_values[i].value - obs_flux_vals[i].value) / obs_flux_unc[i].value) ** 2
    return chi_squared

from scipy.optimize import minimize

result = minimize(minimization_function, x0=1.0,args=(100.4,obs_flux_vals, obs_flux_unc, model_flux_values))
result

# %% 

print(abs(0.8329 - 0.828) / 0.828 * 100)
print(abs(0.8156 - 0.828) / 0.828 * 100)