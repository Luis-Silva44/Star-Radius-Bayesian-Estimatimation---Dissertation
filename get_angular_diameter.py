#%%
from get_flux_values import * 
from SED_fitting import * 
import pandas as pd
import uncertainties.umath as umath
import time
from uncertainties import nominal_value

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
    plt.xlabel('Wavelength of each band')
    plt.ylabel('Star Radius')
    plt.show()

# %% 
def create_dataframe(star_name, Teff, mettalicity, log_g, Ebv, table_value, distance, unit):
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
                stellar_rad = create_dataframe(star_name, Teff, mettalicity, log_g, Ebv, table_value, distance, unit)
                mean_flux_graph(wavelen, stellar_radius,table_value) 
                print('Mean value of radius computed:', stellar_rad)
                print('Table value of radius:', table_value)
                print('--------')

            except Exception as e:
                problem_stars.append(star_name)
                print('Issue with star', star_name)
                print('--------')

        elif show_plot == 'no':
            try:
                stellar_rad = create_dataframe(star_name, Teff, mettalicity, log_g, Ebv, table_value, distance, unit)

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
        stellar_rad = create_dataframe(star_name, Teff, mettalicity, log_g, Ebv, table_value, distance, unit)
        print('Mean value of radius computed:', stellar_rad)
        print('Table value of radius:', table_value)
        
    elif show_plot == 'no':
        stellar_rad = create_dataframe(star_name, Teff, mettalicity, log_g, Ebv, table_value, distance, unit)
        print('Mean value of radius computed:', stellar_rad)
        print('Table value of radius:', table_value)

# %% 

star_data = pd.read_csv('~/git_project/testdata/list_stars.txt', sep="\t", header=0, skiprows=[1])
star_test_subset = star_data.head()
star_test = star_data.iloc[0:1]

#problem_list = star_set_tester(star_test, 'SI', show_plot='yes')
# %% 
#print(problem_list)
## PROBLEM: subset has a different i than cycle 
# %% 

single_star_tester('HIP 41378', 6371, 0.046, 4.07, 0.021, 106.0, 1.299, 'SI', 'yes')

# %% 

