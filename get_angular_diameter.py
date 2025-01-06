#%%
'''from get_flux_values import * 
from SED_fitting import * 
import pandas as pd
import uncertainties.umath as umath
import time
from uncertainties import nominal_value
from astropy.constants import R_sun
from scipy.optimize import minimize


# %% 
def find_nearest_index(array, value):
        index = (np.abs(array - value)).argmin()
        return index

def get_angular_diameter(star_name, Teff, mettalicity, log_g, Ebv):
    wavelen, obs_flux_values_Jy = get_flux_values(star_name)
    #SED_wavelen, SED_fluxes_Jy = SED_attenuated(Teff, mettalicity, log_g, Ebv)
    SED_wavelen, SED_fluxes_Jy = SED_interpolator(Teff, mettalicity,log_g)

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

# %% 
def create_dataframe(star_name, Teff, mettalicity, log_g, Ebv, distance, unit):
    wavelen, obs_flux_values_Jy, model_flux_values_Jy, ang_diam = get_angular_diameter(star_name, Teff, mettalicity, log_g,Ebv)
    #distance = distance.to(R_sun)
    ang_diam = ang_diam.to(u.rad)
    obs_flux_values = flux_unit_change(obs_flux_values_Jy, unit)
    model_flux_values = flux_unit_change(model_flux_values_Jy, unit)

    stellar_radius = distance.to(R_sun) * ang_diam.value / 2 #THIS IS SINE OF A VERY SMALL ANGLE

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
    

# %% 
def star_set_mean_tester(star_list, unit, show_plot):
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

        table_value = table_value * R_sun
        table_value = table_value.to(R_sun)

        print('Star being tested:', star_name)

        wavelen, _, _, ang_diam = get_angular_diameter(star_name, Teff, mettalicity, log_g,Ebv)
        ang_diam = ang_diam.to(u.rad)
        stellar_radius = distance.to(R_sun) * ang_diam.value / 2
        mean_stellar_radius = np.mean(stellar_radius)

        if show_plot == 'yes':
            try:
                mean_flux_graph(wavelen, stellar_radius, table_value) 
                
                print('Mean value of radius computed:', mean_stellar_radius)
                print('Table value of radius:', table_value)
                print('--------')

            except Exception as e:
                problem_stars.append(star_name)
                print('Issue with star', star_name)
                print('--------')

        elif show_plot == 'no':
            try:
                print('Mean value of radius computed:', mean_stellar_radius)
                print('Table value of radius:', table_value)
                print('--------')

            except Exception as e:
                problem_stars.append(star_name)
                print('Issue with star', star_name)
                print('--------')

    time_end = time.time()
    print('Program took', time_end - time_start, 'seconds to run for', len(star_list),'stars')
    print(len(problem_stars), 'stars had issues with computing radius')

    return problem_stars, mean_stellar_radius


def single_star_mean_tester(star_name, Teff, mettalicity, log_g, Ebv, distance, table_value, unit, show_plot):
    distance = distance * u.pc
    table_value = table_value * R_sun
    table_value = table_value.to(R_sun)

    wavelen, _, _, ang_diam = get_angular_diameter(star_name, Teff, mettalicity, log_g,Ebv)
    ang_diam = ang_diam.to(u.rad)
    stellar_radius = distance.to(R_sun) * ang_diam.value / 2
    mean_stellar_radius = np.mean(stellar_radius)
    if show_plot == 'yes':
        mean_flux_graph(wavelen, stellar_radius,table_value)
        print('Value of radius computed from mean:', mean_stellar_radius)
        print('Table value of radius:', table_value)
        print('Error in mean radius value:', abs(mean_stellar_radius - table_value) / table_value * 100, '%')
        
    elif show_plot == 'no':
        print('Value of radius computed from mean:', mean_stellar_radius)
        print('Table value of radius:', table_value)
        print('Error in mean radius value:', abs(mean_stellar_radius - table_value) / table_value * 100, '%')

# %% 

star_data = pd.read_csv('~/git_project/testdata/list_stars.txt', sep="\t", header=0, skiprows=[1])
star_test_subset = star_data.head()
star_test = star_data.iloc[0:1]

problem_list, mean_stellar_radius = star_set_mean_tester(star_test, 'SI', show_plot='yes')
print(problem_list)
print(mean_stellar_radius)
# %% 
single_star_mean_tester('WASP-84', 5221, 0.05, 4.28, 0.020, 100.4, 0.828, 'SI', 'no')
# %% 
def SED_fitting(star_name, Teff, mettalicity, log_g, Ebv, distance, table_value, unit, show_plot):
    distance = distance * u.parsec
    wavelen, obs_flux_values_Jy, model_flux_values_Jy, _ = get_angular_diameter(star_name, Teff, mettalicity, log_g, Ebv)
    model_flux_values_Jy = flux_extinction(wavelen, model_flux_values_Jy, Ebv)
    obs_flux_values = flux_unit_change(obs_flux_values_Jy, unit)
    model_flux_values = flux_unit_change(model_flux_values_Jy, unit)

    obs_flux_vals = []
    obs_flux_unc = []
    for i in range(len(wavelen)):
        obs_flux_vals.append(obs_flux_values[i].value.nominal_value)
        obs_flux_unc.append(obs_flux_values[i].value.std_dev)

    obs_flux_values = np.array(obs_flux_vals) * model_flux_values.unit
    obs_flux_unc = np.array(obs_flux_unc) * model_flux_values.unit

    def minimization_function(radius, distance, model_flux_values, obs_flux_values, obs_flux_unc):
        model_flux_values = model_flux_values * radius**2 / distance.to(R_sun) ** 2
        chi_squared = np.sum(((model_flux_values.value - obs_flux_values.value)/ obs_flux_unc.value) ** 2)
        return chi_squared
    
    minimization_result = minimize(minimization_function, x0=1.0, args=(distance,model_flux_values,obs_flux_values, obs_flux_unc), method='Nelder-Mead')
    minimization_radius= minimization_result.x[0]

    if show_plot == 'yes':
        minimization_flux = model_flux_values * minimization_radius **2 / distance.to(R_sun) ** 2
        
        plt.title('Fitting the modeled flux to the values of observed flux (extinction fixed)')
        plt.xlabel('Wavelength (μm)')
        plt.ylabel('Flux')
        plt.plot(wavelen, minimization_flux, 'o')
        plt.errorbar(wavelen, obs_flux_values, yerr = obs_flux_unc, fmt='o')
        plt.grid()
        plt.legend(['Model flux','Observed flux'])
        plt.show()

    minimization_radius = (minimization_radius * R_sun).to(R_sun)

    print('Value of minimization radius:', minimization_radius)
    print('Table value of radius:', (table_value * R_sun).to(R_sun))
    print('Error in minimization radius:', abs(minimization_radius.value - table_value) / table_value * 100)
    return minimization_radius, minimization_result

def SED_fitting_extinction(star_name, Teff, mettalicity, log_g, distance, table_value, unit, show_plot):
    distance = distance * u.parsec
    wavelen, obs_flux_values_Jy, model_flux_values_Jy, _ = get_angular_diameter(star_name, Teff, mettalicity, log_g, Ebv)
    
    obs_flux_values = flux_unit_change(obs_flux_values_Jy, unit)
    model_flux_values = flux_unit_change(model_flux_values_Jy, unit)

    obs_flux_vals = []
    obs_flux_unc = []
    for i in range(len(wavelen)):
        obs_flux_vals.append(obs_flux_values[i].value.nominal_value)
        obs_flux_unc.append(obs_flux_values[i].value.std_dev)

    obs_flux_values = np.array(obs_flux_vals) * model_flux_values.unit
    obs_flux_unc = np.array(obs_flux_unc) * model_flux_values.unit

    def minimization_function(parameters, distance, wavelen, model_flux_values, obs_flux_values, obs_flux_unc):
        radius, Ebv = parameters
        model_flux_values = flux_extinction(wavelen, model_flux_values, Ebv)
        model_flux_values = model_flux_values * radius**2 / distance.to(R_sun) ** 2
        chi_squared = np.sum(((model_flux_values.value - obs_flux_values.value) / obs_flux_unc.value) ** 2)
        return chi_squared
    
    minimization_result = minimize(minimization_function, x0=(1.0,0.0), args=(distance,wavelen,model_flux_values,obs_flux_values, obs_flux_unc), method='Nelder-Mead')
    minimization_radius, min_Ebv = minimization_result.x[0], minimization_result.x[1]
    if show_plot == 'yes':
        model_flux_values = flux_extinction(wavelen, model_flux_values, min_Ebv)
        minimization_flux = model_flux_values * minimization_radius **2 / distance.to(R_sun) ** 2
        plt.title('Fitting the modeled flux to the values of observed flux (extinction estimated)')
        plt.xlabel('Wavelength (μm)')
        plt.ylabel('Flux')
        plt.plot(wavelen, minimization_flux, 'o')
        plt.errorbar(wavelen, obs_flux_values, yerr = obs_flux_unc, fmt='o')
        plt.grid()
        plt.legend(['Model flux','Observed flux'])
        plt.show()

    minimization_radius = (minimization_radius * R_sun).to(R_sun)

    print('Value of minimization radius:', minimization_radius)
    print('Table value of radius:', (table_value * R_sun).to(R_sun))
    print('Error in minimization radius:', abs(minimization_radius.value - table_value) / table_value * 100)
    return minimization_radius, minimization_result
# %% 
star_name = 'HD 176986'
Teff = 4931
mettalicity = 0.03
log_g = 4.44
distance = 27.9
table_value = 0.805
Ebv = 0.024
table_extinction = 0.010

minimization_radius,result = SED_fitting(star_name, Teff, mettalicity, log_g, Ebv, distance, table_value, 'SI', 'yes')

# %% 
minimization_radius,result = SED_fitting_extinction(star_name, Teff, mettalicity, log_g, distance, table_value, 'SI', 'yes')
print('Extinction is',result.x[1])
print('Table value of extinction:', table_extinction)
print('Error in extinction value:',abs(result.x[1] - table_extinction) / table_extinction * 100)'''