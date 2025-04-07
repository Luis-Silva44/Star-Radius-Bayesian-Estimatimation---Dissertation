# %% 
from MCMC_temperature import star_tester_main
from MCMC_complete import star_tester_complete 


import time
import math 
from astropy.constants import R_sun
import pandas as pd
import os
from astropy import units as u
import numpy as np 
import matplotlib.pyplot as plt
# %% 
output_file = '/home/luis/tese/testdata/star_results.csv'

def main_star_tester(star_list,output_file):

    problem_stars = []

    columns = ['Star', 'Expected_Radius', 'e_Expected_Radius', 'Computed_Radius', 'e_Computed_Radius']

    if os.path.exists(output_file):
        results_df = pd.read_csv(output_file)
    else:
        results_df = pd.DataFrame(columns=columns)

    for i in range(len(star_list)):

        star_name = star_list['Star'][i]
        print('Star being tested:', star_name)
        if star_name in results_df['Star'].values:
            print(f"{star_name} already in results. Skipping.")
            continue

        Teff = star_list['Teff'][i]
        e_Teff = star_list['eTeff'][i]
        metallicity = star_list['Fe/H'][i]
        e_metal = star_list['eFe/H'][i]
        log_g = star_list['logg'][i]
        e_log_g = star_list['elogg'][i]
        Ebv = float(star_list['E(B-V)'][i])
        table_value = star_list['Radius'][i]
        e_table_value = star_list['erRadius'][i]

        table_value = (table_value * R_sun).to(R_sun)
        e_table_value = (e_table_value * R_sun).to(R_sun)

        try:
            radius, e_radius, _ = star_tester_main(star_name, Teff, e_Teff, log_g, e_log_g, metallicity, e_metal, Ebv, table_value, nwalkers=30, npoints=1200)

            if abs(radius.value - table_value.value)*100 > 5:
                problem_stars.append(star_name)
                print('Issue with star ', star_name,':Error too high')

            new_row = {
                'Star': star_name,
                'Expected_Radius': table_value.value,
                'e_Expected_Radius': round(e_table_value.value,3),
                'Computed_Radius': round(radius.value, 3),
                'e_Computed_Radius': round(e_radius, 3)
            }

            results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
        except Exception as e:
            problem_stars.append(star_name)
            print(f"Issue with star {star_name}: {e}")
            print('--------')

        results_df.to_csv(output_file, index=False, float_format='%.3f')

    return problem_stars

# %% 

star_data = pd.read_csv('~/tese/testdata/list_stars.txt', sep="\t", header=0, skiprows=[1])

problem_stars = main_star_tester(star_data,output_file)

# %%
results = pd.read_csv(output_file)
results
# %%
exp_radius = np.array(results['Expected_Radius'])
e_exp_radius = np.array(results['e_Expected_Radius'])

comp_radius = np.array(results['Computed_Radius'])
e_comp_radius = np.array(results['e_Computed_Radius'])

fig, axs = plt.subplots()
axs.errorbar(exp_radius, comp_radius, e_comp_radius, e_exp_radius, 'o', ecolor='red')
axs.axline((0.60,0.60), slope=1, color='black', linestyle='--')
axs.set_xlim(0.6)
axs.set_ylim(0.6)

# %%
def add_extinction_radius(output_file, star_list):
    problem_stars = []
    data = pd.read_csv(output_file)

    if 'Extinction_Radius' not in data.columns:
        data['Extinction_Radius'] = np.nan
    if 'e_Extinction_Radius' not in data.columns:
        data['e_Extinction_Radius'] = np.nan

    for id, row in data.iterrows():
        star_name = row['Star']
        print('Adding to star', star_name)

        if math.isnan(row['Extinction_Radius']):

            match = star_list[star_list['Star'] == star_name]
            
            try: 
                match_row = match.iloc[0]

                Teff = match_row['Teff']
                e_Teff = match_row['eTeff']
                metallicity = match_row['Fe/H']
                e_metal = match_row['eFe/H']
                log_g = match_row['logg']
                e_log_g = match_row['elogg']
                exp_Ebv = float(match_row['E(B-V)'])
                table_value = match_row['Radius']
                e_table_value = match_row['erRadius']

                # Convert to astropy quantities
                table_value = (table_value * R_sun).to(R_sun)
                e_table_value = (e_table_value * R_sun).to(R_sun)

                # Call your model
                ext_radius, e_ext_radius, _ = star_tester_complete(
                    star_name, Teff, e_Teff, log_g, e_log_g,
                    metallicity, e_metal, exp_Ebv,
                    table_value.value,
                    nwalkers=30, npoints=1200
                )

                # Save results
                data.at[id, 'Extinction_Radius'] = round(ext_radius.value, 3)
                data.at[id, 'e_Extinction_Radius'] = round(e_ext_radius, 3)

                data.to_csv(output_file, index=False, float_format='%.3f')
            except Exception as e:
                print(f"Could not compute extinction radius for {star_name}: {e}")
                problem_stars.append(star_name)
                
    return problem_stars


# %%
output_file = '/home/luis/tese/testdata/star_results.csv'
star_data = pd.read_csv('~/tese/testdata/list_stars.txt', sep="\t", header=0, skiprows=[1])

add_extinction_radius(output_file, star_data)
# %%
results = pd.read_csv(output_file)
exp_radius = np.array(results['Expected_Radius'])
e_exp_radius = np.array(results['e_Expected_Radius'])

comp_radius = np.array(results['Extinction_Radius'])
e_comp_radius = np.array(results['e_Extinction_Radius'])

fig, axs = plt.subplots()
axs.errorbar(exp_radius, comp_radius, e_comp_radius, e_exp_radius, 'o', ecolor='red')
axs.axline((0.60,0.60), slope=1, color='black', linestyle='--')
#axs.set_xlim(0.6)
#axs.set_ylim(0.6)
# %%
