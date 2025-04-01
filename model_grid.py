# %% 
import glob
import os
from astropy.io import fits
import numpy as np
import astropy.units as u
# %%
folder_path = '/home/luis/pysynphot_models/trds/grid/ck04models'

def load_model_grid(folder_path):
    u.flam = u.erg / u.s / u.cm**2 / u.angstrom
    model_grid = {}

    metallicity_folders = glob.glob(os.path.join(folder_path, 'ck*'))
    
    for met_folder in metallicity_folders:
        fits_files = glob.glob(os.path.join(met_folder, "*.fits"))

        for file in fits_files:
            with fits.open(file) as hdul: 
                hdr = hdul[0].header
                data_hdr = hdul[1].header
                data = hdul[1].data
                
                Teff = float(hdr['TEFF'])
                metallicity = float(hdr['LOG_Z'])
                wavelength = (data['WAVELENGTH'] * u.angstrom).to(u.um)
                
                for i in np.arange(2, 13):
                    log_g = str(data_hdr['TTYPE' + str(i)])
                    log_g = float(log_g[1:]) / 10

                    flux_flag = data_hdr['TTYPE' + str(i)]
                    flux = (data[flux_flag] * u.flam).to(u.Jy, equivalencies=u.spectral_density(wavelength))
                    
                    if Teff not in model_grid: 
                        model_grid[Teff] = {}
                    if metallicity not in model_grid[Teff]:
                        model_grid[Teff][metallicity] = {}

                    model_grid[Teff][metallicity][log_g] = (wavelength, flux.value)

    sorted_model_grid = {Teff: 
                            {met: 
                                {logg: model_grid[Teff][met][logg] 
                                 for logg in sorted(model_grid[Teff][met])} 
                             for met in sorted(model_grid[Teff])} 
                         for Teff in sorted(model_grid)}      
      
    return sorted_model_grid
# %%
