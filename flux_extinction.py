# %% 
from extinction import ccm89, apply
from get_flux_values import gaia_values, get_flux_values
from sympy import solve, symbols
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt

 #%% 

def intrinsic_color(Teff, mettalicity):
    x = symbols('x')
    sol = solve(8939 - 6395 * x + 2381 * x**2 - Teff + 451 * mettalicity + 154*mettalicity**2)
    if sol[0] < .40 or sol[0] > 1.20:
        sol.pop(0)
    if sol[1] < .40 or sol[1] > 1.20:
        sol.pop(1)   
    return sol

def color_excess(Teff, mettalicity, gaia_id):
    gaia_vals = gaia_values(gaia_id)
    color = float(gaia_vals[0]['BPmag']) - float(gaia_vals[0]['Gmag'])
    int_color =  intrinsic_color(Teff, mettalicity)
    return color - int_color[0]

def flux_extinction(wavelen, flux, Teff, mettalicity, gaia_id):
    col_exc = color_excess(Teff, mettalicity, gaia_id)
    print("E(B-V) value:", col_exc)
    flux_ext = apply(ccm89(wavelen.to(u.angstrom), abs(col_exc*3.1), 3.1), flux)
    return flux_ext
