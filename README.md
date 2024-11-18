Requirements: 

Python modules: 
  - numpy
  - astropy
  - astroquery
  - matplotlib
  - uncertainties
  - scipy
  - pysynphot (check below further requirement related to pysynphot)
  - pandas
  - extinction

Pysynphot requirement: For the functions in pysynphot to work the Kurucz-Castelli models must be downloaded and the folder trds saved.In the terminal the line "export PYSYN_CDBS="/mydir/trds/""  (where mydir corresponds to path to where the models are saved) must be run. 

