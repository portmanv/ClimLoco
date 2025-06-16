# ClimLoco
This repository contains the Python notebooks and data used to produce the figures and values in the article "ClimLoco1.0: CLimate variable confidence Interval of Multivariate Linear Observational COnstraint". You can download and run the repository on your own device.

The "notebooks" repository contains three notebooks:
    main_synthetic_data.ipynb: produces the main figures in the article and the synthetic data required to generate them.
    main_real_data.ipynb applies CLimLoco1.0 to real data in a case study based on global mean temperature.
    install_packages.ipynb runs to install all the necessary packages (see the "Installation" paragraph below).
    
The "functions" repository contains the different functions used in the notebooks.

The "data" repository contains the different datasets used in the notebooks. This data repository contains various files:
- The "ssp245_timeserie_global_tas" repository contains timeseries of global mean temperature between 180 and 2100 for 32 climate models from CMIP6, using the SSP2-4.5 scenario. This data is the result of preprocessing data from the ESPRI database to calculate the global average. The raw data is too large.
- The file called 'HadCRUT.5.0.2.0.analysis.ensemble_series.global.annual.nc' contains observed data from the HadCRUT5 reanalysis.
    
# Use of ClimLoco
As illustrated in the notebook 'notebooks/main_real_data.ipynb', the inputs and outputs of ClimLoco are:
- (inputs) alpha (the confidence level; e.g. 90%), X and Y (e.g. past and future temperatures, respectively), the observational dataset (e.g. HadCRUT5) and the climate model ensemble used (e.g. CMIP6).
- (output) the confidence interval of Y, constrained by real-world (noisy) observations of X.


# Installation:
The required packages may have conflicting versions. We recommend using a virtual environment where the package versions are fixed. We propose the following procedure:
Install Anaconda: https://docs.anaconda.com/anaconda/install/.
Open Anaconda Navigator.
- Go to the 'Environments' tab. Create a new environment named 'toolbox', for example. In this step, choose Python version 3.9.19.
- Go to the "Home" tab. Make sure you are using the new "toolbox" environment, indicated at the top. Click on 'Install' on Jupyter Notebook.
Click on 'Launch' on Jupyter Notebook. A new tab should open in your browser. Open the 'ClimLoco' file and go to /notebook.
Open and run 'install_packages.ipynb' to automatically install all the necessary packages.