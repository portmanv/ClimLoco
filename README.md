# ClimLoco
This repository contains the notebooks and data used to produce the figures and values in the article called "ClimLoco1.0: CLimate variable confidence Interval of Multivariate Linear Observational COnstraint". You can download it and run it on your own device. 

The repository named "notebooks" contains 3 notebooks:
    main_synthetic_data.ipynb: produces the main figures of the article, and the synthetic data necessary to produce them,
    main_real_data.ipynb: apply CLimLoco1.0 to real data, in a case study based on the global mean temperature,
    install_packages.ipynb: run it to install all the necessary packages (see below the paragraph "installation")
    
The repository named "functions" contains the different functions used in the notebooks.

The repository named "data" contains the different dataset used in the notebooks. This data repository contains different files:
    The repository called "ssp245_timeserie_global_tas", containing the timeseries of global mean temperature between 180 and 2100, for 32 climate models from CMIP6 using the scenario SSP2-4.5. This data results from a preprocessing of data from the database ESPRI, that make a global average. The raw data are too heavy.
    The file called "HadCRUT.5.0.2.0.analysis.ensemble_series.global.annual.nc" contains the observed data, from the reanalysis HadCRUT 5.
    

# Installation 
The needed packages may have conflictual versions. We recommand the use of a virtual environment where the packages versions are fixed. We proposes this procedure:
- Install anaconda https://docs.anaconda.com/anaconda/install/
- Open Anaconda Navigator
- Go to the tab "Environments". Create a new environment named for example toolbox. In this step I have chosen python version 3.9.19.
- Go to the tab "Home". Make sure you use the new environment created, "toolbox", indicated on the top. Click on "Install" on Jupyter Notebook.
- Click on "Launch" on Jupyter Notebook. A tab should open on your browser. Open the "ClimLoco" file, go to /notebook
- Open and run "install_packages.ipynb" to install automatically all needed packages.
