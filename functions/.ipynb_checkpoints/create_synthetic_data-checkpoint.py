import numpy as np

def generate_synthetic_data_sets(size_per_dataset, mu_X_theo, sigma_X_theo, mu_Y_theo, sigma_Y_theo, corr_theo):
    """
    This function generates multiple (X,Y) datasets, each one with a different number of samples indicated in size_per_dataset.
    The datasets generated shares the same samples, but have different size.
    For example, if size_per_dataset=[5, 30], the first data set is included in the 5 first samples of the second data set.
    
    The relationship between Y and X is given by Y =  beta_0 + beta_1 * X + epsilon.
    X and epsilon are taken randomly following a gaussian distribution, making random values of Y using the equation above.
        
    size_per_dataset : the number of samples to take per dataset
    mu_X_theo : theoretical expectation of the distribution of X
    sigma_X_theo : theoretical standard deviation of the distribution of X
    mu_Y_theo : theoretical expectation  of the distribution of Y
    sigma_Y_theo : theoretical standard deviation of the distribution of Y
    corr_theo : theoretical correlation between X and Y
    """
    #-------- Compute the linear coefficients of the linear relation Y = beta_0 + beta_1 * X + epsilon
    cov_XY_theo  = corr_theo*sigma_X_theo*sigma_Y_theo  # Theoretical covariance between X and Y
    sigma_eps_theo = np.sqrt(sigma_Y_theo**2 - cov_XY_theo**2 / sigma_X_theo**2) # Theoretical standard deviation of epsilon
    beta_1   = cov_XY_theo/sigma_X_theo**2 # Slope
    beta_0   = mu_Y_theo - beta_1*mu_X_theo # Intercept
    
    #-------- Compute the datasets
    # Prepare storage
    X_simu_per_dataset = []
    Y_simu_per_dataset = []
    nb_datasets = len(size_per_dataset)

    # Prepare the realisations of X and epsilon
    np.random.seed(300)
    X_simu_     = np.random.normal(mu_X_theo, sigma_X_theo, 1000) # Creation of a realisation of X
    epsilon_    = np.random.normal(0, sigma_eps_theo, 1000) # Creation of a realisation of epsilon

    # Extract the right nb of samples of X and epsilon, then Y using the linear regression formulae
    for i in range(nb_datasets):
        size    = size_per_dataset[i] # get the nb of samples to get
        X_simu  = X_simu_[:size] # extract the right nb of samples on X
        epsilon = epsilon_[:size] # extract the right nb of samples on epsilon
        Y_simu  = beta_0 + beta_1*X_simu + epsilon # compute Y

        # Storage 
        X_simu_per_dataset.append(X_simu)
        Y_simu_per_dataset.append(Y_simu)


    # Extract the min and max (useful for the figures)
    Y_simu_  = beta_0 + beta_1*X_simu_
    Ymin,Ymax = Y_simu_.min(), Y_simu_.max()
    Xmin,Xmax = X_simu_.min(), X_simu_.max()

    return X_simu_per_dataset, Y_simu_per_dataset, Ymin, Ymax, Xmin, Xmax

    