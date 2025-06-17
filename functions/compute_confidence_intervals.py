import numpy as np
from scipy.stats import t

def compute_confidence_interval_Y(confidence_level, X_simu_per_dataset, Y_simu_per_dataset):
    """
    Confidence interval of Y : Eq. 6 in the article
    """
    # Compute and store the estimated interval for each dataset
    list_estimated_interval = []
    for X_simu, Y_simu in zip(X_simu_per_dataset, Y_simu_per_dataset):
        M                  = len(Y_simu)
        mu_Y_est           = np.mean(Y_simu)
        sigma_Y_est        = np.std(Y_simu)
        estimated_interval = mu_Y_est + np.array(t.interval(confidence_level, M-1, loc=0, scale=1))*sigma_Y_est*np.sqrt(1+1/M)
        list_estimated_interval.append(estimated_interval)

    return list_estimated_interval


def compute_confidence_interval_Y_X_noiseless(confidence_level, X_simu_per_dataset, Y_simu_per_dataset, X_obs):
    """
    Confidence interval of Y constrained by a noiseless observation of X : Eq. 15 in the article
    """
    # Compute and store the estimated interval for each dataset
    list_estimated_interval = []
    for X_simu, Y_simu in zip(X_simu_per_dataset, Y_simu_per_dataset):
        M                = len(Y_simu)
        mu_Y_est         = np.mean(Y_simu)
        mu_X_est         = np.mean(X_simu)
        sigma_Y_est      = np.std(Y_simu)
        sigma_X_est      = np.std(X_simu)
        beta_1_est       = np.cov(X_simu,Y_simu)[0,1]/sigma_X_est**2
        beta_0_est       = mu_Y_est - beta_1_est*mu_X_est
        corr_est         = np.corrcoef(X_simu,Y_simu)[0,1]

        estimated_interval = beta_0_est+beta_1_est*X_obs + np.array(t.interval(confidence_level, M-2, loc=0, scale=1)
                                                    )*sigma_Y_est*np.sqrt(1-corr_est**2)*np.sqrt(1+1/M+(X_obs-mu_X_est)**2/(M*sigma_X_est**2))      

        list_estimated_interval.append(estimated_interval)

    return list_estimated_interval


def compute_confidence_interval_Y_X_noisy(confidence_level, X_simu_per_dataset, Y_simu_per_dataset, X_obs, sigma_N):
    """
    Confidence interval of Y constrained by a noiseless observation of X : Eq. 25 in the article
    """
    # Compute and store the estimated interval for each dataset
    list_estimated_interval = []
    for X_simu, Y_simu in zip(X_simu_per_dataset, Y_simu_per_dataset):
        M                = len(Y_simu)
        mu_Y_est         = np.mean(Y_simu)
        mu_X_est         = np.mean(X_simu)
        sigma_Y_est      = np.std(Y_simu)
        sigma_X_est      = np.std(X_simu)
        corr_est         = np.corrcoef(X_simu,Y_simu)[0,1]
        SNR_est          = np.std(X_simu)/sigma_N
        beta_1_noisy_est = np.cov(X_simu,Y_simu)[0,1]/(sigma_X_est**2+sigma_N**2)
        beta_0_noisy_est = mu_Y_est - beta_1_noisy_est*mu_X_est
    
        estimated_interval = beta_0_noisy_est+beta_1_noisy_est*X_obs + np.array(t.interval(confidence_level, M-2, loc=0, scale=1)
                        )*sigma_Y_est*np.sqrt(1-(corr_est**2/(1+1/SNR_est**2))) * np.sqrt(1+1/M+(X_obs-mu_X_est)**2/(M*(sigma_X_est**2+sigma_N**2)))

        list_estimated_interval.append(estimated_interval)

    return list_estimated_interval


