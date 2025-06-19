import numpy as np
from scipy.stats import t


def compute_probability_interval_Y(confidence_level, mu_Y_theo, sigma_Y_theo):
    """
    Probability interval of Y : Eq. 2 in the article (warning: only for X univariate)
    """
    probability_interval_Y = mu_Y_theo + np.array(t.interval(confidence_level, np.inf, loc=0, scale=1))*sigma_Y_theo
    return probability_interval_Y

def compute_probability_interval_Y_X_noiseless(confidence_level, X_obs,
                                               mu_X_theo, sigma_X_theo, mu_Y_theo, sigma_Y_theo, corr_theo):
    """
    Probability interval of Y constrained by a noiseless observation of X : Eq. 14 in the article (warning: only for X univariate)
    """
    # Compute the linear coefficients of the linear relation Y = beta_0 + beta_1 * X + epsilon
    cov_XY_theo = corr_theo*sigma_X_theo*sigma_Y_theo  # Theoretical covariance between X and Y
    beta_1      = cov_XY_theo/sigma_X_theo**2 # Slope
    beta_0      = mu_Y_theo - beta_1*mu_X_theo # Intercept
    
    probability_interval_Y_X_noiseless = beta_0+beta_1*X_obs + np.array(t.interval(confidence_level, np.inf, loc=0, scale=1))*sigma_Y_theo*np.sqrt(1-corr_theo**2)

    return probability_interval_Y_X_noiseless

def compute_probability_interval_Y_X_noisy(confidence_level, X_obs, sigma_N,
                                               mu_X_theo, sigma_X_theo, mu_Y_theo, sigma_Y_theo, corr_theo):
    """
    Probability interval of Y constrained by a noisy observation of X : Eq. 24 in the article (warning: only for X univariate)
    """
    # Compute the linear coefficients of the linear relation Y = beta_0_noisy + beta_1_noisy * X_noisy + epsilon_noisy
    cov_XY_theo  = corr_theo*sigma_X_theo*sigma_Y_theo  # Theoretical covariance between X and Y
    beta_1_noisy = cov_XY_theo/(sigma_X_theo**2+sigma_N**2)
    beta_0_noisy = mu_Y_theo - beta_1_noisy*mu_X_theo
    
    # Signal-to-noise ratio
    SNR_theo      = sigma_X_theo/sigma_N
    
    probability_interval_Y_X_noiseless = beta_0_noisy+beta_1_noisy*X_obs + np.array(t.interval(confidence_level, np.inf, loc=0, scale=1)
                                                            )*sigma_Y_theo*np.sqrt(1-(corr_theo**2/(1+1/SNR_theo**2)))
    return probability_interval_Y_X_noiseless



