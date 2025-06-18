import numpy as np
from scipy.stats import t
from sklearn.linear_model import LinearRegression

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



def compute_confidence_intervals_X_multivariate(X, X_obs, Y, confidence_level, period_Y, reference_period,
                                 display=True, return_for_different_x=False,
                                 neglect_M=False, impose_SNR=0, precomputed_Cov_N=0, return_errors=False):
    M, K     = X.shape
    if neglect_M:
        M = 99999999
    mu_Y     = np.mean(Y)
    std_Y    = np.std(Y, ddof=1)
    mu_X     = np.mean(X, axis=0)
    mu_X_obs = np.mean(X_obs, axis=0)
    Cov_XY   = np.array([np.cov(Y, X[:, id_comp], rowvar=False, ddof=1)[0,1] for id_comp in range(K)])
    Cov_X    = np.cov(X, rowvar=False, ddof=1).reshape(K,K)
    

    if impose_SNR: # for special test, where you want a specific signal to nosie ratio (SNR)
        Cov_N = Cov_X / impose_SNR**2
    elif precomputed_Cov_N:
        Cov_N = precomputed_Cov_N
    else:
        Cov_N = np.cov(X_obs, rowvar=False, ddof=1)
        
    Cov_N = Cov_N.reshape(K,K)
    
    b1 = Cov_XY @ np.linalg.inv(Cov_X+Cov_N)
    b0 = mu_Y - b1 @ mu_X
    
    var_eps     = np.var(Y - b0 - X @ b1, ddof=1)
    sigma_eps_N = np.sqrt(var_eps + b1 @ Cov_N @ b1)
    root_term   = np.sqrt(1 + 1/M + (mu_X_obs-mu_X) @ np.linalg.inv(Cov_X+Cov_N) @ (mu_X_obs-mu_X)/M)

    constrained_CI   = b0 + np.matmul(b1, mu_X_obs) + np.array(t.interval(confidence_level, M-1-K, loc=0, scale=1))*sigma_eps_N*root_term
    unconstrained_CI = mu_Y + np.array(t.interval(confidence_level, M-1, loc=0, scale=1))*std_Y*np.sqrt(1+1/M)

    if display:
        print("{}% confidence:".format(int(100*confidence_level)))
        print("Global temperature at surface in {}-{} relative to {}-{}:".format(period_Y[0], period_Y[1], reference_period[0], reference_period[1]))
        print("Constrained : [{:.2f} +- {:.2f}]".format(np.mean(constrained_CI), np.diff(constrained_CI)[0]/2))
        print("Unconstrained : [{:.2f} +- {:.2f}]".format(np.mean(unconstrained_CI), np.diff(unconstrained_CI)[0]/2))

    if return_for_different_x:
        array_x = np.linspace(np.min((X.min(), mu_X_obs.min())), np.max((mu_X_obs.max(), X.max())), 100)
        constrained_CI_per_x = np.zeros((len(array_x), 2))
        for id_x in range(len(array_x)):
            mu_X_obs = array_x[id_x].reshape(-1,1)
            root_term  = np.sqrt(1 + 1/M + (mu_X_obs-mu_X) @ np.linalg.inv(Cov_X+Cov_N) @ (mu_X_obs-mu_X)/M)
            constrained_CI_per_x[id_x, :] = b0 + np.matmul(b1, mu_X_obs) + np.array(t.interval(confidence_level, M-1-K, loc=0, scale=1))*sigma_eps_N*root_term
        
        return constrained_CI, unconstrained_CI, array_x, constrained_CI_per_x

    if return_errors:
        error_unconstrained = std_Y*np.sqrt(1+1/M)
        error_constrained   = sigma_eps_N*root_term
        return error_unconstrained, error_constrained
        
    return constrained_CI, unconstrained_CI




def compute_Cox_confidence_intervals(X, X_obs, Y, confidence_level, var_N):
    M     = len(X)
    mu_Y  = np.mean(Y)
    mu_X  = np.mean(X)
    var_X = np.var(X, ddof=1)
    
    lr = LinearRegression().fit(X, Y)
    coefs = lr.coef_
    estimated_value = lr.predict(X_obs.reshape(1, -1))
    
    z         = t.interval(confidence_level, np.inf, loc=0, scale=1)[1]
    var_eps   = np.var(Y - lr.predict(X), ddof=1)
    sample_uncertainty = 1 + 1/M + (X_obs-mu_X)**2 / (M * var_X)
    proj_var_noise = coefs**2 * var_N


    total_uncertainty = z*np.sqrt(var_eps * sample_uncertainty + proj_var_noise)

    constrained_CI   = (estimated_value + np.array([-total_uncertainty, total_uncertainty])).flatten()
    return constrained_CI




