import numpy as np
from sklearn.linear_model import LinearRegression


def compute_trend_per_sample(data_perSample_perTime, times):
    nb_samples, nb_times = data_perSample_perTime.shape
    trend_per_sample     = np.nan*np.zeros(nb_samples)
    for id_sample in range(nb_samples):
        timeseries   = data_perSample_perTime[id_sample]
        notNan_times = np.logical_not(np.isnan(timeseries))
        trend        = LinearRegression().fit(times[notNan_times].reshape(-1, 1), timeseries[notNan_times]).coef_[0]
        trend_per_sample[id_sample] = trend

    return trend_per_sample

def compute_X_from_period(period, times, data_perSample_perTime, trend=False):
    times_to_keep = np.logical_and(period[0]<=times, times<=period[1])
    if not trend:
        X = np.nanmean(data_perSample_perTime[:, times_to_keep], axis=1)
    else:
        X = compute_trend_per_sample(data_perSample_perTime[:, times_to_keep], times[times_to_keep])
    return X


def compute_correlation_from_period(period, CMIP6_times, CMIP6_global_tas, Y, period_Y, trend=False):
    X = compute_X_from_period(period, CMIP6_times, CMIP6_global_tas, trend=trend)
    corr = np.corrcoef(X,Y)[0,1]
    if trend:
        print("the correlation between the global tas averaged in {}-{} and trend in {}-{} is : {:.2f}".format(
            period_Y[0], period_Y[1], period[0], period[1], corr))
    else:
        print("the correlation between the global tas averaged in {}-{} and averaged in {}-{} is : {:.2f}".format(
            period_Y[0], period_Y[1], period[0], period[1], corr))