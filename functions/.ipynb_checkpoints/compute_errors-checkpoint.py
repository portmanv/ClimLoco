import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

def err_interval_unconstrained(conf, M):
    wrong = np.array(t.interval(conf, np.inf, loc=0, scale=1))[1]
    right = np.array(t.interval(conf, M-1, loc=0, scale=1))[1] * np.sqrt(1+1/M)
    
    return 100*np.abs(wrong-right)/right

def err_interval_constrained(conf, M, normalized_obs):
    wrong = np.array(t.interval(conf, np.inf, loc=0, scale=1))[1]
    right = np.array(t.interval(conf, M-2, loc=0, scale=1))[1] * np.sqrt(1+1/M+normalized_obs**2/M)
    
    return 100*np.abs(wrong-right)/right


