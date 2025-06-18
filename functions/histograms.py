import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np

def plot_histogram(ax, X, color="tab:blue", alpha=0.5, label="multi-model"):
    if len(label):
        label_hist = label+" density histogram"
        label_line = label+" fitted gaussian"
    else:
        label_hist, label_line = "", ""
    x = np.linspace(X.min(), X.max(), 2000)
    pdf = stats.norm.pdf(x, np.mean(X), np.std(X))
    hist = ax.hist(X, density=True, color=color, alpha=alpha, label=label_hist, bins=8)
    line = ax.plot(x, pdf, color=color, label=label_line)
    return hist, line


def plot_histograms(ax, X, title="", xlabel="", ylabel="", X_obs=[],
                    color_simu="tab:blue", color_obs="tab:green",
                    label_simu="multi-model", label_obs="observed"):
    plot_histogram(ax, X, color=color_simu, label=label_simu)
    
    ax.set_title(title)
    ax.tick_params(axis='y', labelcolor=color_simu)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if len(X_obs)>0:
        ax2 = ax.twinx()
        plot_histogram(ax2, X_obs, color=color_obs, label=label_obs)
        ax2.tick_params(axis='y', labelcolor=color_obs)