import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


from functions.compute_probability_intervals import *
from functions.compute_confidence_intervals import compute_confidence_intervals_X_multivariate

# Display parameters
fontsize = 12
lw_low  = 2
w_bar = 0.6


def plot_interval(ax, xposition, interval, color, lw, w_bar=0.4, label=None, linestyle=None, markersize=10, vertical=True):
    center = np.mean(np.array(interval).flatten())
    error  = np.diff(np.array(interval).flatten())[0]/2

    if vertical:
        interval = ax.errorbar(xposition, center, yerr=np.abs(error), color=color, capsize=20*w_bar, fmt="_",
                    lw=lw, capthick=lw, markersize=markersize, label=label, linestyle=linestyle, clip_on=True, zorder=1000)
    else:
        interval = ax.errorbar(center, xposition, xerr=np.abs(error), color=color, capsize=20*w_bar, fmt="_",
                    lw=lw, capthick=lw, markersize=markersize, label=label, linestyle=linestyle, clip_on=True, zorder=1000)
        
    return interval


def display_X_Y_relation(ax, X, Y, X_obs, constrained_CI, constrained_CI_per_x, array_x, lw_low=2, lw_high=3,
                         color_interval="tab:green", color_linear_regression="tab:green"):
    
    markers = ax.scatter(X, Y, label="climate models", s=50, marker="o", facecolor='none', edgecolors="black")
    interval = plot_interval(ax, np.mean(X_obs), constrained_CI, color_interval, lw_low, w_bar=w_bar, label=r"$CI_{90\%}(Y|X^N=x_0^N)$")
    
    ax.fill_between(array_x, constrained_CI_per_x[:,0], constrained_CI_per_x[:,1],
                         alpha=0.15, color=color_linear_regression)
    line = ax.plot(array_x, np.mean(constrained_CI_per_x, axis=1), color=color_linear_regression,
             linewidth=lw_high/2, linestyle='solid', label=r"$y=\hat{b}_0+\hat{b}_1\,x$")
    return markers, interval, line



def display_arrows_axes(ax, remove_x=False, remove_y=False, label_X=r"$X$", label_Y=r"$Y$"):
    # remove the top and right axe
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if remove_x:
        ax.spines['bottom'].set_visible(False)
    
    # display an arrow in the x and y axis
    xmin,xmax=np.copy(ax.get_xlim())
    ymin,ymax=np.copy(ax.get_ylim())
    if not remove_x:
        ax.plot((1), (0), ls="", marker=">", ms=6, color="k", 
                    clip_on=False, transform = ax.transAxes)
    
    ax.plot((0), (1), ls="", marker="^", ms=6, color="k",
            clip_on=False, transform = ax.transAxes)
    ax.set_xlim(xmin,xmax)

    # name of the axes
    if not remove_x:
        ax.text(1,0.07, label_X,
             ha='center', va='center',
             transform = ax.transAxes, fontsize=fontsize)
        ax.text(-0.05,1.0, label_Y,
         ha='center', va='center',
         transform = ax.transAxes, fontsize=fontsize)
    else:
        ax.text(-0.2,1.0, label_Y,
         ha='center', va='center',
         transform = ax.transAxes, fontsize=fontsize)




def display_CI_Y_YX1_YX2(Y, X1, X2, X1_obs, X2_obs, unconstrained_CI,
                        constrained_CI_1, array_x1, constrained_CI_per_x1,
                        constrained_CI_2, array_x2, constrained_CI_per_x2,
                        period_Y, period_X1, period_X2, reference_period):
    
    fig, axes = plt.subplots(1,3, figsize=(7,3), sharey=True, width_ratios=[0.3, 1,1])
    
    #--------------- First axe: the unconstrained interval
    axes[0].scatter(2*np.ones(len(Y)), Y, label="climate models", s=50, marker="o", facecolor='none', edgecolors="black", alpha=1)
    interval_unconstrained = plot_interval(axes[0], 0, unconstrained_CI, "black", lw_low, w_bar=w_bar, label=r"$CI_{90\%}(Y)$")
    axes[0].set_xlim(-2, 3)
    axes[0].set_xticks([])
    
    #--------------- Second and third axes: intervals constrained by X1 and X2
    markers, interval_constrained, _ = display_X_Y_relation(axes[1], X1, Y, X1_obs, constrained_CI_1, constrained_CI_per_x1, array_x1)
    markers, interval_constrained, _ = display_X_Y_relation(axes[2], X2, Y, X2_obs, constrained_CI_2, constrained_CI_per_x2, array_x2)
    
    # Label the axes
    axes[1].set_xlabel("GSAT mean in {}-{} (°C)\nrelative to {}-{}".format(period_X1[0], period_X1[1], reference_period[0], reference_period[1]))
    axes[2].set_xlabel("GSAT trend in {}-{} (°C/yr)\nrelative to {}-{}".format(period_X2[0], period_X2[1], reference_period[0], reference_period[1]))
    axes[0].set_ylabel("GSAT mean in {}-{} (°C)\nrelative to {}-{}".format(period_Y[0], period_Y[1], reference_period[0], reference_period[1]))
    
    # Vertical lines corresponding to the mean
    ymin, ymax = axes[0].get_ylim()
    for i, x in zip([1,2], [np.mean(X1), np.mean(X2)]):
        axes[i].vlines(x, ymin=0, ymax=np.mean(unconstrained_CI), linestyle='dotted',
                   color="black", linewidth=0.8, zorder=1000, alpha=1)
    axes[0].set_ylim(ymin, ymax)
    
    # Horizontal lines corresponding to the unconstrained value
    line_MMM = axes[0].hlines(np.mean(unconstrained_CI), linestyle='dotted', xmin=-2, xmax=0,
                   colors="black", linewidth=0.8, zorder=10000, alpha=1, clip_on=False)
    for i, X in zip([1,2], [X1,X2]):
        xmin, xmax = axes[i].get_xlim()
        axes[i].hlines(np.mean(unconstrained_CI), linestyle='dotted', xmin=xmin, xmax=np.mean(X),
                       colors="black", linewidth=0.8, zorder=10000, alpha=1, clip_on=False)
        axes[i].set_xlim(xmin, xmax)
        
    
    # Display arrows in the axes
    display_arrows_axes(axes[0], remove_x=True)
    display_arrows_axes(axes[1], label_X=r"$X_1$")
    display_arrows_axes(axes[2], label_X=r"$X_2$")
    
    # Display the multi-model mean and observation in the x axis
    for i, X_obs in zip([1,2], [X1_obs, X2_obs]):
        axes[i].text(np.mean(X_obs), ymin+0.2, "obs", ha='center', va='center', color="tab:green")
        axes[i].vlines(np.mean(X_obs), ymin=ymin-0.5, ymax=ymin+0.08, color="tab:green")
    
    leg = fig.legend([interval_unconstrained, interval_constrained, markers, line_MMM],
               ["unconstrained", "constrained", "climate models", "multi-model-mean"],
               loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, title="Confidence interval of Y:")
    leg._legend_box.align = "left"
    
    # Titles
    axes[0].set_title("(a) Unconstrained", pad=20, fontsize=11)
    axes[1].set_title("(b) Constrained by "+r"$X_1$", pad=20, fontsize=11)
    axes[2].set_title("(c) Constrained by "+r"$X_2$", pad=20, fontsize=11)
    
    plt.subplots_adjust(wspace=0.3)#, top=0.2)
    
    plt.show()


def display_figure_key_statistical_concepts(x, y, x_obs,
                                            confidence_level=0.9):

    mu_Y = np.mean(y)
    sigma_Y = np.std(y)
    mu_X = np.mean(x)
    M = len(y)
    df = M
    xmin, xmax = np.min(x)*0.95, np.max(x)*1.05
    
    print("correlation : {:.2f}".format(np.corrcoef(x.flatten(),y)[0,1]))

    #------------ Compute the probability and confidence intervals
    unconstrained_PI = compute_probability_interval_Y(confidence_level, mu_Y, sigma_Y)
    
    # without observational noise: take only the mean x_obs to remove the observational noise
    [constrained_CI, unconstrained_CI, array_x, constrained_CI_per_x
    ] = compute_confidence_intervals_X_multivariate(x, np.mean(x_obs).repeat(200).reshape(-1,1), y, confidence_level, "", "", return_for_different_x=True, display=False)
    
    [error_unconstrained, error_constrained
    ] = compute_confidence_intervals_X_multivariate(x, np.mean(x_obs).repeat(200).reshape(-1,1), y, confidence_level, "", "", return_errors=True, display=False)
    
    
    
    
    fig, axes = plt.subplots(1,3, figsize=(12, 5), sharex='col', sharey='col', width_ratios=[1,1,3], dpi=200)
    
    #------------ Colomn 1, Probability interval of Y
    ax = axes[0]
    color = "black"
    ymin, ymax = mu_Y-3*sigma_Y, mu_Y+3*sigma_Y
    array_y = np.linspace(ymin, ymax, 100)
    array_y_interval = np.linspace(unconstrained_PI[0], unconstrained_PI[1], 100)
    ax.plot(stats.norm.pdf(array_y, mu_Y, sigma_Y), array_y, color=color)
    cross_interval = stats.norm.pdf(unconstrained_PI[0], mu_Y, sigma_Y)
    interval = plot_interval(ax, cross_interval,
                             unconstrained_PI, color, lw_low*1.7, w_bar=w_bar, label="", vertical=True)
    ax.fill_betweenx(array_y_interval, stats.norm.pdf(array_y_interval, mu_Y, sigma_Y), color=color, alpha=0.3)
    #ax.text(np.mean(unconstrained_PI), 0.4, "90% probability"+r" ($1-\alpha$)", ha='center')
    #ax.text((np.max(unconstrained_CI)+ymax)/2, 0.4, "5% ("+r"$\alpha/2$)", ha='center')
    #ax.text((np.min(unconstrained_CI)+ymin)/2, 0.4, "5% ("+r"$\alpha/2$)", ha='center')
    ax.set_xlim(0)
    ax.axhline(unconstrained_PI[0], color="black", linestyle="dashed")
    ax.axhline(unconstrained_PI[1], color="black", linestyle="dashed")
    ax.set_xlabel("density")
    ax.set_title("(a) Probability interval of Y\n({:.1f} - {:.1f})".format(unconstrained_PI[0], unconstrained_PI[1]), pad=15)
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(0, 1.1)
    display_arrows_axes(ax, label_Y=r"$y$", label_X="")
    ax.set_xticks([0.,1.])
    
    
    #------------ Colomn 2, Confidence interval of Y
    ax = axes[1]
    array_y_interval = np.linspace(unconstrained_CI[0], unconstrained_CI[1], 100)
    ax.plot(stats.norm.pdf(array_y, mu_Y, error_unconstrained), array_y, color=color)
    cross_interval = stats.norm.pdf(unconstrained_CI[0], mu_Y, error_unconstrained)
    interval = plot_interval(ax, cross_interval,
                             unconstrained_CI, color, lw_low*1.7, w_bar=w_bar, label="", vertical=True)
    ax.fill_betweenx(array_y_interval, stats.norm.pdf(array_y_interval, mu_Y, error_unconstrained), color=color, alpha=0.3)
    #ax.text(np.mean(unconstrained_CI), 0.4, "90% confidence"+r" ($1-\alpha$)", ha='center')
    #ax.text((np.max(unconstrained_CI)+ymax)/2, 0.4, "5% ("+r"$\alpha/2$)", ha='center')
    #ax.text((np.min(unconstrained_CI)+ymin)/2, 0.4, "5% ("+r"$\alpha/2$)", ha='center')
    ax.set_xlim(0)
    #ax.scatter(y, 0.5*np.ones(len(y)), s=50, marker="o", facecolor='none', edgecolors="black")
    ax.scatter(stats.norm.pdf(y, mu_Y, error_unconstrained), y, s=50, marker="o", facecolor='none', edgecolors="black")
    ax.axhline(unconstrained_CI[0], color="black", linestyle="dashed")
    ax.axhline(unconstrained_CI[1], color="black", linestyle="dashed")
    ax.set_xlabel("density")
    ax.set_title("(b) Confidence interval of Y\n({:.1f} - {:.1f})".format(unconstrained_CI[0], unconstrained_CI[1]), pad=15)
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(0, 1.1)
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    display_arrows_axes(ax, label_Y=r"$y$", label_X="")
    ax.set_xticks([0.,1.])
    
    
    #------------ Colomn 3, Confidence interval of Y constrained
    ax = axes[2]
    markers, interval, line = display_X_Y_relation(ax, x, y, x_obs, constrained_CI, constrained_CI_per_x, array_x, lw_low=lw_low*1.7,
                                             color_interval="black", color_linear_regression="tab:red")
    ax.set_title("(c) Confidence interval of "+r"$Y|X=x_0$"+"\n({:.1f} - {:.1f})".format(constrained_CI[0], constrained_CI[1]), pad=15)
    display_arrows_axes(ax, label_X=r"$x$", label_Y=r"$y$")
    
    ax.text(np.mean(x_obs), ymin*0.95, r"$x_0$", ha='center', va='center', fontsize=14, color='tab:blue')
    obs_line = ax.vlines(np.mean(x_obs), -1000, constrained_CI[0], linestyle='dashed', color='tab:blue')
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xmin, xmax)
    ax.hlines(constrained_CI[0], -1000, np.mean(x_obs), color="black", linestyle="dashed")
    ax.hlines(constrained_CI[1], -1000, np.mean(x_obs), color="black", linestyle="dashed")
    
    ax.legend([markers, interval]+line,
               ["climate models", "confidence interval", r"$y=a_0+a_1\,x$"], fontsize=11)
    
    fig.subplots_adjust(wspace=0.4)
    
    plt.show()


    