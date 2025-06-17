

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t


# DISPLAY PARAMETERS
# Confidence of the confidence interval
lw = 2
dpi = 300
fontsize = 12
lw_low  = 2
lw_high = 3
w_bar = 0.6
color_cons_noiseless = "tab:red"
color_cons_noisy = "tab:green"




def plot_interval(ax, xposition, interval, color, lw, w_bar=0.4, label=None, linestyle=None, markersize=10):
    center = np.mean(np.array(interval).flatten())
    error  = np.diff(np.array(interval).flatten())[0]/2
    ax.errorbar(xposition, center, yerr=np.abs(error), color=color, capsize=20*w_bar, fmt="_",
                lw=lw, capthick=lw, markersize=markersize, label=label, linestyle=linestyle, clip_on=True, zorder=1000)


def display_figure2(mu_X_theo, sigma_X_theo, confidence_level):
    M = 10
    np.random.seed(5)
    fig, ax = plt.subplots(1,1, dpi=dpi, figsize=(12,3))
    k = 0
    
    lw_    = lw/1.3
    w_bar_ = 0.4/4
    s      = 5
    for i in range(100):
        Y_simu_intervals = np.random.normal(mu_X_theo, sigma_X_theo, M)
        new_Y = np.random.normal(mu_X_theo, sigma_X_theo, 1)[0]
        
        interval = np.mean(Y_simu_intervals) + np.array(t.interval(confidence_level, M-1, loc=0, scale=1))*np.std(Y_simu_intervals)*np.sqrt(1+1/M)
        if (interval.min()<=new_Y) and (new_Y<=interval.max()):
            color="black"
        else:
            color='tab:red'
            k+= 1
        if i==0:
            label_interval= "One realisation of "+r"$CI_{90\%}(Y)$"
            labels_Ys="One realisation of "+r"$Y$"
        else:
            label_interval=""
            labels_Ys = ""
        plot_interval(ax, i, interval, color, lw_, w_bar=w_bar_, markersize=w_bar_, label=label_interval)
        ax.scatter(i, new_Y, color="tab:red", s=s, label=labels_Ys)
        
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("100 realisations of the confidence interval", fontsize=fontsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    xmin,xmax=np.copy(ax.get_xlim())
    ymin,ymax=np.copy(ax.get_ylim())
    xmin, xmax = -1, 100
        
    ax.plot((xmax), (ymin), ls="", marker=">", ms=6, color="k", #transform=ax.get_yaxis_transform(), 
                clip_on=False)
    ax.plot((xmin), (ymax), ls="", marker="^", ms=6, color="k", #transform=ax.get_xaxis_transform(), 
            clip_on=False)
    
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    ax.text(xmin, ymax+0.5, r'$y$', fontsize=fontsize,
                       horizontalalignment='center', verticalalignment='center')
    
    
    # Legend
    plt.legend(fontsize=fontsize, ncol=2) # loc='center left', bbox_to_anchor=(1, 0.5), 
    plt.show()
    








def display_figure3(probability_interval_Y, confidence_interval_Y_per_dataset, X_simu_per_dataset, Y_simu_per_dataset):
    step      = 0
    tiny_step = 0.08
    lw        = 2
    label1 = r"$PI_{90\%}(Y)$"
    label3 = r"$CI_{90\%}(Y)$"
       
    colors = ["black", "black", "black"]
    
    fig, axes = plt.subplots(1,1, figsize=(6,3), sharey=True, dpi=dpi)
    plot_interval(axes, -1, probability_interval_Y, "black", lw_low, w_bar=w_bar, label=None)

    nb_datasets = len(confidence_interval_Y_per_dataset)
    for i in np.flip(np.arange(nb_datasets)):
        X_simu = X_simu_per_dataset[i]
        Y_simu = Y_simu_per_dataset[i]
        interval_unconstrained_est = confidence_interval_Y_per_dataset[i]
    
        label1 = "Theoretical "+r"$PI_{90\%}(Y)$"
        label3 = "Realisation of "+r"$CI_{90\%}(Y)$"
    
        if i==0:
            label="One climate model"
        else:
            label=None
        axes.scatter(i*np.ones(len(X_simu))+step+tiny_step, Y_simu, s=50, marker="o", facecolor='none', edgecolors=colors[i],
                     label=label)
        plot_interval(axes, i+step-tiny_step, interval_unconstrained_est, colors[i], lw_low, w_bar=w_bar, label=None)
    
    if False:
        axes.set_xticks([0, 1])
        axes.set_xticklabels(["Sample of\n{} climate models".format(len(X_simu_per_dataset[i])) for i in np.arange(nb_datasets)],
                             fontsize=fontsize)#, rotation=45)
        for xtick, color in zip(axes.get_xticklabels(), [colors[i] for i in np.arange(nb_datasets)]):
            xtick.set_color(color)
    else:
        axes.set_xticks([-1, 0,1])
        axes.set_xticklabels([r"$PI_{90\%}(Y)$",
                                 "One realisation of\n"+r"$CI_{90\%}(Y)$"+"\n"+r"(sample size: 5)",
                                 "One realisation of\n"+r"$CI_{90\%}(Y)$"+"\n"+r"(sample size: 30)"], fontsize=fontsize, rotation=0)
        for xtick, color in zip(axes.get_xticklabels(), ["black"]+[colors[i] for i in np.arange(nb_datasets)]):
            xtick.set_color(color)
    
    
    axes.set_yticks([])
    ymin,ymax=axes.get_ylim()
    xmin,xmax=axes.get_xlim()
    axes.text(xmin-0.2, ymax+0.5, r'$y$', fontsize=fontsize,
               horizontalalignment='center', verticalalignment='center')
    
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    
    axes.set_xlim(xmin-0.2,xmax+0.2)
        
    plt.legend()
    
    #--------------- Flèche au bout des axes
    for i in np.flip(np.arange(nb_datasets)):
        xmin,xmax=np.copy(axes.get_xlim())
        ymin,ymax=np.copy(axes.get_ylim())
        axes.plot((xmax), (ymin), ls="", marker=">", ms=6, color="k", #transform=ax.get_yaxis_transform(), 
                    clip_on=False)
        axes.plot((xmin), (ymax), ls="", marker="^", ms=6, color="k", #transform=ax.get_xaxis_transform(), 
                clip_on=False)
        axes.set_xlim(xmin,xmax)
        axes.set_ylim(ymin,ymax)
    
    
    plt.show()


def display_figure4(list_confidences, size_per_dataset , relativeError_im):
    plt.figure(dpi=dpi)
    X, Y = np.meshgrid(list_confidences, size_per_dataset)
    plt.axhline(y=0.68, linestyle='dotted', color='black')
    for lev in [3,5,10,20,30]:
        CS = plt.contour(Y.T, X.T, relativeError_im, levels=[lev], colors='black')
        plt.clabel(CS, inline=True, fontsize=fontsize, fmt="{} %%".format(lev))
    plt.ylabel("Confidence\n"+r"$1-\alpha$", rotation=0, ha='center', fontsize=fontsize, labelpad=30)
    plt.xlabel("Sample size\n"+r"$M$", fontsize=fontsize)
    plt.title("Relative error (in percent)\nbetween the wrong and correct confidence intervals "+r"$CI_{1-\alpha}(Y)$"+"\nfor different configurations", fontsize=fontsize)
    
    plt.xticks([5,10,20,30,40], fontsize=fontsize)
    yticks = np.array([0.6,0.7,0.8,0.9,1])
    plt.yticks(yticks, [str(int(100*yticks[i]))+"%" for i in range(len(yticks))], fontsize=fontsize)
    plt.grid()
    plt.show()



def display_figure5(mu_X_theo, mu_Y_theo, sigma_X_theo, sigma_Y_theo, corr_theo,
                    x, Ymin, Ymax, Xmin, Xmax, X_obs,
                    probability_interval_Y, probability_interval_Y_X_noiseless):

    import matplotlib as mpl
    mpl.rcParams['hatch.linewidth'] = 0.5
    
    step  = 0.5
    alpha = 0.15
    
    ymin,ymax = np.copy(Ymin), np.copy(Ymax)
    xmin,xmax = np.copy(Xmin), np.copy(Xmax)-1
    
    
    fig, axes = plt.subplots(1,2, figsize=(10,3), sharey=True, width_ratios=(1,1.02), dpi=dpi)#, sharex=True)
    
    label_uncons_theo = r"$PI_{90\%}(Y)$"
    label_cons_theo   = r"$PI_{90\%}(Y|X=x_0)$"
    
    # Compute the linear coefficients of the linear relation Y = beta_0 + beta_1 * X + epsilon
    cov_XY_theo = corr_theo*sigma_X_theo*sigma_Y_theo  # Theoretical covariance between X and Y
    beta_1      = cov_XY_theo/sigma_X_theo**2 # Slope
    beta_0      = mu_Y_theo - beta_1*mu_X_theo # Intercept
    
    
    # Constrained
    
    colors = ["black", color_cons_noiseless]
    axes[0].plot(x, beta_0+beta_1*x, color=color_cons_noiseless, linewidth=lw_low, linestyle='solid', label=r"$y=a_0+a_1\,x$")
    axes[0].set_xticks([mu_X_theo, X_obs], labels=[r"$\mu_X$", r"$x_0$"], fontsize=fontsize)
    for ticklabel, tickcolor in zip(axes[0].get_xticklabels(), colors):
        ticklabel.set_color(tickcolor)
    plot_interval(axes[0], X_obs, probability_interval_Y_X_noiseless, color_cons_noiseless, lw_low, w_bar=w_bar, label=label_cons_theo)
    axes[0].fill_between(x, beta_0+beta_1*x-np.abs(np.diff(probability_interval_Y_X_noiseless))/2,
                            beta_0+beta_1*x+np.abs(np.diff(probability_interval_Y_X_noiseless))/2,
                            alpha=alpha, color=color_cons_noiseless)
    axes[0].set_title("(a) Constrained prediction interval\n\n")
    axes[0].legend(fontsize=fontsize, bbox_to_anchor=(0.5, -0.2), loc='upper center')
    
    j = 0
    for x_temp in [mu_X_theo, X_obs]:
        axes[0].vlines(x_temp, color=colors[j], linestyle="dotted", ymax=beta_0+beta_1*x_temp, ymin=-10000)
        axes[0].hlines(beta_0+beta_1*x_temp, color=colors[j], linestyle="dotted", xmin=-1000, xmax=x_temp)
        axes[1].hlines(beta_0+beta_1*x_temp, color=colors[j], linestyle="dotted", xmin=-1000, xmax=j)
        j += 1
        
    # Constrained VS unconstrained
    plot_interval(axes[1], 0, probability_interval_Y, "black", lw_low, w_bar=w_bar, label=label_uncons_theo)
    plot_interval(axes[1], 1, probability_interval_Y_X_noiseless, color_cons_noiseless, lw_low, w_bar=w_bar, label=label_cons_theo)
    axes[1].set_title("(b) Comparison\n\n", fontsize=fontsize)
    axes[1].set_xlim(-0.5,1.5)
    axes[1].set_xticks([0,1])
    #axes[1].set_xticklabels(["Constrained\n\n"+label_uncons_theo, "Unconstrained\n\n"+label_cons_theo], fontsize=fontsize, rotation=0)
    axes[1].set_xticklabels([label_uncons_theo+"\n\n(Unconstrained)",
                             label_cons_theo+"\n\n(Constrained)"], fontsize=fontsize, rotation=0)
    for ticklabel, tickcolor in zip(axes[1].get_xticklabels(), colors):
        ticklabel.set_color(tickcolor)
    axes[1].text(-0.5, ymax+0.5, r'$y$', fontsize=fontsize,
                   horizontalalignment='center', verticalalignment='center')
    
    
    for i in range(1):
        axes[i].set_ylim(ymin, ymax)
        axes[i].set_xlim(xmin, xmax)
        axes[i].text(xmin, ymax+0.5, r'$y$', fontsize=fontsize,
                       horizontalalignment='center', verticalalignment='center')
        axes[i].text(xmax+0.3, ymin, r'$x$', fontsize=fontsize,
                       horizontalalignment='center', verticalalignment='center')
    for i in range(2):
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].set_yticks([beta_0+beta_1*mu_X_theo, beta_0+beta_1*X_obs], labels=[r"$\mu_Y$", r"$\mu_{Y|X=x_0}$"], fontsize=fontsize)
        for ticklabel, tickcolor in zip(axes[i].get_yticklabels(), colors):
            ticklabel.set_color(tickcolor)
        
    #--------------- Flèche au bout des axes
    for i in range(2):
        xmin,xmax=np.copy(axes[i].get_xlim())
        ymin,ymax=np.copy(axes[i].get_ylim())
        axes[i].plot((xmax), (ymin), ls="", marker=">", ms=6, color="k", #transform=ax.get_yaxis_transform(), 
                    clip_on=False)
        axes[i].plot((xmin), (ymax), ls="", marker="^", ms=6, color="k", #transform=ax.get_xaxis_transform(), 
                clip_on=False)
        axes[i].set_xlim(xmin,xmax)
        axes[i].set_ylim(ymin,ymax)
    
    
    plt.show()


def display_figure6(mu_X_theo, mu_Y_theo, sigma_X_theo, sigma_Y_theo, corr_theo,
                    X_simu_per_dataset, Y_simu_per_dataset,
                    x, Ymin, Ymax, Xmin, Xmax, X_obs,
                    confidence_interval_Y_per_dataset, confidence_interval_Y_X_noiseless_per_dataset,
                    confidence_interval_Y_X_noiseless_per_dataset_per_X_obs, probability_interval_Y_X_noiseless):
    import matplotlib as mpl
    mpl.rcParams['hatch.linewidth'] = 0.5
    
    step  = 0.5
    alpha = 0.15
    
    ymin,ymax = np.copy(Ymin)-0.5, np.copy(Ymax)+1.5
    xmin,xmax = np.copy(Xmin), np.copy(Xmax)+1
    
    
    fig, axes = plt.subplots(2,2, figsize=(10,6), sharey='row', width_ratios=(1,1.02), dpi=dpi)#, sharex=True)
    
    
    list_letters = ["(a)", "(b)", "(c)", "(d)"]
    # Estimated
    colors = ["black", color_cons_noiseless]
    nb_datasets = len(confidence_interval_Y_X_noiseless_per_dataset)
    for i in range(nb_datasets):
        X_simu = X_simu_per_dataset[i]
        Y_simu = Y_simu_per_dataset[i]
        interval_nonnoisy_est        = confidence_interval_Y_X_noiseless_per_dataset[i]
        intervals_nonnoisy_est       = confidence_interval_Y_X_noiseless_per_dataset_per_X_obs[i]
        interval_unconstrained_est       = confidence_interval_Y_per_dataset[i]
        label1 = r"$PI_{90\%}(Y|X=x_0)$" 
        label3 = "One realisation of the "+r"$CI_{90\%}(Y|X=x_0)$"
        label  = "One climate model"
            
        # Constrained
        axes[i,0].scatter(X_simu, Y_simu, s=50, marker="o", facecolor='none', edgecolors=colors[0], label=label)
        axes[i,0].set_xticks([np.mean(X_simu), X_obs], labels=[r"$\hat{\mu}_X$", r"$x_0$"], fontsize=fontsize)
        for ticklabel, tickcolor in zip(axes[i,0].get_xticklabels(), colors):
                ticklabel.set_color(tickcolor)
        plot_interval(axes[i,0], X_obs, interval_nonnoisy_est, colors[1], lw_low, w_bar=w_bar, label=label3)
        #for k in range(2): axes[i,0].plot(x, intervals_nonnoisy_est[:, k], color=colors[1], linewidth=lw_low/2, linestyle='solid')
    
        axes[i,0].fill_between(x, intervals_nonnoisy_est[:,0], intervals_nonnoisy_est[:,1],
                             alpha=alpha, color=colors[1])
        if i==0: axes[i,0].set_title("(a) Constrained prediction interval\n\n")
        axes[i,0].plot(x, np.mean(intervals_nonnoisy_est, axis=1), color=colors[1],
                 linewidth=lw_high/2, linestyle='solid', label="One realisation of "+r"$y=\hat{a}_0+\hat{a}_1\,x$")
        if i==1: axes[i,0].legend(fontsize=fontsize, bbox_to_anchor=(0.5, -0.2), loc='upper center')
            
        # Pointillés
        j = -1
        for (x_temp,y_temp) in [(np.mean(X_simu),np.mean(interval_unconstrained_est)), (X_obs,np.mean(interval_nonnoisy_est))]:
            axes[i,0].vlines(x_temp, color=colors[j+1], linestyle="dotted", ymax=y_temp, ymin=-10000)
            axes[i,0].hlines(y_temp, color=colors[j+1], linestyle="dotted", xmin=-1000, xmax=x_temp)
            axes[i,1].hlines(y_temp, color=colors[j+1], linestyle="dotted", xmin=-1000, xmax=j)
            j += 1
    
        
        # Constrained VS unconstrained
        plot_interval(axes[i,1], -1, interval_unconstrained_est, colors[0], lw_low, w_bar=w_bar, label=label3)
        plot_interval(axes[i,1], 0, interval_nonnoisy_est, colors[1], lw_low, w_bar=w_bar, label=label3)
        plot_interval(axes[i,1], 1, probability_interval_Y_X_noiseless, colors[1], lw_low, w_bar=w_bar, label=label1)
        if i==0: axes[i,1].set_title("(b) Comparison\n\n", fontsize=fontsize)
        axes[i,1].set_xlim(-1.5,1.5)
        if i==1:
            axes[i,1].set_xticks([1, 0, -1])
            axes[i,1].set_xticklabels([r"$PI_{90\%}(Y|X=x_0)$",
                                       "Realisation\nof the\n"+r"$CI_{90\%}(Y|X=x_0)$",
                                       "Realisation\nof the\n"+r"$CI_{90\%}(Y)$"],
                                      fontsize=fontsize, rotation=0)
            for ticklabel, tickcolor in zip(axes[i,1].get_xticklabels(), [color_cons_noiseless, color_cons_noiseless, "black"]):
                ticklabel.set_color(tickcolor)
            ticks = axes[i,1].get_xticklabels()
            #ticks[-1].set_rotation(10)
        else: axes[i,1].set_xticks([])
        axes[i,1].text(-1.5, ymax+0.7, r'$y$', fontsize=fontsize,
                       horizontalalignment='center', verticalalignment='center')
    
    
        axes[i,0].set_ylim(ymin, ymax)
        axes[i,0].set_xlim(xmin, xmax)
        axes[i,0].text(xmin, ymax+0.7, r'$y$', fontsize=fontsize,
                       horizontalalignment='center', verticalalignment='center')
        axes[i,0].text(xmax+0.4, ymin, r'$x$', fontsize=fontsize,
                       horizontalalignment='center', verticalalignment='center')
        
        for j in range(2):
            axes[i,j].spines['top'].set_visible(False)
            axes[i,j].spines['right'].set_visible(False)
            axes[i,j].set_yticks([np.mean(interval_unconstrained_est), np.mean(interval_nonnoisy_est)],
                                 labels=[r"$\hat{\mu}_Y$", r"$\hat{\mu}_{Y|X=x_0}$"], fontsize=fontsize)
            for ticklabel, tickcolor in zip(axes[i,j].get_yticklabels(), colors):
                ticklabel.set_color(tickcolor)
    
        
        axes[i,0].set_ylabel("Sample\nof size {}".format(len(X_simu)), 
                             rotation=0, horizontalalignment='right', fontsize=fontsize)
    
    
       
    #--------------- Flèche au bout des axes
    for i in range(nb_datasets):
        for j in range(2):
            xmin,xmax=np.copy(axes[i,j].get_xlim())
            ymin,ymax=np.copy(axes[i,j].get_ylim())
            axes[i,j].plot((xmax), (ymin), ls="", marker=">", ms=6, color="k", #transform=ax.get_yaxis_transform(), 
                        clip_on=False)
            axes[i,j].plot((xmin), (ymax), ls="", marker="^", ms=6, color="k", #transform=ax.get_xaxis_transform(), 
                    clip_on=False)
            axes[i,j].set_xlim(xmin,xmax)
            axes[i,j].set_ylim(ymin,ymax)
            
            
    
    plt.show()



def display_figure7(list_obs_stand, size_per_dataset, relativeError_im_con):
    plt.figure(dpi=dpi)
    X, Y = np.meshgrid(list_obs_stand, size_per_dataset)
    for lev in [3,5,10,20,30]:
        CS = plt.contour(Y.T, X.T, relativeError_im_con, levels=[lev], colors='black')
        plt.clabel(CS, inline=True, fontsize=fontsize, fmt="{} %%".format(lev))
    plt.ylabel("Standardized\nobservation\n"+r"$\frac{|x_0-\hat{\mu}_X|}{\hat{\sigma}_X}$", rotation=0, ha='center', fontsize=fontsize, labelpad=40)
    plt.xlabel("Sample size\n"+r"$M$", fontsize=fontsize)
    plt.title("Relative error (in percent)\nbetween the wrong and correct confidence intervals "+r"$CI_{68\%}(Y|X=x_0)$", fontsize=fontsize)
    
    plt.xticks([5,10,20,30,40], fontsize=fontsize)
    yticks = np.array([0,0.5,1,1.5,2])
    plt.yticks(yticks, fontsize=fontsize)#, [str(int(100*yticks[i]))+"%" for i in range(len(yticks))], fontsize=fontsize)
    plt.grid()
    plt.show()





def display_figure8(mu_X_theo, mu_Y_theo, sigma_X_theo, sigma_Y_theo, corr_theo, sigma_N,
                    x, Ymin, Ymax, Xmin, Xmax, X_obs,
                    probability_interval_Y, probability_interval_Y_X_noiseless, probability_interval_Y_X_noisy):

    # Compute the linear coefficients of the linear relation Y = beta_0 + beta_1 * X + epsilon
    cov_XY_theo = corr_theo*sigma_X_theo*sigma_Y_theo  # Theoretical covariance between X and Y
    beta_1      = cov_XY_theo/sigma_X_theo**2 # Slope
    beta_0      = mu_Y_theo - beta_1*mu_X_theo # Intercept
    
    # Compute the linear coefficients of the linear relation Y = beta_0_noisy + beta_1_noisy * X_noisy + epsilon_noisy
    beta_1_noisy = cov_XY_theo/(sigma_X_theo**2+sigma_N**2)
    beta_0_noisy = mu_Y_theo - beta_1_noisy*mu_X_theo
    
    import matplotlib as mpl
    mpl.rcParams['hatch.linewidth'] = 0.5
    
    step  = 0.5
    alpha = 0.15
    
    ymin,ymax = np.copy(Ymin), np.copy(Ymax)
    xmin,xmax = np.copy(Xmin), np.copy(Xmax)-1
    
    fig, axes = plt.subplots(1,2, figsize=(10,3), sharey=True, width_ratios=(1,1.02), dpi=dpi)#, sharex=True)
    
    label_uncons_theo = r"$PI_{90\%}(Y|X^N=x_0^N)$"# r"$[b_0+b_1\,x_0^N \pm z\, \sigma_{\varepsilon^N}]$"
    label_cons_theo   = r"$PI_{90\%}(Y)$" # r"$[\mu_Y \pm z\, \sigma_{Y}]$"
    
    
    colors = ["black", color_cons_noiseless, color_cons_noisy]
    
    # Constrained by non-noisy
    axes[0].plot(x, beta_0+beta_1*x, color=colors[1], linewidth=lw_low, linestyle='solid', label=r"$y=a_0+a_1\,x$")
    axes[0].set_xticks([mu_X_theo, X_obs], labels=[r"$\mu_X$", r"$x_0$"], fontsize=fontsize)
    for ticklabel, tickcolor in zip(axes[0].get_xticklabels(), ["black", color_cons_noisy]):
                ticklabel.set_color(tickcolor)
    axes[0].fill_between(x, beta_0+beta_1*x-np.abs(np.diff(probability_interval_Y_X_noiseless))/2,
                            beta_0+beta_1*x+np.abs(np.diff(probability_interval_Y_X_noiseless))/2,
                            alpha=alpha, color=colors[1])
    
    # Constrained by noisy
    axes[0].plot(x, beta_0_noisy+beta_1_noisy*x, color=colors[2], linewidth=lw_low, linestyle='solid', label=r"$y=b_0+b_1\,x$")
    axes[0].set_xticks([mu_X_theo, X_obs], labels=[r"$\mu_X$", r"$x_0^N$"], fontsize=fontsize)
    plot_interval(axes[0], X_obs, probability_interval_Y_X_noisy, colors[2], lw_low, w_bar=w_bar, label=label_uncons_theo)
    axes[0].fill_between(x, beta_0_noisy+beta_1_noisy*x-np.abs(np.diff(probability_interval_Y_X_noisy))/2,
                            beta_0_noisy+beta_1_noisy*x+np.abs(np.diff(probability_interval_Y_X_noisy))/2,
                            alpha=alpha, color=colors[2])
    axes[0].set_title("(a) Constrained prediction interval\n\n")
    axes[0].legend(fontsize=fontsize, bbox_to_anchor=(0.5, -0.2), loc='upper center')
    
    
    # Pointillés
    j = -1
    colors = ["black", color_cons_noisy]
    for (x_temp,y_temp) in [(mu_X_theo,np.mean(probability_interval_Y)), (X_obs,np.mean(probability_interval_Y_X_noisy))]:
        axes[0].vlines(x_temp, color=colors[j+1], linestyle="dotted", ymax=y_temp, ymin=-10000)
        axes[0].hlines(y_temp, color=colors[j+1], linestyle="dotted", xmin=-1000, xmax=x_temp)
        axes[1].hlines(y_temp, color=colors[j+1], linestyle="dotted", xmin=-1000, xmax=j)
        j += 1
    colors = ["black", color_cons_noiseless, color_cons_noisy]
    
    
    
    # Constrained non-noisy VS constrained noisy VS unconstrained
    plot_interval(axes[1], -1, probability_interval_Y, colors[0], lw_low, w_bar=w_bar, label=label_uncons_theo)
    plot_interval(axes[1], 1, probability_interval_Y_X_noiseless, colors[1], lw_low, w_bar=w_bar, label=label_cons_theo)
    plot_interval(axes[1], 0, probability_interval_Y_X_noisy, colors[2], lw_low, w_bar=w_bar, label=label_cons_theo)
    axes[1].set_title("(b) Comparison\n\n", fontsize=fontsize)
    axes[1].set_xlim(-1.5,1.5)
    axes[1].set_xticks([-1,1,0])
    #axes[1].set_xticklabels(["Constrained\n\n"+label_uncons_theo, "Unconstrained\n\n"+label_cons_theo], fontsize=fontsize, rotation=0)
    axes[1].set_xticklabels([r"$PI_{90\%}(Y)$",
                             r"$PI_{90\%}(Y|X=x_0)$",
                             r"$PI_{90\%}(Y|X^N=x_0^N)$"], fontsize=fontsize, rotation=-15)
    for ticklabel, tickcolor in zip(axes[1].get_xticklabels(), ["black", color_cons_noiseless, color_cons_noisy]):
        ticklabel.set_color(tickcolor)
    
    
    axes[1].text(-1.5, ymax+0.5, r'$y$', fontsize=fontsize,
                   horizontalalignment='center', verticalalignment='center')
    
    
    for i in range(1):
        axes[i].set_ylim(ymin, ymax)
        axes[i].set_xlim(xmin, xmax)
        axes[i].text(xmin, ymax+0.5, r'$y$', fontsize=fontsize,
                       horizontalalignment='center', verticalalignment='center')
        axes[i].text(xmax+0.3, ymin, r'$x$', fontsize=fontsize,
                       horizontalalignment='center', verticalalignment='center')
    for i in range(2):
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].set_yticks([np.mean(probability_interval_Y), np.mean(probability_interval_Y_X_noisy)],
                                 labels=[r"$\mu_Y$", r"$\mu_{Y|X^N=x_0^N}$"], fontsize=fontsize)
        for ticklabel, tickcolor in zip(axes[i].get_yticklabels(), ["black", color_cons_noisy]):
            ticklabel.set_color(tickcolor)
    
        
    #--------------- Flèche au bout des axes
    for i in range(2):
        xmin,xmax=np.copy(axes[i].get_xlim())
        ymin,ymax=np.copy(axes[i].get_ylim())
        axes[i].plot((xmax), (ymin), ls="", marker=">", ms=6, color="k", #transform=ax.get_yaxis_transform(), 
                    clip_on=False)
        axes[i].plot((xmin), (ymax), ls="", marker="^", ms=6, color="k", #transform=ax.get_xaxis_transform(), 
                clip_on=False)
        axes[i].set_xlim(xmin,xmax)
        axes[i].set_ylim(ymin,ymax)
    
    
    plt.show()




def display_figure9(mu_X_theo, mu_Y_theo, sigma_X_theo, sigma_Y_theo, corr_theo, sigma_N,
                    x, Ymin, Ymax, Xmin, Xmax, X_obs,
                    X_simu_per_dataset, Y_simu_per_dataset,
                    confidence_interval_Y_per_dataset, confidence_interval_Y_X_noiseless_per_dataset, confidence_interval_Y_X_noisy_per_dataset,
                    confidence_interval_Y_X_noiseless_per_dataset_per_X_obs, confidence_interval_Y_X_noisy_per_dataset_per_X_obs):
    import matplotlib as mpl
    mpl.rcParams['hatch.linewidth'] = 0.5
    
    step  = 0.5
    alpha = 0.15
    
    ymin,ymax = np.copy(Ymin)-0.5, np.copy(Ymax)+1.5
    xmin,xmax = np.copy(Xmin), np.copy(Xmax)+1
    

    fig, axes = plt.subplots(2,2, figsize=(10,6), sharey='row', width_ratios=(1,1.02), dpi=dpi)#, sharex=True)
    
    
    colors = ["black", color_cons_noiseless, color_cons_noisy]
    list_letters = ["a", "b", "c", "d"]
    # Estimated
    nb_datasets = len(confidence_interval_Y_per_dataset)
    for i in range(nb_datasets):
        X_simu = X_simu_per_dataset[i]
        Y_simu = Y_simu_per_dataset[i]
        interval_nonnoisy_est       = confidence_interval_Y_X_noiseless_per_dataset[i]
        intervals_nonnoisy_est      = confidence_interval_Y_X_noiseless_per_dataset_per_X_obs[i]
        interval_noisy_est          = confidence_interval_Y_X_noisy_per_dataset[i]
        intervals_noisy_est         = confidence_interval_Y_X_noisy_per_dataset_per_X_obs[i]
        interval_unconstrained_est  = confidence_interval_Y_per_dataset[i]
    
        label1 = r"$[a_0+a_1\,x_0 \pm z\, \sigma_{\varepsilon}]$"
        label2 = r"$[\hat{a}_0+\hat{a}_1\,x_0 \pm z \, \hat{\sigma}_{\varepsilon}]$"
        label3 = r"$CI_{90\%}(Y|X^N=x_0^N)$"
            
        label  = "One climate model"
            
        # Constrained
        axes[i,0].scatter(X_simu, Y_simu, s=50, marker="o", facecolor='none', edgecolors=colors[0], label=label)
        axes[i,0].set_xticks([np.mean(X_simu), X_obs], labels=[r"$\hat{\mu}_X$", r"$x_0^N$"], fontsize=fontsize)
        for ticklabel, tickcolor in zip(axes[i,0].get_xticklabels(), ["black", color_cons_noisy]):
                ticklabel.set_color(tickcolor)
        plot_interval(axes[i,0], X_obs, interval_noisy_est, colors[2], lw_low, w_bar=w_bar, label=label3)
        axes[i,0].fill_between(x, intervals_nonnoisy_est[:,0], intervals_nonnoisy_est[:,1],
                             alpha=alpha, color=colors[1])
        axes[i,0].fill_between(x, intervals_noisy_est[:,0], intervals_noisy_est[:,1],
                             alpha=alpha, color=colors[2])
        if i==0: axes[i,0].set_title("(a) Constrained prediction interval\n\n")
        axes[i,0].plot(x, np.mean(intervals_nonnoisy_est, axis=1), color=colors[1],
                 linewidth=lw_high/2, linestyle='solid', label=r"$y=\hat{a}_0+\hat{a}_1\,x$")
        axes[i,0].plot(x, np.mean(intervals_noisy_est, axis=1), color=colors[2],
                 linewidth=lw_high/2, linestyle='solid', label=r"$y=\hat{b}_0+\hat{b}_1\,x$")
        if i==1: axes[i,0].legend(fontsize=fontsize, bbox_to_anchor=(0.5, -0.2), loc='upper center', ncols=2)
        
        # Pointillés
        j = -1
        colors = ["black", color_cons_noisy]
        for (x_temp,y_temp) in [(np.mean(X_simu),np.mean(interval_unconstrained_est)), (X_obs,np.mean(interval_noisy_est))]:
            axes[i,0].vlines(x_temp, color=colors[j+1], linestyle="dotted", ymax=y_temp, ymin=-10000)
            axes[i,0].hlines(y_temp, color=colors[j+1], linestyle="dotted", xmin=-1000, xmax=x_temp)
            axes[i,1].hlines(y_temp, color=colors[j+1], linestyle="dotted", xmin=-1000, xmax=j)
            j += 1
        colors = ["black", color_cons_noiseless, color_cons_noisy]
    
        
        # Constrained VS unconstrained
        plot_interval(axes[i,1], -1, interval_unconstrained_est, colors[0], lw_low, w_bar=w_bar, label=label3)
        plot_interval(axes[i,1], 1, interval_nonnoisy_est, colors[1], lw_low, w_bar=w_bar, label=label3)
        plot_interval(axes[i,1], 0, interval_noisy_est, colors[2], lw_low, w_bar=w_bar, label=label3)
        if i==0: axes[i,1].set_title("(b) Comparison\n\n", fontsize=fontsize)
        axes[i,1].set_xlim(-1.5,1.5)
        if i==1:
            axes[i,1].set_xticks([-1,1,0])
            axes[i,1].set_xticklabels(["Realisation\nof\n"+r"$CI_{90\%}(Y)$",
                                     "Realisation\nof\n"+r"$CI_{90\%}(Y|X=x_0)$",
                                     "Realisation\nof\n"+r"$CI_{90\%}(Y|X^N=x_0^N)$"], fontsize=fontsize, rotation=-12)
            for ticklabel, tickcolor in zip(axes[i,1].get_xticklabels(), ["black", color_cons_noiseless, color_cons_noisy]):
                ticklabel.set_color(tickcolor)
            ticks = axes[i,1].get_xticklabels()
            #ticks[-1].set_rotation(10)
        else: axes[i,1].set_xticks([])
        axes[i,1].text(-1.5, ymax+0.7, r'$y$', fontsize=fontsize,
                       horizontalalignment='center', verticalalignment='center')
    
    
        axes[i,0].set_ylim(ymin, ymax)
        axes[i,0].set_xlim(xmin, xmax)
        axes[i,0].text(xmin, ymax+0.7, r'$y$', fontsize=fontsize,
                       horizontalalignment='center', verticalalignment='center')
        axes[i,0].text(xmax+0.4, ymin, r'$x$', fontsize=fontsize,
                       horizontalalignment='center', verticalalignment='center')
        
        for j in range(2):
            axes[i,j].spines['top'].set_visible(False)
            axes[i,j].spines['right'].set_visible(False)
            axes[i,j].set_yticks([np.mean(interval_unconstrained_est), np.mean(interval_noisy_est)],
                                 labels=[r"$\hat{\mu}_Y$", r"$\hat{\mu}_{Y|X^N=x_0^N}$"], fontsize=fontsize)
            for ticklabel, tickcolor in zip(axes[i,j].get_yticklabels(), ["black", color_cons_noisy]):
                ticklabel.set_color(tickcolor)
        
        axes[i,0].set_ylabel("Sample\nof size {}".format(len(X_simu)), color="black",
                             rotation=0, horizontalalignment='right', fontsize=fontsize)
    
    plt.show()


    
def display_figure10(X_simu_per_dataset, Y_simu_per_dataset, x, X_obs, sigma_N,
                    confidence_interval_Y_per_dataset, confidence_interval_Y_X_noiseless_per_dataset, confidence_interval_Y_X_noisy_per_dataset,
                    confidence_interval_Y_X_noiseless_per_dataset_per_X_obs, confidence_interval_Y_X_noisy_per_dataset_per_X_obs):
    linestyle3_ = 'solid'
    linewidth = 2
    s = 20
        
    i = 1
    interval_obs0 = confidence_interval_Y_per_dataset[i]
    interval_obs1 = confidence_interval_Y_X_noiseless_per_dataset[i]
    interval_obs2 = confidence_interval_Y_X_noisy_per_dataset[i]

    intervals_obs1 = confidence_interval_Y_X_noiseless_per_dataset_per_X_obs[i]
    intervals_obs2 = confidence_interval_Y_X_noisy_per_dataset_per_X_obs[i]

    interval_noisy_est = interval_obs2
    interval_nonnoisy_est = interval_obs1
    
    X_simu, Y_simu = X_simu_per_dataset[i], Y_simu_per_dataset[i]

    x_space = 0.5
    before_interv = 0.1
    
    xmargin = 1
    ymargin = 1
    xmin, xmax = X_simu.min()-xmargin, X_simu.max()+xmargin
    ymin, ymax = Y_simu.min()-ymargin, Y_simu.max()+ymargin
    M = len(X_simu)
    
    color1='black'
    color2='tab:red'
    color3='tab:green'
    marker = 'o'
    facecolor = 'none'
    linestyle2 = 'solid'
    
    fig, ax1 = plt.subplots(1,1, figsize=(10,6), dpi=dpi)
    
    #--------- Unconstrained
    plot_interval(ax1, xmax+3*x_space, interval_obs0, color1, lw, w_bar=0.4)
    
    #--------- Constrained (non-noisy) intervals
    beta_1_est = np.cov(X_simu, Y_simu)[0,1]/(np.var(X_simu))
    beta_0_est = np.mean(Y_simu) - beta_1_est*np.mean(X_simu)
    y          = beta_0_est + beta_1_est*x
    line4 = ax1.scatter(X_simu, Y_simu, s=4*s, marker=marker, color=color1, facecolor=facecolor,
                edgecolors=color1, linewidths=2, label="One climate model")
          
    line2, = ax1.plot(x,y, color=color2, linestyle=linestyle2, linewidth=linewidth)

    ax1.fill_between(x, intervals_obs1[:,0], intervals_obs1[:,1], color=color2, alpha=0.2)
    plot_interval(ax1, xmax+1*x_space, interval_obs1, color2, lw, w_bar=0.4)
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)

    beta_1_est = np.cov(X_simu, Y_simu)[0,1]/(np.var(X_simu)+sigma_N**2)
    beta_0_est = np.mean(Y_simu) - beta_1_est*np.mean(X_simu)
    y          = beta_0_est + beta_1_est*x

    line3, = ax1.plot(x,y, color=color3, linestyle=linestyle3_, linewidth=linewidth)
    ax1.fill_between(x, intervals_obs2[:,0], intervals_obs2[:,1], color=color3, alpha=0.2)
    
    plot_interval(ax1, xmax+2*x_space, interval_obs2, color3, lw, w_bar=0.4)

    j = 0
    list_colors = [color3, color2] # color1
    list_colors_bis = ["gray", "gray", "gray"]
    mmax_ = -1000
    mmin_ = 1000
    
    
    
    alpha_behind = 0.35
    linestyle_main = 'dotted'
    linestyle_second = 'dotted'
    linewidth_main = 0.8 #0.8
    linewidth_second = 1.2 #1.6
    alpha_lines     = 0.5
    
    ax1.hlines(np.mean(Y_simu), linestyle=linestyle_main, xmin=np.mean(X_simu), xmax=xmax+3*x_space, colors="black", linewidth=linewidth_main, alpha=alpha_lines)
    ax1.vlines(np.mean(X_simu), linestyle=linestyle_main, ymin=-1000, ymax=np.mean(Y_simu), colors="black", linewidth=1.2*linewidth_main, alpha=alpha_lines)
    
    for (x_temp,y_temp, min_, max_, xmax_) in [
        (X_obs,np.mean(interval_noisy_est),np.min(interval_noisy_est),np.max(interval_noisy_est), xmax+2*x_space),
        (X_obs,np.mean(interval_nonnoisy_est),np.min(interval_nonnoisy_est),np.max(interval_nonnoisy_est), xmax+1*x_space)]:
        
        if j==0:
            ax1.hlines(min_, linestyle=linestyle_second, xmin=x_temp, xmax=xmax_-before_interv, colors=list_colors[j], linewidth=linewidth_second, alpha=alpha_lines, zorder=1000) # (0,(1,5))
            if False:
                ax1.hlines(y_temp, linestyle=linestyle_main, xmin=x_temp, xmax=xmax_-before_interv, colors=list_colors[j], linewidth=linewidth_main, zorder=1000, alpha=alpha_lines)
                ax1.hlines(max_, linestyle=linestyle_second, xmin=xmax+1*x_space+0.1, xmax=xmax_-before_interv, colors=list_colors[j], linewidth=linewidth_second, alpha=alpha_lines, zorder=1000)
            else:
                ax1.hlines(y_temp, linestyle=linestyle_main, xmin=x_temp, xmax=xmax_-before_interv, colors=list_colors[j], linewidth=linewidth_main, zorder=1000, alpha=alpha_lines)
                ax1.hlines(max_, linestyle=linestyle_second, xmin=x_temp, xmax=xmax_-before_interv, colors=list_colors[j], linewidth=linewidth_second, alpha=alpha_lines, zorder=1000)
        else:
            ax1.hlines(y_temp, linestyle=linestyle_main, xmin=x_temp, xmax=xmax_-before_interv, colors=list_colors[j], linewidth=linewidth_main, alpha=alpha_lines, zorder=1000)
            ax1.hlines(min_, linestyle=linestyle_second, xmin=x_temp, xmax=xmax_-before_interv, colors=list_colors[j], linewidth=linewidth_second, alpha=alpha_lines, zorder=1000) # (0,(1,5))
            ax1.hlines(max_, linestyle=linestyle_second, xmin=x_temp, xmax=xmax_-before_interv, colors=list_colors[j], linewidth=linewidth_second, alpha=alpha_lines, zorder=1000)
    
        array_x_temp = np.linspace(x_temp, xmax_)
        if j==0:
            ax1.fill_between(array_x_temp, min_, max_, color="gray", alpha=0.01)
            j = j
        elif j==1:
            ax1.fill_between(array_x_temp, np.max(interval_noisy_est)+0.02, max_, color="gray", alpha=0.01)
            j=j
        j += 1
        
        if min_<mmin_: mmin_=np.copy(min_)
        if max_>mmax_: mmax_=np.copy(max_)
            
    
    ax1.vlines(X_obs, color="gray", linestyle="solid", ymax=mmax_, ymin=-1000, linewidth=3*linewidth_main, zorder=1000) #linestyle_main
    
    
    #--------------- Axes et titres
    ax1.set_xticks([X_obs, np.mean(X_simu)])
    ax1.set_yticks([])
    ax1.set_xticklabels(["observation", "multi-model mean"], fontsize=fontsize, color="gray")
    ax1.set_ylabel("Projected\nvariable", fontsize=fontsize, rotation=0, ha="right")
    ax1.set_xlabel("Observable variable", fontsize=fontsize, rotation=0)
    ax1.text(-0.02,1.0, r"$Y$",
         ha='center', va='center',
         transform = ax1.transAxes, fontsize=fontsize)
    ax1.text(0.8,-0.02, r"$X$",
         ha='center', va='center',
         transform = ax1.transAxes, fontsize=fontsize)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    
        
    line1, = plt.plot(x+1000, y, color=color1)
    fig.legend([line1, line2, line3, line4],
               ["unconstrained",
                "constrained by a\nnoiseless observation of X",
                "constrained by a\nnoisy observation of X", "One climate model"
                ],
                loc='center left', fontsize=fontsize,
               title="Confidence intervals of Y:", title_fontsize=fontsize,
               alignment='left', bbox_to_anchor=(0.11, 0.78))
    
    
    
    #--------------- Flèche au bout des axes
    xmin,xmax=np.copy(ax1.get_xlim())
    ymin,ymax=np.copy(ax1.get_ylim())
    ax1.axhline(ymin, xmax=0.78, clip_on=True, color="black") 
    ax1.plot((xmax), (ymin), ls="", marker=">", ms=6, color="k", 
                clip_on=False)
    ax1.plot((xmin), (ymax), ls="", marker="^", ms=6, color="k", 
            clip_on=False)
    ax1.set_xlim(xmin,xmax)
    
    ax1.set_xlim(xmin,xmax+4*x_space)
    
    plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.02, 
                        hspace=0)
    plt.show()
    
    


    

    
