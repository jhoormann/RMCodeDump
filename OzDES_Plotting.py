# ---------------------------------------------------------- #
# -------------------- OzDES_Plotting.py ------------------- #
# --------- https://github.com/jhoormann/RMCodeDump -------- #
# ---------------------------------------------------------- #
# This is a dump of all the functions I have written to make #
# plots look nice.                                           #
# ---------------------------------------------------------- #

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# -------------------------------------------------- #
# ------------------ plot_fonts -------------------- #
# -------------------------------------------------- #
# Function to define the font used for plotting.     #
# -------------------------------------------------- #
def plot_fonts(size, color='black', weight='normal', align='bottom'):
    font = {'size': size, 'color': color, 'weight': weight, 'verticalalignment': align}
    return font


# -------------------------------------------------- #
# ------------------ plot_ticks ---------------------#
# -------------------------------------------------- #
# Function to change the plot tick size.             #
# -------------------------------------------------- #
def plot_ticks(ax, size):
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(size)
    return


# -------------------------------------------------- #
# ----------------- plot_share_x --------------------#
# -------------------------------------------------- #
# Define figure and axis variables for plot which    #
# shares the x axis for a specified number of plots. #
# -------------------------------------------------- #
def plot_share_x(number, title, xlabel, ylabel, xlim=(0, 0), ylim=(0, 0), asize=22, tsize=22, xdim=10,
                 ydim=10, xtick=5, ytick=5):
    fig, ax_array = plt.subplots(number, sharex=True)
    if number == 1:
        ax_array = [ax_array]
    fig = plt.gcf()
    fig.set_size_inches(xdim, ydim, forward=True)
    fig.subplots_adjust(hspace=0)

    title_font = plot_fonts(tsize, align='bottom')
    x_axis_font = plot_fonts(asize, align='top')
    y_axis_font = plot_fonts(asize, align='bottom')

    for [i, ax] in enumerate(ax_array):
        plot_ticks(ax, asize)
        ax.set_ylabel(ylabel[i], **y_axis_font)
        ax.yaxis.set_major_locator(MaxNLocator(prune='upper'))
        ax.tick_params(axis='y', pad=15)
        if ylim != (0, 0) and ylim[0] < ylim[1]:
            ax.set_ylim(ylim[i])
        if i == 0:
            ax.set_title(title, **title_font)
            ax.tick_params(axis='x', pad=15)
            if xlim != (0, 0) and xlim[0] < xlim[1]:
                ax.set_xlim(xlim)
        if i == number - 1:
            ax.set_xlabel(xlabel, **x_axis_font)
        if ytick is not None:
            ax.yaxis.major.locator.set_params(nbins=ytick)
        if xtick is not None:
            ax.xaxis.major.locator.set_params(nbins=xtick)

    return fig, ax_array


# -------------------------------------------------- #
# ----------------- makeFigSingle ------------------ #
# -------------------------------------------------- #
# -------------------------------------------------- #
# A function that defines a figure with legible axis #
# labels.                                            #
# -------------------------------------------------- #
font = {'size': '20', 'color': 'black', 'weight': 'normal'}


def makeFigSingle(title, xlabel, ylabel, xlim=[0, 0], ylim=[0, 0]):
    fig = plt.gcf()
    fig.set_size_inches(10, 10, forward=True)

    ax = fig.add_subplot(111)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(20)

    ax.set_ylabel(ylabel, **font)
    if ylim != [0, 0] and ylim[0] < ylim[1]:
        ax.set_ylim(ylim)

    ax.set_xlabel(xlabel, **font)
    if xlim != [0, 0] and xlim[0] < xlim[1]:
        ax.set_xlim(xlim)

    ax.set_title(title, **font)

    return fig, ax


# -------------------------------------------------- #
# ----------------- makeFigDouble ------------------ #
# -------------------------------------------------- #
# -------------------------------------------------- #
# A function that defines a figure and axes with two #
# panels that shares an x axis and has legible axis  #
# labels.                                            #
# -------------------------------------------------- #
def makeFigDouble(title, xlabel, ylabel1, ylabel2, xlim=[0, 0], ylim1=[0, 0], ylim2=[0, 0]):

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig = plt.gcf()
    fig.set_size_inches(10, 10, forward=True)
    fig.subplots_adjust(hspace=0)

    for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
        label.set_fontsize(20)
    for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
        label.set_fontsize(20)

    ax1.set_ylabel(ylabel1, **font)
    if ylim1 != [0, 0] and ylim1[0] < ylim1[1]:
        ax1.set_ylim(ylim1)

    ax2.set_ylabel(ylabel2, **font)
    if ylim2 != [0, 0] and ylim2[0] < ylim2[1]:
        ax2.set_ylim(ylim2)

    ax2.set_xlabel(xlabel, **font)
    if xlim != [0, 0] and xlim[0] < xlim[1]:
        ax2.set_xlim(xlim)

    ax1.set_title(title, **font)

    return fig, ax1, ax2


title_font = {'size': '22', 'color': 'black', 'weight': 'normal', 'verticalalignment': 'bottom'}
axis_font = {'size': '22'}

def makeFigTriple(title, xlabel, ylabel1, ylabel2, ylabel3, xlim=[0, 0], ylim1=[0, 0], ylim2=[0, 0], ylim3=[0, 0]):
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    fig = plt.gcf()
    fig.set_size_inches(22, 15, forward=True)
    fig.subplots_adjust(hspace=0)

    for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
        label.set_fontsize(25)
    for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
        label.set_fontsize(25)
    for label in (ax3.get_xticklabels() + ax3.get_yticklabels()):
        label.set_fontsize(25)

    ax1.set_ylabel(ylabel1, **axis_font)
    if ylim1 != [0, 0] and ylim1[0] < ylim1[1]:
        ax1.set_ylim(ylim1)

    ax2.set_ylabel(ylabel2, **axis_font)
    if ylim2 != [0, 0] and ylim2[0] < ylim2[1]:
        ax2.set_ylim(ylim2)

    ax3.set_ylabel(ylabel3, **axis_font)
    if ylim3 != [0, 0] and ylim3[0] < ylim3[1]:
        ax3.set_ylim(ylim3)

    ax3.set_xlabel(xlabel, **axis_font)
    if xlim != [0, 0] and xlim[0] < xlim[1]:
        ax3.set_xlim(xlim)

    ax1.set_title(title, **title_font)

    return fig, ax1, ax2, ax3


def makeFigQuad(title, xlabel, ylabel1, ylabel2, ylabel3, ylabel4, xlim=[0, 0], ylim1=[0, 0], ylim2=[0, 0],
                ylim3=[0, 0], ylim4=[0, 0]):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
    fig = plt.gcf()
    fig.set_size_inches(15, 15, forward=True)
    fig.subplots_adjust(hspace=0)

    for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
        label.set_fontsize(25)
    for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
        label.set_fontsize(25)
    for label in (ax3.get_xticklabels() + ax3.get_yticklabels()):
        label.set_fontsize(25)
    for label in (ax4.get_xticklabels() + ax4.get_yticklabels()):
        label.set_fontsize(25)

    ax1.set_ylabel(ylabel1, **axis_font)
    if ylim1 != [0, 0] and ylim1[0] < ylim1[1]:
        ax1.set_ylim(ylim1)

    ax2.set_ylabel(ylabel2, **axis_font)
    if ylim2 != [0, 0] and ylim2[0] < ylim2[1]:
        ax2.set_ylim(ylim2)

    ax3.set_ylabel(ylabel3, **axis_font)
    if ylim3 != [0, 0] and ylim3[0] < ylim3[1]:
        ax3.set_ylim(ylim3)

    ax4.set_ylabel(ylabel4, **axis_font)
    if ylim4 != [0, 0] and ylim4[0] < ylim4[1]:
        ax4.set_ylim(ylim4)

    ax4.set_xlabel(xlabel, **axis_font)
    if xlim != [0, 0] and xlim[0] < xlim[1]:
        ax4.set_xlim(xlim)

    ax1.set_title(title, **title_font)

    return fig, ax1, ax2, ax3, ax4


def makeFigQuint(title, xlabel, ylabel1, ylabel2, ylabel3, ylabel4, ylabel5, xlim=[0, 0], ylim1=[0, 0], ylim2=[0, 0],
                ylim3=[0, 0], ylim4=[0, 0], ylim5=[0, 0]):
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True)
    fig = plt.gcf()
    fig.set_size_inches(10, 10, forward=True)
    fig.subplots_adjust(hspace=0)

    for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
        label.set_fontsize(25)
    for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
        label.set_fontsize(25)
    for label in (ax3.get_xticklabels() + ax3.get_yticklabels()):
        label.set_fontsize(25)
    for label in (ax4.get_xticklabels() + ax4.get_yticklabels()):
        label.set_fontsize(25)
    for label in (ax5.get_xticklabels() + ax5.get_yticklabels()):
        label.set_fontsize(25)

    ax1.set_ylabel(ylabel1, **axis_font)
    if ylim1 != [0, 0] and ylim1[0] < ylim1[1]:
        ax1.set_ylim(ylim1)

    ax2.set_ylabel(ylabel2, **axis_font)
    if ylim2 != [0, 0] and ylim2[0] < ylim2[1]:
        ax2.set_ylim(ylim2)

    ax3.set_ylabel(ylabel3, **axis_font)
    if ylim3 != [0, 0] and ylim3[0] < ylim3[1]:
        ax3.set_ylim(ylim3)

    ax4.set_ylabel(ylabel4, **axis_font)
    if ylim4 != [0, 0] and ylim4[0] < ylim4[1]:
        ax4.set_ylim(ylim4)

    ax5.set_ylabel(ylabel5, **axis_font)
    if ylim5 != [0, 0] and ylim5[0] < ylim5[1]:
        ax5.set_ylim(ylim5)

    ax5.set_xlabel(xlabel, **axis_font)
    if xlim != [0, 0] and xlim[0] < xlim[1]:
        ax5.set_xlim(xlim)

    ax1.set_title(title, **title_font)

    return fig, ax1, ax2, ax3, ax4, ax5


def plotLC(date1, line1, error1, date2, line2, error2, figLoc, obj_name):
    # This figure will plot the light curves with error bars for two components on the same graph
    # It will plot two different scales on the same graph

    axis_font = {'size': '13'}

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.errorbar(date1, line1, yerr=error1, color='mediumblue', marker='o')
    ax1.set_xlabel('Date (MJD)', **axis_font)
    ax1.set_ylabel('CIV Line Flux (10$^{-17}$ erg/s/cm$^2$/$\AA$)', color='mediumblue', **axis_font)
    for tl in ax1.get_yticklabels():
        tl.set_color('mediumblue')

    ax2 = ax1.twinx()
    ax2.errorbar(date2, line2, yerr=error2, color='darkgreen', marker='o')
    ax2.set_ylabel('g-Band  Flux (10$^{-17}$ erg/s/cm$^2$/$\AA$)', color='darkgreen', **axis_font)
    for tl in ax2.get_yticklabels():
        tl.set_color('darkgreen')

    figName = figLoc + obj_name + "_" + "LC.png"
    print("Saving figure " + figName + "\n")
    plt.savefig(figName)

    return
