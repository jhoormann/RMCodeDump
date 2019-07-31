# ---------------------------------------------------------- #
# ----------------------- RL-fitting.py -------------------- #
# --------- https://github.com/jhoormann/RMCodeDump -------- #
# ---------------------------------------------------------- #
# This is where I fit the lag-luminosity data to get the     #
# R-L relationship. This code calls the publicly available   #
# BCES code found here https://github.com/rsnemmen/BCES.     #
# ---------------------------------------------------------- #

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.integrate import fixed_quad
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FormatStrFormatter
import bces.bces

title_font = {'size':'22', 'color':'black', 'weight':'normal', 'verticalalignment':'bottom'}
axis_font = {'size':'22'}

def makeFigSingle(title, xlabel, ylabel, xlim=[0, 0], ylim=[0, 0]):
    fig = plt.figure()
    fig = plt.gcf()
    fig.set_size_inches(12, 9, forward=True)

    ax = fig.add_subplot(111)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(25)

    ax.set_ylabel(ylabel, **axis_font)
    if ylim != [0, 0] and ylim[0] < ylim[1]:
        ax.set_ylim(ylim)

    ax.set_xlabel(xlabel, **axis_font)
    if xlim != [0, 0] and xlim[0] < xlim[1]:
        ax.set_xlim(xlim)

    ax.set_title(title, **title_font)

    return fig, ax


# Read in the data of the lag/luminosity values you want to fit
# Note - luminosities are assumed to be log10(lum)
name_prev, lum_prev, lum_err_prev, lag_prev, lag_min_err_prev, lag_max_err_prev, z_prev = \
    np.loadtxt("exisitingLags_all.txt", dtype={'names':('name', 'lum', 'lumerr', 'lag', 'lagerrmin', 'lagerrmax', 'z'),
                                               'formats':('|S100', np.float, np.float, np.float, np.float, np.float,
                                                          np.float)},skiprows=1,unpack=True)


lum = lum_prev
lum_err = lum_err_prev

# Normalize by 10^44 which tends to be the value the CIV R-L relationship is normalized by - change as needed for your
# emission line of choice
lum = lum - 44
lum_err = lum*(lum_err_prev/lum_prev)


lag = np.log10(lag_prev)
lag_err_min = np.log10(lag_prev) - np.log10(lag_prev - lag_min_err_prev)
lag_err_max = np.log10(lag_prev+lag_max_err_prev) - np.log10(lag_prev)

# Now I call the bces package, see the documentation (https://github.com/rsnemmen/BCES) for the description of all the
# methods.  While this method does allow for inclusion of 2D error bars it does not allow for the use of asymmetric
# ones. I tested the results using both the positive and negative errors and it did not make a difference.

a, b, aerr, berr, covab = bces.bces.bces(lum, lum_err, lag, lag_err_max, np.zeros_like(lag))
print("BCES Results using + errors")
print("y|x")
print("Slope = " + str(round(a[0],3)) + " +/-" + str(round(aerr[0],3)))
print("Intercept = " + str(round(b[0], 3)) + " +/- " + str(round(berr[0],3)))
print("\nbissector")
print("Slope = " + str(round(a[2],3)) + " +/-" + str(round(aerr[2],3)))
print("Intercept = " + str(round(b[2], 3)) + " +/- " + str(round(berr[2],3)))
print("orthogonal") # I am pretty sure this is the one I used, although it doesn't make a huge difference.
# Bear in mind many other papers use the bissector method but it is now suggested that approach is not self consistent
print("Slope = " + str(round(a[3],3)) + " +/-" + str(round(aerr[3],3)))
print("Intercept = " + str(round(b[3], 3)) + " +/- " + str(round(berr[3],3)) + "\n")

a, b, aerr, berr, covab = bces.bces.bces(lum, lum_err, lag, lag_err_min, np.zeros_like(lag))
print("BCES Results using - errors")
print("y|x")
print("Slope = " + str(round(a[0],3)) + " +/-" + str(round(aerr[0],3)))
print("Intercept = " + str(round(b[0], 3)) + " +/- " + str(round(berr[0],3)))
print("\nbissector")
print("Slope = " + str(round(a[2],3)) + " +/-" + str(round(aerr[2],3)))
print("Intercept = " + str(round(b[2], 3)) + " +/- " + str(round(berr[2],3)))
print("orthogonal")
print("Slope = " + str(round(a[3],3)) + " +/-" + str(round(aerr[3],3)) + "\n")


# I also tried my own monte-carlo-esque method where randomly shuffle the lag/lum points within the uncertainty,
# fit the data, do this a bunch of times, and find the mean slope/intercept.  This gives a consistent value although
# the uncertainties are a bit small, I think the other method will be viewed as more realistic, particularly since we
# are already showing smaller errors than others.

nAGN = len(lum_prev)
nSim = 1000
slopes = np.zeros(nSim)
intercepts = np.zeros(nSim)

nLum = 100
nLag = 200

lum_array = np.zeros((nAGN, nLum))
lag_array = np.zeros((nAGN, nLag))
index = np.linspace(0,nAGN-1, nAGN).astype(int)

for i in range(nAGN):
    lum_array[i,:] = np.linspace(lum[i]-lum_err[i], lum[i]+lum_err[i], nLum)
    lag_array[i,:] = np.linspace(lag[i] - lag_err_min[i], lag[i] + lag_err_max[i], nLag)

for i in range(nSim):

    lag_index = nLag*np.random.rand(nAGN)
    lum_index = nLum*np.random.rand(nAGN)
    lag_index = lag_index.astype(int)
    lum_index = lum_index.astype(int)

    lag_temp = [lag_array[x, y] for x,y in zip(index, lag_index)]
    lum_temp = [lum_array[x, y] for x,y in zip(index, lum_index)]

    ourFit = np.poly1d(np.polyfit(lum_temp, lag_temp, 1))
    intercepts[i] = ourFit[0]
    slopes[i] = ourFit[1]


mean_slope = np.mean(slopes)
mean_intercept = np.mean(intercepts)
std_slope = np.std(slopes)
std_intercept = np.std(intercepts)

# Now plot the results
print("Monte Carlo-esque method")
print("Slope = " + str(round(mean_slope,3)) + " +/- " + str(round(std_slope,3)) + "\n")


fig0, ax0 = makeFigSingle("", "Log Luminosity", "Log Lag")

ax0.errorbar(lum, lag, yerr=[lag_err_min, lag_err_max], xerr=lum_err, fmt='o',
            color = 'black', markersize = 7)

lum_OzDES = np.linspace(38-44, 48-44, 10)
lag_OzDES = [b[3]+ a[3]*x for x in lum_OzDES]
ax0.plot(lum_OzDES, lag_OzDES, color = 'black')

plt.show()
