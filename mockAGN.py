# ---------------------------------------------------------- #
# ----------------------- mockAGN.py ----------------------- #
# --------- https://github.com/jhoormann/RMCodeDump -------- #
# ---------------------------------------------------------- #
# At one point I wanted to test how the continuum subtraction#
# window sizes and location (ie if you got so far away from  #
# the line the linear model broke down) affected the line    #
# flux value.  To do that I created a mock spectrum where    #
# the underlying properties were known that I could try and  #
# recover.                                                   #
# ---------------------------------------------------------- #

import numpy as np
import OzDES_Calculation as ozcalc
from astropy.modeling import models
import OzDES_Plotting as ozplot
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.patches as patches


# First define an array of wavelengths for your spectrum, here I chose rest frame wavelengths comparable to one of my
# CIV AGN
wave = np.linspace(1448, 3460, num=5000)

# Now I define a power law continuum over that wavelength range.
powerlaw = [10*pow(x, -0.4) + 1.5 for x in wave]

# Make CIV and MgII ish lines
# I have found a double Gaussian (one for the core, one for the wings) tends to model broad emission lines well so that
# is what I chose to use here.
cIVMod = models.Gaussian1D(amplitude=0.5, mean=1949, stddev=10) + models.Gaussian1D(amplitude=0.25, mean=1949, stddev=30)
mgIIMod = models.Gaussian1D(amplitude=0.4, mean=2998, stddev=20) + models.Gaussian1D(amplitude=0.2, mean=2998, stddev=70)

cIV = cIVMod(wave)
mgII = mgIIMod(wave)

# Now I will add all of the components together in order to get a complete spectrum
spectrum = powerlaw + cIV + mgII

# But we know better, that is never actually what we would observe, we have noise.  I will set a noise level and add
# in some random Gaussian noise to the spectrum

noise = 0.05  # change this depending on how noisy you want to get.
spectrum_noise = spectrum + spectrum*np.random.normal(size=spectrum.shape)*noise

# Plot the various spectrum components
fig, ax = ozplot.makeFigSingle("Mock AGN " + str(round(100*noise,1)) + "% Noise", "Wavelength", "Flux",
                               [wave[0], wave[-1]], [-0.1, 3.5])
ax.plot(wave, spectrum_noise, color='grey', linewidth=2)
ax.plot(wave, spectrum, color='black', linewidth=3)
ax.plot(wave, cIV, color='mediumblue', linestyle="--", linewidth=2)
ax.plot(wave, mgII, color='forestgreen', linestyle="--", linewidth=2)
ax.plot(wave, powerlaw, color='firebrick', linestyle="--", linewidth=5)

plt.show()
