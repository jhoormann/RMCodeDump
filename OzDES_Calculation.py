# ---------------------------------------------------------- #
# ------------------ OzDES_Calculation.py ------------------ #
# --------- https://github.com/jhoormann/RMCodeDump -------- #
# ---------------------------------------------------------- #
# This is a dump of all the functions I have collated for    #
# the OzDES RM program.  This includes funtions defined in   #
# OzDES_calibSpec/getPhoto/makeLC plus some others.          #
# Unless otherwise noted this code was written by            #
# Janie Hoormann.                                            #
# ---------------------------------------------------------- #


from astropy.io import fits
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import fixed_quad
import OzDES_Plotting as ozplot
from scipy.interpolate import InterpolatedUnivariateSpline
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
import sys


# -------------------------------------------------- #
# Modified from a function originally provided by    #
# Anthea King                                        #
# -------------------------------------------------- #
# ----------------- SpectrumCoadd ------------------ #
# -------------------------------------------------- #
# Read in calibrated spectral data assuming data is  #
# in the format provided by OzDES_calibSpec after    #
# coadding.                                          #
# -------------------------------------------------- #
class SpectrumCoadd(object):
    # Spectrum class for latest version of the OzDES pipeline

    def __init__(self, filepath=None):
        assert filepath != None, "No file name is specified."
        self.filepath = filepath
        try:
            self.data = fits.open(filepath)
        except IOError:
            print("Error: file {0} could not be found".format(filepath))
            exit()
        data = fits.open(filepath)
        self.combined = data[0]
        self.combinedVariance = data[1]
        self._wavelength = None
        self._flux = None
        self._variance = None
        self._fluxCoadd = None
        self._varianceCoadd = None
        self._dates = None
        self._runs = None
        self.numEpochs = int((np.size(data) - 3) / 3)
        self.redshift = self.combined.header['z']
        self.RA = self.combined.header['RA']
        self.DEC = self.combined.header['DEC']
        self.field = self.combined.header['FIELD']


    @property
    def wavelength(self):
        """Define wavelength solution."""
        if getattr(self, '_wavelength', None) is None:
            crpix = self.combined.header[
                        'crpix1'] - 1.0  # Central pixel value. The -1.0 is needed as Python is ZERO indexed
            crval = self.combined.header['crval1']  # central wavelength value
            self.cdelt = self.combined.header['cdelt1']  # Wavelength interval between subsequent pixels
            n_pix = self.combined.header["NAXIS1"]
            wave = ((np.arange(n_pix) - crpix) * self.cdelt) + crval
            self._wavelength = wave
        return self._wavelength

    @property
    def flux(self):
        if getattr(self, '_flux', None) is None:
            self._flux = np.zeros((5000, self.numEpochs), dtype=float)
            for i in range(self.numEpochs):
                self._flux[:, i] = self.data[i * 3 + 3].data
        return self._flux

    @property
    def variance(self):
        if getattr(self, '_variance', None) is None:
            self._variance = np.zeros((5000, self.numEpochs), dtype=float)
            for i in range(self.numEpochs):
                self._variance[:, i] = self.data[i * 3 + 4].data
        return self._variance

    @property
    def fluxCoadd(self):
        if getattr(self, '_fluxCoadd', None) is None:
            self._fluxCoadd = np.zeros(5000, dtype=float)
            self._fluxCoadd[:] = self.data[0].data
        return self._fluxCoadd

    @property
    def varianceCoadd(self):
        if getattr(self, '_varianceCoadd', None) is None:
            self._varianceCoadd = np.zeros(5000, dtype=float)
            self._varianceCoadd[:] = self.data[1].data
        return self._varianceCoadd

    @property
    def dates(self):
        if getattr(self, '_dates', None) is None:
            self._dates = np.zeros(self.numEpochs, dtype=float)
            for i in range(self.numEpochs):
                self._dates[i] = self.data[i * 3 + 3].header[
                    'AVGDATE']  # this give the average Modified Julian Date (UTC) that observation was taken
        return self._dates

    @property
    def runs(self):
        if getattr(self, '_runs', None) is None:
            self._runs = np.zeros(self.numEpochs, dtype=float)
            for i in range(self.numEpochs):
                self._runs[i] = self.data[i * 3 + 3].header['RUN']  # this give the run number of the observation
        return self._runs


# -------------------------------------------------- #
# ------------------- magToFlux -------------------- #
# -------------------------------------------------- #
# Reads in magnitude, error, and pivot wavelength    #
# and converts to f_lambda in units of ergs/s/cm^2/A #
# -------------------------------------------------- #

def magToFlux(mag, err, pivot):
    flux = (3*pow(10,18)/pow(pivot,2))*pow(10, -(2./5.)*(mag + 48.6))
    flux_err = abs(flux*(-2./5.)*2.30259*err)
    return flux, flux_err


# -------------------------------------------------- #
# ------------------- outputLC --------------------- #
# -------------------------------------------------- #
# Creates an output file with date, flux, error      #
# columns as is expected by lag recovery tools       #
# Javelin and PyCCF.                                 #
# -------------------------------------------------- #
def outputLC(date, flux, error, name, loc, obj_name):

    length = len(date)
    outname = loc + obj_name + "_" + name + ".txt"
    output = open(outname, 'w')

    for i in range(length):
        if np.isnan(flux[i]) == False:
            output.write("%s    %s    %s \n" % (date[i], flux[i], error[i]))
        else:
            # Sometimes the flux ends up as nan, this is generally because the SNR is so bad/the emission line so
            # small that the continuum subtraction makes the line negative.  These are not saved in the data file
            # but a warning is outputted so you can have a look at what the problem is.
            print("-------\n  Houston, we have a problem! " + obj_name + " Night " + str(i) + "\n-------\n ")

    output.close()

    return

# -------------------------------------------------- #
# ---------------- convertPhotoLC -------------------#
# -------------------------------------------------- #
# Converts photometric light curves from magnitudes  #
# to flux and saves the light curves separately for  #
# each band.                                         #
# -------------------------------------------------- #
def convertPhotoLC(photoName, source, bandName, bandPivot, scale, makeFig, outLoc):
    # Read in the photometric data
    photo = pd.read_table(photoName, delim_whitespace=True)

    if makeFig == True:
        # Define figure and axis for light curves of all bands
        fig_photo, ax_photo = ozplot.plot_share_x(len(bandName), source, "Date (MJD)", bandName)

    # Make a light curve for each band
    for b in range(len(bandName)):
        # Create an array for observations of a specified band and sort observations by date
        band_data = photo[photo['BAND'] == bandName[b]].sort_values('MJD')
        # Find date, mag, and magerr array for the specified band
        ph_date = np.array(band_data['MJD'])
        ph_mag = np.array(band_data['MAG'])
        ph_magerr = np.array(band_data['MAGERR'])

        # Loop over each epoch and convert magnitude to flux
        ph_flux = np.zeros(len(ph_date))
        ph_fluxerr = np.zeros(len(ph_date))

        for e in range(len(ph_date)):
            ph_flux[e], ph_fluxerr[e] = magToFlux(ph_mag[e], ph_magerr[e], bandPivot[b])

        # Scale the fluxes before they are saved, if you are concerned about remembering the scale factor perhaps
        # included it in the outputted file name.
        ph_flux = ph_flux / scale
        ph_fluxerr = ph_fluxerr / scale

        # Save the data as a light curve with filename outLoc + source + _ + bandName[b] + .txt
        outputLC(ph_date, ph_flux, ph_fluxerr, bandName[b], outLoc, source)

        if makeFig == True:
            # plot the light curve on the subplot defined above.
            ax_photo[b].errorbar(ph_date, ph_flux, yerr=ph_fluxerr, fmt='o', color='black')

    # Once all the light curves are plotted save the figure as outLoc + source + "_photo.png"
    if makeFig == True:
        fig_photo.savefig(outLoc + source + "_photo.png")

    return


# -------------------------------------------------- #
# ------------------ findLines ----------------------#
# -------------------------------------------------- #
# Determines which emission lines are present in the #
# spectrum.  Returns an array of booleans where True #
# means the emission line is present.                #
# -------------------------------------------------- #
def findLines(wavelength, z, lineName, contWinBSMin, contWinBSMax):
    # decide which emission lines are available in the spectrum
    availLines = np.zeros(len(lineName)).astype(bool)

    for l in range(len(lineName)):
        # for a line to be in the spectrum you need to include the continuum subtraction windows as well.  This can
        # be limiting but as we need continuum subtracted spectra it is necessary.
        minWave = min(contWinBSMin[lineName[l]])
        maxWave = max(contWinBSMax[lineName[l]])

        if minWave * (1 + z) > wavelength[0] and maxWave * (1 + z) < wavelength[-1]:
            availLines[l] = True

    return availLines


# -------------------------------------------------- #
# -------------------- findBin ----------------------#
# -------------------------------------------------- #
# Finds the bin of the given vector (wavelength)     #
# where the specified quantity (line) is located.    #
# -------------------------------------------------- #
def findBin(line, wavelength):
    bin = 0
    for i in range(len(wavelength)-1):
        if line >= wavelength[i] and line <= wavelength[i+1]:
            bin = i
            i = len(wavelength)
        if line > wavelength[-1]:
            bin = len(wavelength)-1
            i = len(wavelength)
    return bin


# -------------------------------------------------- #
# ---------------- interpolateVals ------------------#
# -------------------------------------------------- #
# Interpolates a linear line between two points and  #
# propagates the uncertainty.                        #
# -------------------------------------------------- #
def interpolateVals(x, y, s, val):
    # uncertainty is variance

    interp = y[0] + (val - x[0]) * (y[1] - y[0]) / (x[1] - x[0])

    interp_var = s[0] + (s[0] + s[1]) * ((val - x[0]) / (x[1] - x[0])) ** 2.

    return interp, interp_var


# -------------------------------------------------- #
# ------------------ meanUncert ---------------------#
# -------------------------------------------------- #
# Finds the uncertainty corresponding to the mean    #
# of a set of numbers.                               #
# -------------------------------------------------- #
def meanUncert(variance):
    length = len(variance)
    var = 0
    num = 0
    for i in range(length):
        if np.isnan(variance[i]) == False:
            var = var + variance[i]
            num += 1

    sigma2 = (var / (num ** 2))

    return sigma2


# -------------------------------------------------- #
# ---------------- cont_fit_reject ------------------#
# -------------------------------------------------- #
# Interpolate a linear line through the mean of the  #
# continuum subtraction windows to represent the     #
# continuum and subtract this line.  Modifies the    #
# given flux/variance vectors.                       #
# -------------------------------------------------- #
def cont_fit_reject(wavelength, fluxes, variances, minWin, maxWin):

    # Define the wavelength range for the continuum model, between the mean of both windows
    wave = np.array([np.nanmean(minWin), np.nanmean(maxWin)])
    nBins = len(wavelength)

    # Determine how many epochs there are to continuum subtract
    number = int(fluxes.size / nBins)

    for epoch in range(number):
        if number == 1:
            flux = fluxes
            variance = variances
        else:
            flux = fluxes[:, epoch]
            variance = variances[:, epoch]

        # Calculate the average flux at each extreme of the wave vector (ie mean of the continuum subtraction window)
        fvals = np.array([np.nanmean(flux[findBin(minWin[0], wavelength):findBin(minWin[1], wavelength)]),
                          np.nanmean(flux[findBin(maxWin[0], wavelength):findBin(maxWin[1], wavelength)])])

        # Calculate the average uncertainty at each extreme of the wave vector
        svals = np.array([meanUncert(variance[findBin(minWin[0], wavelength):findBin(minWin[1], wavelength)]),
                          meanUncert(variance[findBin(maxWin[0], wavelength):findBin(maxWin[1], wavelength)])])

        cont = np.zeros(nBins)
        contVar = np.zeros(nBins)

        # Find the interpolated linear continuum model
        for i in range(nBins):
            cont[i], contVar[i] = interpolateVals(wave, fvals, svals, wavelength[i])

        # Subtract the continuum from the flux and add the error of the model in quadrature with the spectral error
        flux -= cont
        variance += contVar

    return


# -------------------------------------------------- #
# The next three functions are modified from code    #
# provided by Dale Mudd                              #
# -------------------------------------------------- #
# ------------------ filterCurve ------------------- #
# -------------------------------------------------- #
# creates a class to hold the transmission function  #
# for each band.                                     #
# -------------------------------------------------- #
class filterCurve:
    """A filter"""

    def __init__(self):
        self.wave = np.array([], 'float')
        self.trans = np.array([], 'float')
        return

    def read(self, file):
        # DES filter curves express the wavelengths in nms
        if 'DES' in file:
            factor = 10.
        else:
            factor = 1.
        file = open(file, 'r')
        for line in file.readlines():
            if line[0] != '#':
                entries = line.split()
                self.wave = np.append(self.wave, float(entries[0]))
                self.trans = np.append(self.trans, float(entries[1]))
        file.close()
        # We use Angstroms for the wavelength in the filter transmission file
        self.wave = self.wave * factor
        return


# -------------------------------------------------- #
# ---------------- readFilterCurve ----------------- #
# -------------------------------------------------- #
# Reads in the filter curves and stores it as the    #
# filter curve class.                                #
# -------------------------------------------------- #
def readFilterCurves(bands, filters):

    filterCurves = {}
    for f in bands:
        filterCurves[f] = filterCurve()
        filterCurves[f].read(filters[f])

    return filterCurves


# -------------------------------------------------- #
# ----------------- computeABmag ------------------- #
# -------------------------------------------------- #
# computes the AB magnitude for given transmission   #
# functions and spectrum (f_lambda).  Returns the    #
# magnitude and variance.                            #
# -------------------------------------------------- #
def computeABmag(trans_flux, trans_wave, tmp_wave, tmp_flux, tmp_var):
    # Takes and returns variance
    # trans_ : transmission function data
    # tmp_ : spectral data

    # trans/tmp not necessarily defined over the same wavelength range
    # first determine the wavelength range over which both are defined
    minV = min(trans_wave)
    if minV < min(tmp_wave):
        minV = min(tmp_wave)
    maxV = max(trans_wave)
    if maxV > max(trans_wave):
        maxV = max(trans_wave)

    interp_wave = []
    tmp_flux2 = []
    tmp_var2 = []

    # Make new vectors for the flux just using that range (assuming spectral binning)

    for i in range(len(tmp_wave)):
        if minV < tmp_wave[i] < maxV:
            interp_wave.append(tmp_wave[i])
            tmp_flux2.append(tmp_flux[i])
            tmp_var2.append(tmp_var[i])

    # interpolate the transmission function onto this range
    # the transmission function is interpolated as it is generally much smoother than the spectral data
    trans_flux2 = interp1d(trans_wave, trans_flux)(interp_wave)

    # And now calculate the magnitude and uncertainty

    c = 2.992792e18  # Angstrom/s
    Num = np.nansum(tmp_flux2 * trans_flux2 * interp_wave)
    Num_var = np.nansum(tmp_var2 * (trans_flux2 * interp_wave) ** 2)
    Den = np.nansum(trans_flux2 / interp_wave)

    with np.errstate(divide='raise'):
        try:
            magAB = -2.5 * np.log10(Num / Den / c) - 48.60
            magABvar = 1.17882 * Num_var / (Num ** 2)
        except FloatingPointError:
            magAB = 99.
            magABvar = 99.

    return magAB, magABvar


# --------------------------------------------------- #
# --------------- uncertainty_cont ------------------ #
# --------------------------------------------------- #
# This function finds the uncertainty in line flux    #
# and width measurements.  For line flux you can      #
# input a range of potential continuum windows and    #
# it will randomly pick regions to use for continuum  #
# subtraction. You can also input a region over which #
#  to randomly choose the integration window.  These  #
# all also include flux randomization in order to     #
# consider the effect of the variance spectrum.       #
# You can also look at the effect flux randomization  #
# has on the line width measurements FWHM and         #
# velocity dispersion.  You can also specify to look  #
# at the RMS spectrum (flag='rms') for the line width #
# measurements, the default is to look at the provided#
# spectrum as is.  The error is calculated through    #
# bootstrap resampling using strapNum iterations.     #
# The standard deviation of the calculated quantity   #
# is then the associated error.                       #
# --------------------------------------------------- #
def uncertainty_cont(wavelength, flux, variance, strapNum, z, line, pivotLC, winLimMin, winLimMax, winsize, scale,
                     calc='cont', flag='mean', res=0):

    # calc = cont -> continuum subtraction
    # calc = win -> integration window
    # calc = fwhm -> FWHM line width: can specify flag=rms
    # calc = sigma -> line velocity dispersion: can specify flag=rms

    # Performs bootstrap resampling in the range of potentially clean continuum to determine
    # uncertainties on the flux measurement

    # Continuum window in Angstroms - will be scaled according to redshift

    # Winsize means the continuum subtraction windows are all the same size, just the locations shift
    winsize = winsize/(1+z)

    lineMin = line[0]
    lineMax = line[1]

    # Option for potentially clean continuum region pass in bootstrap

    # Calculate the width of the bootstrapping region on each side of the line
    lowW = (winLimMin[1]-winLimMin[0])/(1+z)
    highW = (winLimMax[1]-winLimMax[0])/(1+z)

    # Check edge conditions: if the bootstraping region goes outside the region of the spectrograph use the spectrograph
    # bounds as the edges
    if winLimMin[0] < wavelength[0]:
        winLimMin[0] = wavelength[0]
        winLimMin[1] = (winLimMin[0] / (1 + z) + lowW) * (1 + z)
    if winLimMin[1] > wavelength[line[0]]:
        winLimMin[1] = wavelength[line[0]]
    if winLimMax[1] > wavelength[-1]:
        winLimMax[1] = wavelength[-1]
        winLimMax[0] = (winLimMax[1] / (1 + z) - highW) * (1 + z)
    if winLimMax[0] < wavelength[line[1]]:
        winLimMax[0] = wavelength[line[1]]

    # Wavelengths to choose in each window in steps of 0.5A
    winMinVect = np.arange(winLimMin[0], winLimMin[1] - (winsize - 0.5) * (1 + z), 0.5 * (1 + z))
    winMaxVect = np.arange(winLimMax[0], winLimMax[1] - (winsize - 0.5) * (1 + z), 0.5 * (1 + z))

    # Array of random continuum window starting points
    randVectMin = len(winMinVect) * np.random.rand(strapNum)
    randVectMin = randVectMin.astype(int)

    randVectMax = len(winMaxVect) * np.random.rand(strapNum)
    randVectMax = randVectMax.astype(int)

    # An array of values obtained through bootstrapping to determine uncertainties
    vals = np.zeros(strapNum)

    for i in range(strapNum):

        if calc == 'win':
            # subtracts from standard continuum but changes integration window, in this case feed in potential
            # integration windows instead of bootstrapping regions

            lineMinNew = findBin(winMinVect[randVectMin[i]], wavelength)
            lineMaxNew = findBin(winMaxVect[randVectMax[i]], wavelength)

            # Performs flux resampling to account for variance spectrum.  Flux values shifted by Gaussian scaled by
            # variance
            varC = np.copy(variance)
            fluxC = flux + np.random.normal(size=flux.shape) * (variance ** 0.5)

            # Continuum Subtract this new vector
            cont_fit_reject(wavelength, fluxC, varC, winLimMin, winLimMax)

            # Calculate the flux
            lc_mag, lc_mag_err = computeABmag(np.ones(len(wavelength[lineMinNew:lineMaxNew])),
                                              wavelength[lineMinNew:lineMaxNew], wavelength[lineMinNew:lineMaxNew],
                                              fluxC[lineMinNew:lineMaxNew]*scale, varC[lineMinNew:lineMaxNew]*
                                              pow(scale,2))

            vals[i], lc_mag_err = magToFlux(lc_mag, lc_mag_err**0.5, pivotLC)

        if calc == "cont":
            # changes cont region
            minWin = [winMinVect[randVectMin[i]], winMinVect[randVectMin[i]] + winsize * (1 + z)]
            maxWin = [winMaxVect[randVectMax[i]], winMaxVect[randVectMax[i]] + winsize * (1 + z)]

            # Performs flux resampling to account for variance spectrum.  Flux values shifted by Gaussian scaled by
            # variance
            varC = np.copy(variance)
            fluxC = flux + np.random.normal(size=flux.shape) * (variance ** 0.5)

            # Continuum Subtract this new vector
            cont_fit_reject(wavelength, fluxC, varC, minWin, maxWin)

            # Calculate the flux
            lc_mag, lc_mag_err = computeABmag(np.ones(len(wavelength[lineMin:lineMax])),wavelength[lineMin:lineMax],
                                              wavelength[lineMin:lineMax], fluxC[lineMin:lineMax]*scale,
                                              varC[lineMin:lineMax]*pow(scale, 2))

            vals[i], lc_mag_err = magToFlux(lc_mag, lc_mag_err**0.5, pivotLC)

        if calc == "fwhm":
            # Determine uncertainty in FWHM line measurement
            # do flux randomization and continuum subtraction
            varC = np.copy(variance)
            fluxC = flux + np.random.normal(size=flux.shape) * (variance ** 0.5)
            cont_fit_reject(wavelength, fluxC, varC, winLimMin, winLimMax)

            if flag == 'rms':
                # first calculate the RMS spectrum if requested
                fluxC, varC = rmsSpec(fluxC, varC)

            vals[i] = fwhm(wavelength[lineMin:lineMax], fluxC[lineMin:lineMax], res)

        if calc == "sigma":
            # Determine uncertainty in velocity dispersion measurement
            # do flux randomization and continuum subtraction
            varC = np.copy(variance)
            fluxC = flux + np.random.normal(size=flux.shape) * (variance ** 0.5)
            cont_fit_reject(wavelength, fluxC, varC, winLimMin, winLimMax)

            if flag == 'rms':
                # first calculate the RMS spectrum if requested
                fluxC, varC = rmsSpec(fluxC, varC)
            vals[i] = lineVar(wavelength[lineMin:lineMax], fluxC[lineMin:lineMax], res)

    stddev_bs = np.nanstd(vals)
    return stddev_bs


# --------------------------------------------------- #
# ----------------------- fwhm ---------------------- #
# --------------------------------------------------- #
# Takes an input spectrum and calculate the FWHM of   #
# the provided emission line.  It will search over    #
# the entire provided wavelength window so just       #
# include the relevant region of the spectrum.        #
# --------------------------------------------------- #
def fwhm(wave, flux, res):
    # First I am smoothing the spectrum
    exponential_smooth(flux)

    # Find the half maximum
    peak = max(flux)
    valley = min(flux)
    peakLoc = wave[np.where(flux == peak)[0][0]]
    peakLocB = findBin(peakLoc, wave)
    hm = (peak-valley) / 2 + valley

    leftUp = wave[0]
    leftDown = wave[peakLocB]
    rightUp = wave[-1]
    rightDown = wave[peakLocB]

    # First search for the half max to the left of the line
    for i in range(peakLocB):
        # First search going up the line
        if flux[i] < hm < flux[i+1]:
            leftUp = (wave[i] + wave[i+1])/2
        # Then going down the line
        if flux[peakLocB-i-1] < hm < flux[peakLocB-i]:
            leftDown = (wave[peakLocB-i-1] + wave[peakLocB-i])/2

    # Then take the average which will account for any double peaks/noise in the spectrum
    left = (leftUp + leftDown)/2

    # And now to the right
    maxSize = len(wave) - 1

    for i in range(maxSize - peakLocB):
        # Go up
        if flux[peakLocB + i + 1] < hm < flux[peakLocB + i]:
            rightDown = (wave[peakLocB + i] + wave[peakLocB + i + 1])/2
        # Go down
        if flux[maxSize-i] < hm < flux[maxSize-i-1]:
            rightUp = (wave[maxSize-i] + wave[maxSize-i-1])/2

    right = (rightUp + rightDown)/2

    # Now calculate the velocity

    # km/s
    c = 299792.458

    widthObs = (right-left)
    widthT = pow(widthObs**2 - res**2,0.5)/2

    zLeft = -widthT/peakLoc
    zRight = widthT/peakLoc

    zComb = (1+zRight)/(1+zLeft)-1

    vel = c*((1+zComb)**2-1)/((1+zComb)**2+1)

    return vel


# --------------------------------------------------- #
# ---------------------- lineVar -------------------- #
# --------------------------------------------------- #
# Takes an input spectrum and calculate the velocity  #
# dispersion of the emission line.  It will search    #
# over the entire provided wavelength window so just  #
# include the relevant region of the spectrum.        #
# --------------------------------------------------- #
def lineVar(wave, flux, res):
    length = len(wave)

    peak = max(flux)
    peakLoc = wave[np.where(flux == peak)[0][0]]

    # Calculate velocity dispersion following equation written in Peterson 2004, the three following constants
    # correspond to the main terms in that equation.

    Pdl = 0
    lPdl = 0
    l2Pdl = 0

    for i in range(length):
        Pdl += flux[i]

        lPdl += flux[i] * wave[i]

        l2Pdl += flux[i] * pow(wave[i], 2)

    lambda0 = lPdl / Pdl

    lambda2 = l2Pdl / Pdl

    lambda02 = pow(lambda0, 2)

    linevar = lambda2 - lambda02

    sigma = linevar ** 0.5

    c = 299792.458

    sigmaT = pow(sigma**2 - res**2, 0.5)

    left = peakLoc - sigmaT / 2
    right = peakLoc + sigmaT / 2

    zLeft = (left - peakLoc) / peakLoc
    zRight = (right - peakLoc) / peakLoc


    #redshift from lambda_l to lambda_r
    zComb = (1 + zRight) / (1 + zLeft) - 1

    vel = c * ((1 + zComb) ** 2 - 1) / ((1 + zComb) ** 2 + 1)

    return vel


# --------------------------------------------------- #
# --------------- exponential_smooth ---------------- #
# --------------------------------------------------- #
# Function to apply an exponential smoothing kernel   #
# to the data.  Written by Harry Hobson.              #
# --------------------------------------------------- #
def exponential_smooth(fluxes):

    number = int(fluxes.size/fluxes.shape[0])

    search_pixels = 5
    decay = 0.9

    window = np.arange(-search_pixels, search_pixels + 1)
    weights = decay ** np.abs(window)
    weights /= np.sum(weights)

    if (number == 1):
        flux = fluxes[:]
        flux[:] = np.convolve(flux, weights, mode='same')
    else:
        for epoch in range(fluxes.shape[1]):
            flux = fluxes[:, epoch]
            flux[:] = np.convolve(flux, weights, mode='same')


# --------------------------------------------------- #
# -------------------- meanSpec --------------------- #
# --------------------------------------------------- #
# Calculates the mean of multiple spectra as well as  #
# the corresponding variance spectrum.                #
# --------------------------------------------------- #
def meanSpec(flux, variance):

    length = len(flux[:,0])

    meanFlux = np.zeros(length)
    meanVar = np.zeros(length)

    for i in range(length):
        meanFlux[i] = np.nanmean(flux[i,:])
        meanVar[i] = np.nanmean(variance[i,:])

    return meanFlux, meanVar


# --------------------------------------------------- #
# -------------------- rmsSpec ---------------------- #
# --------------------------------------------------- #
# Calculates the RMS of the inputted spectra.  Will   #
# expect fluxes in [wavelength, epoch] format.  An    #
# exponential smoothing function is applied to the    #
# data as a first and last step to mitigate some of   #
# the noise.                                          #
# --------------------------------------------------- #
def rmsSpec(flux, variance):
    # smooth the input spectra
    exponential_smooth(flux)

    length = len(flux[:, 0])
    epochs = len(flux[0, :])

    # Calculate the RMS spectrum, variance propagated through but not used later
    mean, meanVar = meanSpec(flux, variance)
    rms = np.zeros(length)
    rmsVar = np.zeros(length)
    rmsVar2 = np.zeros(length)

    for b in range(length):
        for e in range(epochs):
            rms[b] += (flux[b, e] - mean[b]) ** 2
            rmsVar2[b] += 4 * rms[b] * (variance[b, e] + meanVar[b])
        rms[b] = (rms[b] / (epochs - 1)) ** 0.5
        rmsVar2[b] = rmsVar2[b] / ((epochs - 1) ** 2)
        rmsVar[b] = rmsVar2[b] * (0.5 / rms[b]) ** 2

    # smooth the final RMS spectrum
    exponential_smooth(rms)

    return rms, rmsVar


# -------------------------------------------------- #
# -------------------- lineLC ---------------------- #
# -------------------------------------------------- #
# Create emission line light curves by integrating   #
# the emission lines after local continuum           #
# subtraction.  The uncertainties due to the variance#
# of the spectrum and the continuum subtraction is   #
# performed through bootstrap resampling.  This is   #
# done for every emission line from the provided list#
# that is present in the spectrum.                   #
# -------------------------------------------------- #
def lineLC(dates, lineName, availLines, lineInt, contWinMin, contWinMax, contWinBSMin, contWinBSMax, wavelength,
           origFluxes, origVariances, fluxCoadd, numEpochs, scale, z, strapNum, outLoc, source, makeFig, makeFigEpoch):

    if makeFig == True:
        # Define figure and axis for light curves of all available emission lines
        lineAxis = [lineName[i] for i in range(len(lineName)) if availLines[i] == True]
        fig_spec, ax_spec = ozplot.plot_share_x(len(lineAxis), source, "Date (MJD)", lineAxis)

    for l in range(len(lineName)):
        if availLines[l] == True:
            line = lineName[l]

            # Copy the flux/variance vectors so you have an uncontinuum subtracted version to use for other lines
            fluxes = np.copy(origFluxes)
            variances = np.copy(origVariances)

            # define some variables for line/continuum windows in observed frame
            contMin = np.array(contWinMin[line]) * (1 + z)
            contMax = np.array(contWinMax[line]) * (1 + z)
            contMinBS = np.array(contWinBSMin[line]) * (1 + z)
            contMaxBS = np.array(contWinBSMax[line]) * (1 + z)

            # similar for the line integration window but I want the wavelength bin number, not just the wavelength
            lineMin = findBin(lineInt[line][0] * (1 + z), wavelength)
            lineMax = findBin(lineInt[line][1] * (1 + z), wavelength)

            # Perform the continuum subtraction
            cont_fit_reject(wavelength, fluxes, variances, contMin, contMax)

            lc_mag = np.zeros(numEpochs)
            lc_mag_sigma = np.zeros(numEpochs)
            lc_flux = np.zeros(numEpochs)
            lc_flux_sigma = np.zeros(numEpochs)
            total_error = np.zeros(numEpochs)

            # Calculate the pivot wavelength associated with each line window
            pivotLC = pow(np.nansum(wavelength[lineMin:lineMax]) / np.nansum(1 / wavelength[lineMin:lineMax]), 0.5)

            # Calculate magnitudes and fluxes for each line
            for epoch in range(numEpochs):
                # first calculate magnitudes, save these if you want to compare this instead of fluxes
                # Here the transmission function is 1 for all wavelengths within the integration window.
                lc_mag[epoch], lc_mag_sigma[epoch] = computeABmag(np.ones(len(wavelength[lineMin:lineMax])),
                                                                  wavelength[lineMin:lineMax],
                                                                  wavelength[lineMin:lineMax],
                                                                  fluxes[lineMin:lineMax, epoch] * scale,
                                                                  variances[lineMin:lineMax, epoch] * pow(scale, 2))
                # Now convert to flux, this is what is saved.  Note: all fluxes here are actually flux densities
                # This uncertainty just considers the variance spectrum, we will take everything in the next step
                lc_flux[epoch], lc_flux_sigma[epoch] = magToFlux(lc_mag[epoch], lc_mag_sigma[epoch] ** 0.5, pivotLC)
                total_error[epoch] = uncertainty_cont(wavelength, origFluxes[:, epoch], origVariances[:, epoch],
                                                      strapNum, z, [lineMin, lineMax], pivotLC, contMinBS,
                                                      contMaxBS, contMin[1] - contMin[0], scale)

                if makeFigEpoch == True:
                    # Save figures showing spectrum before/after continuum subtraction for each epoch and line
                    fig_epoch, ax_epoch = ozplot.plot_share_x(2, source + " epoch " + str(epoch), "Wavelength ($\AA$)",
                                                       ["Before", " After"], [wavelength[0], wavelength[-1]])
                    for p in range(2):
                        ax_epoch[p].axvspan(contMinBS[0], contMinBS[1], color='mediumblue', alpha=0.3)
                        ax_epoch[p].axvspan(contMaxBS[0], contMaxBS[1], color='mediumblue', alpha=0.3)
                        ax_epoch[p].axvspan(contMin[0], contMin[1], color='mediumblue', alpha=0.5)
                        ax_epoch[p].axvspan(contMax[0], contMax[1], color='mediumblue', alpha=0.5)
                        ax_epoch[p].axvspan(wavelength[lineMin], wavelength[lineMax], color='forestgreen', alpha=0.3)
                    ax_epoch[0].plot(wavelength, origFluxes[:, epoch], color='black')
                    ax_epoch[1].plot(wavelength, fluxes[:, epoch], color='black')
                    fig_epoch.savefig(outLoc + source + "_" + lineName[l] + "_epoch_" + str(epoch) + ".png")
                    plt.close(fig_epoch)

            # Scale the line fluxes as with the photometry
            lc_flux = lc_flux / scale
            total_error = total_error / scale

            # Save the data as a light curve with filename outLoc + source + _ + line + .txt
            outputLC(dates, lc_flux, total_error, line, outLoc, source)

            if makeFig == True:
                # plot the light curve on the subplot defined above. First get the index for the axis associated with
                # the line being analyzed.
                lbin = lineAxis.index(line)

                ax_spec[lbin].errorbar(dates, lc_flux, yerr=total_error, fmt='o', color='black')

                # make a plot to show the continuum subtraction regions on the coadded spectrum
                fig_coadd, ax_coadd = ozplot.plot_share_x(1, source, "Wavelength ($\AA$)", ["Total Coadded Flux (" +
                                                                                            str(scale) +
                                                                                            " erg/s/cm$^2$/$\AA$)"],
                                                          [wavelength[0], wavelength[-1]])
                ax_coadd[0].axvspan(contMinBS[0], contMinBS[1], color='mediumblue', alpha=0.3)
                ax_coadd[0].axvspan(contMaxBS[0], contMaxBS[1], color='mediumblue', alpha=0.3)
                ax_coadd[0].axvspan(contMin[0], contMin[1], color='mediumblue', alpha=0.5)
                ax_coadd[0].axvspan(contMax[0], contMax[1], color='mediumblue', alpha=0.5)
                ax_coadd[0].axvspan(wavelength[lineMin], wavelength[lineMax], color='forestgreen', alpha=0.3)
                ax_coadd[0].plot(wavelength, fluxCoadd, color='black')
                fig_coadd.savefig(outLoc + source + "_" + lineName[l] + "_coadd.png")
                plt.close(fig_coadd)

    # Once all the light curves are plotted save the figure as outLoc + source + "_spec.png"
    if makeFig == True:
        fig_spec.savefig(outLoc + source + "_spec.png")

    return


# -------------------------------------------------- #
# ------------------ makePhotoLC --------------------#
# -------------------------------------------------- #
# Makes light curves by applying photometric filters #
# to a series of spectral data.  The data is saved   #
# as fluxes.                                         #
# -------------------------------------------------- #
def makePhotoLC(dates, bandName, bandPivot, filters, wavelength, origFluxes, origVariances, numEpochs, scale, outLoc,
                source, makeFig):

    filterCurves = readFilterCurves(bandName, filters)

    if makeFig == True:
        # Define figure and axis for light curves of all available emission lines
        fig_phot, ax_phot = ozplot.plot_share_x(len(bandName), source, "Date (MJD)", bandName)

    for b in range(len(bandName)):
        mags = np.zeros(numEpochs)
        mags_var = np.zeros(numEpochs)
        flux = np.zeros(numEpochs)
        flux_err = np.zeros(numEpochs)

        for e in range(numEpochs):
            # Calculate the magntiude given the transmission function provided
            mags[e], mags_var[e] = computeABmag(filterCurves[bandName[b]].trans, filterCurves[bandName[b]].wave,
                                                wavelength, origFluxes[:, e] * scale,
                                                origVariances[:, e] * pow(scale, 2))
            # Then convert to fluxes
            flux[e], flux_err[e] = magToFlux(mags[e], mags_var[e] ** 0.5, bandPivot[b])

        # Scale the  fluxes
        flux = flux / scale
        flux_err = flux_err / scale

        # Save the data as a light curve with filename outLoc + source + _ + calc_bandName + .txt
        outputLC(dates, flux, flux_err, 'calc_' + bandName[b], outLoc, source)

        if makeFig == True:
            # plot the light curve on the subplot defined above.
            ax_phot[b].errorbar(dates, flux, yerr=flux_err, fmt='o', color='black')

    # Once all the light curves are plotted save the figure as outLoc + source + "_makePhot.png"
    if makeFig == True:
        fig_phot.savefig(outLoc + source + "_makePhot.png")

    return


# -------------------------------------------------- #
# ------------------- calcWidth ---------------------#
# -------------------------------------------------- #
# Calculates emission line width (FWHM and velocity  #
# dispersion) using the mean and RMS spectra.  If    #
# possible calculates the BH mass using the R-L      #
# relationship.  The data is saved to a text file.   #
# -------------------------------------------------- #
def calcWidth(wavelength, lineName, lineLoc, availLines, lineInt, lumLoc, contWinMin, contWinMax, contWinBSMin,
              contWinBSMax, origFluxes, origVariances, origFluxCoadd, origVarCoadd, z, strapNum, scale, outLoc, source,
              makeFig, calcBH):

    # open a file to save the data to - outLoc + source + _vel.txt
    out = open(outLoc + source + "_vel_and_mass.txt", 'w')

    # Type (Mean/RMS), Measure (FWHM, Vel Disp)
    out.write("Line Type Measure Vel Vel_Err Mass Lag Lag_Err_Min Lag_Err_Max Mass_Err_Min, Mass_Err_Max\n")

    # Convert wavelength vector to rest frame
    wave = wavelength/(1+z)
    for l in range(len(lineName)):
        if availLines[l] == True:
            line = lineName[l]

            # If calcBH == True estimate BH mass from the R-L relationship.  Here I will calculate the lag.  If you want
            # to use the measured lag feed that in here.  If the luminosity needed isn't in the spectroscopic window
            # I will just give nan for the black hole mass.  The luminosity is determined from the coadded flux

            if calcBH == True:
                lum, lumerr = luminosity(wavelength, origFluxCoadd, origVarCoadd, z, lumLoc[l]*(1+z), strapNum, scale)
                if np.isnan(lum) == True:
                    lag = np.nan
                    lag_err_min = np.nan
                    lag_err_max = np.nan
                elif line == 'CIV':
                    lag, lag_err_max, lag_err_min = RL_CIV(lum, lumerr)
                elif line == 'MgII':
                    lag, lag_err_max, lag_err_min = RL_MgII(lum, lumerr)
                elif line == 'Hbeta':
                    lag, lag_err_max, lag_err_min = RL_Hbeta(lum, lumerr)
            else:
                lag = np.nan
                lag_err_min = np.nan
                lag_err_max = np.nan

            # Calculate the resolution of the spectrograph at the specified wavelength
            res = findRes(lineLoc[l], z)

            # define some variables for line/continuum windows in rest frame
            contMin = np.array(contWinMin[line])
            contMax = np.array(contWinMax[line])
            contMinBS = np.array(contWinBSMin[line])
            contMaxBS = np.array(contWinBSMax[line])

            # similar for the line integration window but I want the wavelength bin number, not just the wavelength
            lineMin = findBin(lineInt[line][0], wave)
            lineMax = findBin(lineInt[line][1], wave)

            fluxes = np.copy(origFluxes)
            variances = np.copy(origVariances)

            fluxCoadd = np.copy(origFluxCoadd)
            varCoadd = np.copy(origVarCoadd)

            # Perform the continuum subtraction on epochs and coadd
            cont_fit_reject(wave, fluxes, variances, contMin, contMax)
            cont_fit_reject(wave, fluxCoadd, varCoadd, contMin, contMax)

            # First look at the mean spectrum, let's smooth it
            # FWHM
            vel_mean_fwhm = fwhm(wave[lineMin:lineMax], fluxCoadd[lineMin:lineMax], res)
            err_mean_fwhm = uncertainty_cont(wave, origFluxCoadd, origVarCoadd, strapNum, 0, [lineMin, lineMax], 0,
                                             contMinBS, contMaxBS, contMin[1] - contMin[0], scale, calc='fwhm',
                                             flag='mean', res=res)


            # Sigma
            vel_mean_sigma = lineVar(wave[lineMin:lineMax], fluxCoadd[lineMin:lineMax], res)
            err_mean_sigma = uncertainty_cont(wave, origFluxCoadd, origVarCoadd, strapNum, 0, [lineMin, lineMax], 0,
                                              contMinBS, contMaxBS, contMin[1] - contMin[0], scale, calc='sigma',
                                              flag='mean', res=res)

            # Now look at the RMS spectrum
            rms, rms_var = rmsSpec(fluxes, variances)
            vel_rms_fwhm = fwhm(wave[lineMin:lineMax], rms[lineMin:lineMax], res)
            err_rms_fwhm = uncertainty_cont(wave, origFluxes, origVariances, strapNum, 0, [lineMin, lineMax], 0,
                                            contMinBS, contMaxBS, contMin[1] - contMin[0], scale, calc='fwhm',
                                            flag='rms', res=res)

            # Sigma
            vel_rms_sigma = fwhm(wave[lineMin:lineMax], rms[lineMin:lineMax], res)
            err_rms_sigma = uncertainty_cont(wave, origFluxes, origVariances, strapNum, 0, [lineMin, lineMax], 0,
                                             contMinBS, contMaxBS, contMin[1] - contMin[0], scale, calc='sigma',
                                             flag='rms', res=res)

            if calcBH == True and np.isnan(lag) == False:
                # Calculate BH mass for all 4 line measurements
                mass_mean_fwhm, mass_min_mean_fwhm, mass_max_mean_fwhm = \
                    blackHoleMass(lag, lag_err_min, lag_err_max, vel_mean_fwhm, err_mean_fwhm)
                mass_mean_sigma, mass_min_mean_sigma, mass_max_mean_sigma = \
                    blackHoleMass(lag, lag_err_min, lag_err_max, vel_mean_sigma, err_mean_sigma)

                mass_rms_fwhm, mass_min_rms_fwhm, mass_max_rms_fwhm = \
                    blackHoleMass(lag, lag_err_min, lag_err_max, vel_rms_fwhm, err_rms_fwhm)
                mass_rms_sigma, mass_min_rms_sigma, mass_max_rms_sigma = \
                    blackHoleMass(lag, lag_err_min, lag_err_max, vel_rms_sigma, err_rms_sigma)
            else:
                mass_mean_fwhm, mass_min_mean_fwhm, mass_max_mean_fwhm = np.nan, np.nan, np.nan
                mass_mean_sigma, mass_min_mean_sigma, mass_max_mean_sigma = np.nan, np.nan, np.nan

                mass_rms_fwhm, mass_min_rms_fwhm, mass_max_rms_fwhm = np.nan, np.nan, np.nan
                mass_rms_sigma, mass_min_rms_sigma, mass_max_rms_sigma = np.nan, np.nan, np.nan

            out.write(line + " MEAN FWHM %d %d %d %d %d %2.2f %2.2f %2.2f \n" %(vel_mean_fwhm, err_mean_fwhm, lag,
                                                                                lag_err_min, lag_err_max,
                                                                                mass_mean_fwhm, mass_min_mean_fwhm,
                                                                                mass_max_mean_fwhm))
            out.write(line + " MEAN Sigma %d %d %d %d %d %2.2f %2.2f %2.2f \n" %(vel_mean_sigma, err_mean_sigma, lag,
                                                                                 lag_err_min, lag_err_max,
                                                                                 mass_mean_sigma, mass_min_mean_sigma,
                                                                                 mass_max_mean_sigma))
            out.write(line + " RMS FWHM %d %d %d %d %d %2.2f %2.2f %2.2f \n" %(vel_rms_fwhm, err_rms_fwhm, lag,
                                                                               lag_err_min, lag_err_max, mass_rms_fwhm,
                                                                               mass_min_rms_fwhm, mass_max_rms_fwhm))
            out.write(line + " RMS Sigma %d %d %d %d %d %2.2f %2.2f %2.2f \n" %(vel_rms_sigma, err_rms_sigma,
                                                                                lag,lag_err_min, lag_err_max,
                                                                                mass_rms_sigma, mass_min_rms_sigma,
                                                                                mass_max_rms_sigma))

            if makeFig == True:
                # Define figure and axis for mean and rms spectrum
                fig_width, ax_width = ozplot.plot_share_x(2, source, "Wavelength ($\AA$)", ["Mean Flux", "RMS Flux"],
                                                   [contMin[1], contMax[0]])
                ax_width[0].plot(wave, fluxCoadd, color='black')
                ax_width[0].axvline(wave[lineMin], color='forestgreen')
                ax_width[0].axvline(wave[lineMax], color='forestgreen')
                ax_width[1].plot(wave, rms, color='black')
                ax_width[1].axvline(wave[lineMin], color='forestgreen')
                ax_width[1].axvline(wave[lineMax], color='forestgreen')
                fig_width.savefig(outLoc + source + "_" + line + "_width.png")
                plt.close(fig_width)

    out.close()

    return


# -------------------------------------------------- #
# -------------------- findRes ----------------------#
# -------------------------------------------------- #
# The line width measurements are dependent on the   #
# resolution of the spectrograph.  The OzDES spectra #
# are made up of two arms of AAOmega with different  #
# resolutions.  This function will find the          #
# resolution at the emission line in question.  You  #
# will need to modify this if you are using a        #
# different spectrograph.  Input rest frame emission #
# line wavelength and convert.                       #
# -------------------------------------------------- #
def findRes(line, z):
    #Use OzDES data - splice 5700 and resolution for red/blue arms
    splice = 5700
    resO = [1600, 1490]  #blue/red arm of spectrograph resolution
    obsLine = line*(1+z)
    if obsLine < splice:
        dL = obsLine/resO[0]
    else:
        dL = obsLine/resO[1]
    return dL


# --------------------------------------------------- #
# ---------------- comoving_distance ---------------- #
# --------------------------------------------------- #
# Function to calculate the comoving distance at a    #
# given redshift. Written by Harry Hobson.            #
# --------------------------------------------------- #
def comoving_distance(z):
    # returns the comoving distance in Mpc
    # c in km/s
    c = 299792.458
    # H0 in km/s/Mpc
    H0 = 70.0

    f_E = lambda x: 1.0 / np.sqrt(0.3 * (1 + x) ** 3 + 0.7)
    d_C = c / H0 * fixed_quad(f_E, 0.0, z, n=500)[0]

    return d_C


# --------------------------------------------------- #
# ------------------- luminosity -------------------- #
# --------------------------------------------------- #
# Calculates the lambda L_lambda luminosity for the   #
# specified wavelength and gives uncertainty via      #
# bootstrapping.  If the luminosity is not present in #
# the spectrum return nan.                            #
# --------------------------------------------------- #
def luminosity(wavelength, flux, variance, z, lum, strapNum, scale):
    # Check if the luminosity windows used (lum +/- 10 A in observed frame) are present in the spectrum.  If not return
    # nan for the luminosity
    if wavelength[0] < lum - 10 and lum + 10 < wavelength[-1]:
        lumBin = findBin(lum, wavelength)

        # calculate the mean flux around the specified luminosity
        fluxV = np.nanmean(flux[lumBin-2:lumBin+2]) * scale

        # calculate the range of fluxes based on bootstrapping
        flux_std = Lum_uncertainty(wavelength, flux, variance, lum, strapNum, scale)

        # scale by luminosity - we want lambda L_lambda
        fluxV = fluxV*lum
        flux_std = flux_std*lum

        # flux should be in erg/s/cm^2 the above statement gets rid of the angstroms
        d_C = comoving_distance(z)
        d_L = (1.0 + z) * d_C

        # convert d_L from Mpc to cm
        d_L *= 3.0857E24

        # scale factor used for uncertainty propogation
        scalefact = 4. * np.pi * d_L ** 2
        L = fluxV * scalefact
        L_std = flux_std * scalefact

        # calculate log Luminosity and error
        lgL = np.log10(L)
        err = lgL- np.log10(L-L_std)
    else:
        lgL = np.nan
        err = np.nan
    return lgL, err


# --------------------------------------------------- #
# ---------------- Lum_uncertainty ------------------ #
# --------------------------------------------------- #
# Calculates the uncertainty due to flux resampling   #
# and shifting luminosity window.                     #
# --------------------------------------------------- #
def Lum_uncertainty(wavelength, flux, variance, lum, strapNum, scale):
    # Performs bootstrap resampling in the range of potentially clean continuum to determine
    # 10 Angstroms on either size of luminosity
    nBins = len(wavelength)
    winLim = [findBin(lum-10, wavelength), findBin(lum+10, wavelength)]

    # vector of wavelengths within winLim spaced by 1 Angstrom
    winVect = np.arange(winLim[0], winLim[1]+1, 1)

    # Array of random continuum window starting points
    randVect = len(winVect)*np.random.rand(strapNum)
    randVect = randVect.astype(int)

    fluxes = np.zeros(strapNum)

    # For each iteration do flux resampling and calculate the line flux and shift window slightly
    for i in range(strapNum):
        varC = np.copy(variance)
        fluxC = np.zeros(nBins)
        for w in range(nBins):
            err = varC[w] ** 0.5
            fluxC[w] = np.random.normal(flux[w], err)

        fluxes[i] = np.nanmean(fluxC[winVect[randVect[i]] - 2:winVect[randVect[i]] + 2]) * scale

    return np.nanstd(fluxes)


# --------------------------------------------------- #
# -------------------- RL_CIV ----------------------- #
# --------------------------------------------------- #
# Radius Luminosity using CIV line and L1350 from     #
# Hoormann et al 2019.  L and L_std are log_10.       #
# --------------------------------------------------- #
def RL_CIV(L, L_std):
    # From Hoormann et al 2019 using L1350
    lag = pow(10, 0.81 + 0.47 * (L - 44))
    lag_err_p = abs(pow(10, (0.81 + 0.09) + (0.47 + 0.03) * ((L + L_std) - 44)) - lag)
    lag_err_m = abs(pow(10, (0.81 - 0.09) + (0.47 - 0.03) * ((L - L_std) - 44)) - lag)

    return lag, lag_err_p, lag_err_m


# --------------------------------------------------- #
# -------------------- RL_MgII ---------------------- #
# --------------------------------------------------- #
# Radius Luminosity using MgII line and L3000 from    #
# Trakhenbrot & Netzer 2012 best fit BCES method.     #
# L and L_std are log_10.                             #
# --------------------------------------------------- #
def RL_MgII(L, L_std):
    lag = pow(10, 1.34 + 0.615 * (L - 44))
    lag_err_p = abs(pow(10, (1.34 + 0.019) + (0.615 + 0.014) * ((L + L_std) - 44)) - lag)
    lag_err_m = abs(pow(10, (1.34 - 0.019) + (0.615 - 0.014) * ((L - L_std) - 44)) - lag)

    return lag, lag_err_p, lag_err_m


# --------------------------------------------------- #
# -------------------- RL_Hbeta --------------------- #
# --------------------------------------------------- #
# Radius Luminosity using Hbeta line and L5100 from   #
# Bentz et al 2013.  L and L_std are log_10.          #
# --------------------------------------------------- #
def RL_Hbeta(L, L_std):
    lag = pow(10, 1.527 + 0.533 * (L - 44))
    lag_err_p = abs(pow(10, (1.527 + 0.031) + (0.533 + 0.035) * ((L + L_std) - 44)) - lag)
    lag_err_m = abs(pow(10, (1.527 - 0.031) + (0.533 - 0.033) * ((L - L_std) - 44)) - lag)

    return lag, lag_err_p, lag_err_m


# --------------------------------------------------- #
# ------------------ blackHoleMass ------------------ #
# --------------------------------------------------- #
# Given a lag and velocity calculate the black hole   #
# mass.  Given in units of 10^9 Solar Masses.         #
# --------------------------------------------------- #
def blackHoleMass(lag, lErrMin, lErrMax, velocity, vErr):
    # convert everything to cgs
    G = 6.67*10**-11
    c = 2.998*10**8
    Msun = 1.989*10**30
    lag = lag*86400
    lErrMin = lErrMin*86400
    lErrMax = lErrMax*86400
    velocity = velocity*1000
    vErr = vErr*1000

    # Define f factor
    f = 4.47
    ferr = 1.25  #Woo et al 2014

    # Calculate Mass
    mass = f*(pow(velocity, 2)*c*lag/G)/Msun/10**9

    sigmaMin = mass*pow((ferr/f)**2 + (2*vErr/velocity)**2 + (lErrMin/lag)**2 ,0.5)

    sigmaMax = mass*pow((ferr/f)**2 + (2*vErr/velocity)**2 + (lErrMax/lag)**2 ,0.5)

    return mass, sigmaMin, sigmaMax

# -------------------------------------------------- #
# Modified from a function originally provided by    #
# Anthea King                                        #
# -------------------------------------------------- #
# ------------------ Spectrumv18 ------------------- #
# -------------------------------------------------- #
# Read in spectral data assuming the format from v18 #
# of the OzDES reduction pipeline. Modify if your    #
# input data is stored differently                   #
# -------------------------------------------------- #

class Spectrumv18(object):
    def __init__(self, filepath=None):
        assert filepath is not None
        self.filepath = filepath
        try:
            self.data = fits.open(filepath)
        except IOError:
            print("Error: file {0} could not be found".format(filepath))
            exit()
        data = fits.open(filepath)
        self.combinedFlux = data[0]
        self.combinedVariance = data[1]
        self.combinedPixels = data[2]
        self.numEpochs = int((np.size(data) - 3) / 3)
        self.field = self.data[3].header['SOURCEF'][19:21]
        self.cdelt1 = self.combinedFlux.header['cdelt1']  # Wavelength interval between subsequent pixels
        self.crpix1 = self.combinedFlux.header['crpix1']
        self.crval1 = self.combinedFlux.header['crval1']
        self.n_pix = self.combinedFlux.header['NAXIS1']
        self.RA = self.combinedFlux.header['RA']
        self.DEC = self.combinedFlux.header['DEC']

        self.fluxCoadd = self.combinedFlux.data
        self.varianceCoadd = self.combinedVariance.data
        self.badpixCoadd = self.combinedPixels.data

        self._wavelength = None
        self._flux = None
        self._variance = None
        self._badpix = None
        self._dates = None
        self._run = None
        self._ext = None
        self._qc = None
        self._exposed = None

    @property
    def wavelength(self):
        """Define wavelength solution."""
        if getattr(self, '_wavelength', None) is None:
            wave = ((np.arange(self.n_pix) - self.crpix1) * self.cdelt1) + self.crval1
            self._wavelength = wave
        return self._wavelength

    @property
    def flux(self):
        if getattr(self, '_flux', None) is None:
            self._flux = np.zeros((5000, self.numEpochs), dtype=float)
            for i in range(self.numEpochs):
                self._flux[:, i] = self.data[i * 3 + 3].data
        return self._flux

    @property
    def variance(self):
        if getattr(self, '_variance', None) is None:
            self._variance = np.zeros((5000, self.numEpochs), dtype=float)
            for i in range(self.numEpochs):
                self._variance[:, i] = self.data[i * 3 + 4].data
        return self._variance

    @property
    def badpix(self):
        if getattr(self, '_badpix', None) is None:
            self._badpix = np.zeros((5000, self.numEpochs), dtype=float)
            for i in range(self.numEpochs):
                self._badpix[:, i] = self.data[i * 3 + 5].data
        return self._badpix

    @property
    def dates(self):
        if getattr(self, '_dates', None) is None:
            self._dates = np.zeros(self.numEpochs, dtype=float)
            for i in range(self.numEpochs):
                self._dates[i] = round(self.data[i * 3 + 3].header['UTMJD'],3)
                # this give Modified Julian Date (UTC) that observation was taken
        return self._dates


    @property
    def ext(self):
        if getattr(self, '_ext', None) is None:
            self._ext = []
            for i in range(self.numEpochs):
                self._ext.append(i * 3 + 3)  # gives the extension in original fits file
        return self._ext

    @property
    def run(self):
        if getattr(self, '_run', None) is None:
            self._run = []
            for i in range(self.numEpochs):
                source = self.data[i * 3 + 3].header['SOURCEF']
                self._run.append(int(source[3:6]))  # this gives the run number of the observation
        return self._run

    @property
    def qc(self):
        if getattr(self, '_qc', None) is None:
            self._qc = []
            for i in range(self.numEpochs):
                self._qc.append(self.data[i * 3 + 3].header['QC'])
                # this tell you if there were any problems with the spectra that need to be masked out
        return self._qc

    @property
    def exposed(self):
        if getattr(self, '_exposed', None) is None:
            self._exposed = []
            for i in range(self.numEpochs):
                self._exposed.append(self.data[i * 3 + 3].header['EXPOSED'])
                # this will give you the exposure time of each observation
        return self._exposed


# -------------------------------------------------- #
# ------------------- calibSpec -------------------- #
# -------------------------------------------------- #
# This function does the bulk of the work.  It will  #
# 1) determine extensions which can be calibrated    #
# 2) calculate the scale factors                     #
# 3) calculate the warping function                  #
# 4) output new fits file with scaled spectra        #
# -------------------------------------------------- #

def calibSpec(obj_name, spectra, photo, spectraName, photoName, outBase, bands, filters, centers, plotFlag, coaddFlag,
              redshift):
    # Assumes scaling given is of the form
    # gScale = scaling[0,:]   gError = scaling[3,:]
    # rScale = scaling[1,:]   rError = scaling[4,:]
    # iScale = scaling[2,:]   iError = scaling[5,:]
    # inCoaddWeather = scaling[6,:]
    # inCoaddPhoto = scaling[7,:]
    # gMag = scaling[8,:]   gMagErr = scaling[9,:]
    # rMag = scaling[10,:]  rMagErr = scaling[11,:]
    # iMag = scaling[12,:]  iMagErr = scaling[13,:]

    # First we decide which extensions are worth scaling
    extensions, noPhotometry, badQC = prevent_Excess(spectra, photo, bands)

    # Then we calculate the scale factors
    nevermind, scaling = scaling_Matrix(spectra, extensions, badQC, noPhotometry, photo, bands, filters)

    # Remove last minute trouble makers
    extensions = [e for e in extensions if e not in nevermind]
    badQC = badQC + nevermind

    # And finally warp the data
    for s in extensions:
        # scale the spectra
        if plotFlag != False:
            plotName = plotFlag + obj_name + "_" + str(s)
        else:
            plotName = False
        spectra.flux[:, s], spectra.variance[:, s] = warp_spectra(scaling[0:3, s], scaling[3:6, s], spectra.flux[:, s],
                                                                  spectra.variance[:, s], spectra.wavelength, centers,
                                                                  plotName)
    if coaddFlag == False:
        create_output_single(obj_name, extensions, scaling, spectra, noPhotometry, badQC, spectraName, photoName,
                             outBase, redshift)
    elif coaddFlag in ['Run', 'Date']:
        coadd_output(obj_name, extensions, scaling, spectra, noPhotometry, badQC, spectraName, photoName, outBase,
                     plotFlag, coaddFlag, redshift)
    else:
        print("What do you want me to do with this data? Please specify output type.")

    return

# -------------------------------------------------- #
# ---------------- prevent_Excess ------------------ #
# -------------------------------------------------- #
# This function removes extensions from the list to  #
# calibrate because of insufficient photometric data #
# or bad quality flags                               #
# -------------------------------------------------- #

def prevent_Excess(spectra, photo, bands):

    # First, find the min/max date for which we have photometry taken on each side of the spectroscopic observation
    # This will be done by finding the highest date for which we have photometry in each band
    # and taking the max/min of those values
    # This is done because we perform a linear interpolation between photometric data points to estimate the magnitudes
    # observed at the specific time of the spectroscopic observation

    maxPhot = np.zeros(3)

    for e in range(len(photo['Date'][:])):
        if photo['Band'][e] == bands[0]:
            if photo['Date'][e] > maxPhot[0]:
                maxPhot[0] = photo['Date'][e]
        if photo['Band'][e] == bands[1]:
            if photo['Date'][e] > maxPhot[1]:
                maxPhot[1] = photo['Date'][e]
        if photo['Band'][e] == bands[2]:
            if photo['Date'][e] > maxPhot[2]:
                maxPhot[2] = photo['Date'][e]
    photLim = min(maxPhot)

    minPhot = np.array([100000, 100000, 100000])
    for e in range(len(photo['Date'][:])):
        if photo['Band'][e] == bands[0]:
            if photo['Date'][e] < minPhot[0]:
                minPhot[0] = photo['Date'][e]
        if photo['Band'][e] == bands[1]:
            if photo['Date'][e] < minPhot[1]:
                minPhot[1] = photo['Date'][e]
        if photo['Band'][e] == bands[2]:
            if photo['Date'][e] < minPhot[2]:
                minPhot[2] = photo['Date'][e]
    photLimMin = max(minPhot)
    noPhotometry = []
    badQC = []

    allowedQC = ['ok', 'backup']

    for s in range(spectra.numEpochs):
        # Remove data with insufficient photometry
        if spectra.dates[s] > photLim:
            noPhotometry.append(s)
        if spectra.dates[s] < photLimMin:
            noPhotometry.append(s)
        # Only allow spectra with quality flags 'ok' and 'backup'
        if spectra.qc[s] not in allowedQC:

            badQC.append(s)

    extensions = []

    # Make a list of extensions which need to be analyzed
    for s in range(spectra.numEpochs):
        if s not in noPhotometry and s not in badQC:
            extensions.append(s)

    return extensions, noPhotometry, badQC

# -------------------------------------------------- #
# ---------------- scaling_Matrix ------------------ #
# -------------------------------------------------- #
# finds the nearest photometry and interpolates mags #
# to find values at the time of the spectroscopic    #
# observations.  Calculates the mag that would be    #
# observed from the spectra and calculates the scale #
# factor to bring them into agreement. Saves the     #
# data in the scaling matrix.                        #
# -------------------------------------------------- #

def scaling_Matrix(spectra, extensions, badQC, noPhotometry, photo, bands, filters):
    # scale factors for each extension saved in the following form
    # gScale = scaling[0,:]   gError = scaling[3,:]
    # rScale = scaling[1,:]   rError = scaling[4,:]
    # iScale = scaling[2,:]   iError = scaling[5,:]
    # inCoaddWeather = scaling[6,:]
    # inCoaddPhoto = scaling[7,:]
    # gMag = scaling[8,:]   gMagError = scaling[9,:] (interpolated from neighbouring observations)
    # rMag = scaling[10,:]   rMagError = scaling[11,:]
    # iMag = scaling[12,:]   iMagError = scaling[13,:]

    scaling = np.zeros((14, spectra.numEpochs))

    # Judge goodness of spectra
    for e in range(spectra.numEpochs):
        if e in badQC:
            scaling[6, e] = False
        else:
            scaling[6, e] = True
        if e in noPhotometry:
            scaling[7, e] = False
        else:
            scaling[7, e] = True

    ozdesPhoto = np.zeros((3, spectra.numEpochs))
    desPhoto = np.zeros((3, spectra.numEpochs))

    ozdesPhotoU = np.zeros((3, spectra.numEpochs))
    desPhotoU = np.zeros((3, spectra.numEpochs))

    filterCurves = readFilterCurves(bands, filters)

    nevermind = []

    for e in extensions:
        # Find OzDES photometry

        ozdesPhoto[0, e], ozdesPhotoU[0, e] = computeABmag(filterCurves[bands[0]].trans, filterCurves[bands[0]].wave,
                                                           spectra.wavelength, spectra.flux[:, e],
                                                           spectra.variance[:, e])
        ozdesPhoto[1, e], ozdesPhotoU[1, e] = computeABmag(filterCurves[bands[1]].trans, filterCurves[bands[1]].wave,
                                                           spectra.wavelength, spectra.flux[:, e],
                                                           spectra.variance[:, e])
        ozdesPhoto[2, e], ozdesPhotoU[2, e] = computeABmag(filterCurves[bands[2]].trans, filterCurves[bands[2]].wave,
                                                           spectra.wavelength, spectra.flux[:, e],
                                                           spectra.variance[:, e])

        # Sometimes the total flux in the band goes zero and this obviously creates issues further down the line and
        # is most noticeable when the calculated magnitude is nan.  Sometimes it is because the data is very noisy
        # or the occasional negative spectrum is a known artifact of the data, more common in early OzDES runs.  In the
        # case where the observation doesn't get cut based on quality flag it will start getting ignored here.  The runs
        # ignored will eventually be saved with the badQC extensions.

        if np.isnan(ozdesPhoto[:, e]).any() == True:
            nevermind.append(e)

        # Find DES photometry
        desPhoto[:, e], desPhotoU[:, e] = des_photo(photo, spectra.dates[e], bands)

        scaling[8, e] = desPhoto[0, e]
        scaling[10, e] = desPhoto[1, e]
        scaling[12, e] = desPhoto[2, e]

        scaling[9, e] = desPhotoU[0, e]
        scaling[11, e] = desPhotoU[1, e]
        scaling[13, e] = desPhotoU[2, e]

        # Find Scale Factor
        scaling[0, e], scaling[3, e] = scale_factors(desPhoto[0, e] - ozdesPhoto[0, e],
                                                     desPhotoU[0, e] + ozdesPhotoU[0, e])
        scaling[1, e], scaling[4, e] = scale_factors(desPhoto[1, e] - ozdesPhoto[1, e],
                                                     desPhotoU[1, e] + ozdesPhotoU[1, e])
        scaling[2, e], scaling[5, e] = scale_factors(desPhoto[2, e] - ozdesPhoto[2, e],
                                                     desPhotoU[2, e] + ozdesPhotoU[2, e])

    return nevermind, scaling


# -------------------------------------------------- #
# --------------- interpolatePhot  ----------------- #
# -------------------------------------------------- #
# Performs linear interpolation and propagates the   #
# uncertainty to return you a variance.              #
# -------------------------------------------------- #
def interpolatePhot(x, y, s, val):
    # takes sigma returns variance
    # x - x data points (list)
    # y - y data points (list)
    # s - sigma on y data points (list)
    # val - x value to interpolate to (number)

    mag = y[0] + (val - x[0]) * (y[1] - y[0]) / (x[1] - x[0])

    err = s[0] ** 2 + (s[0] ** 2 + s[1] ** 2) * ((val - x[0]) / (x[1] - x[0])) ** 2

    return mag, err

# -------------------------------------------------- #
# ------------------ des_photo  -------------------- #
# -------------------------------------------------- #
# Finds nearest photometry on both sides of spectral #
# observations and interpolates to find value at the #
# time of the spectral observation                   #
# -------------------------------------------------- #

def des_photo(photo, spectral_mjd, bands):

    """Takes in an mjd from the spectra, looks through a light curve file to find the nearest photometric epochs and
    performs linear interpolation to get estimate at date, return the photo mags."""

    # Assumes dates are in chronological order!!!

    for l in range(len(photo['Date']) - 1):
        if photo['Band'][l] == bands[0] and photo['Date'][l] < spectral_mjd < photo['Date'][l + 1]:
            g_date_v = np.array([photo['Date'][l], photo['Date'][l + 1]])
            g_mag_v = np.array([photo['Mag'][l], photo['Mag'][l + 1]])
            g_err_v = np.array([photo['Mag_err'][l], photo['Mag_err'][l + 1]])
        if photo['Band'][l] == bands[1] and photo['Date'][l] < spectral_mjd < photo['Date'][l + 1]:
            r_date_v = np.array([photo['Date'][l], photo['Date'][l + 1]])
            r_mag_v = np.array([photo['Mag'][l], photo['Mag'][l + 1]])
            r_err_v = np.array([photo['Mag_err'][l], photo['Mag_err'][l + 1]])
        if photo['Band'][l] == bands[2] and photo['Date'][l] < spectral_mjd < photo['Date'][l + 1]:
            i_date_v = np.array([photo['Date'][l], photo['Date'][l + 1]])
            i_mag_v = np.array([photo['Mag'][l], photo['Mag'][l + 1]])
            i_err_v = np.array([photo['Mag_err'][l], photo['Mag_err'][l + 1]])

    g_mag, g_mag_err = interpolatePhot(g_date_v, g_mag_v, g_err_v, spectral_mjd)
    r_mag, r_mag_err = interpolatePhot(r_date_v, r_mag_v, r_err_v, spectral_mjd)
    i_mag, i_mag_err = interpolatePhot(i_date_v, i_mag_v, i_err_v, spectral_mjd)

    return [g_mag, r_mag, i_mag], [g_mag_err, r_mag_err, i_mag_err]


# -------------------------------------------------- #
# ---------------- scale_factors  ------------------ #
# -------------------------------------------------- #
# Calculates the scale factor and variance needed to #
# change spectroscopically derived magnitude to the  #
# observed photometry.                               #
# -------------------------------------------------- #

def scale_factors(mag_diff, mag_diff_var):
    # takes and returns variance

    flux_ratio = np.power(10., 0.4 * mag_diff)  # f_synthetic/f_photometry
    scale_factor = (1. / flux_ratio)
    scale_factor_sigma = mag_diff_var * (scale_factor * 0.4 * 2.3) ** 2   # ln(10) ~ 2.3

    return scale_factor, scale_factor_sigma

# -------------------------------------------------- #
# ----------------- warp_spectra  ------------------ #
# -------------------------------------------------- #
# Fits polynomial to scale factors and estimates     #
# associated uncertainties with gaussian processes.  #
# If the plotFlag variable is not False it will save #
# some diagnostic plots.                             #
# -------------------------------------------------- #

def warp_spectra(scaling, scaleErr, flux, variance, wavelength, centers, plotFlag):

    # associate scale factors with centers of bands and fit 2D polynomial to form scale function.
    scale = InterpolatedUnivariateSpline(centers, scaling, k=2)
    fluxScale = flux * scale(wavelength)

    # add in Gaussian process to estimate uncertainties, /10**-17 because it gets a bit panicky if you use small numbers
    stddev = (scaleErr ** 0.5) / 10 ** -17
    scale_v = scaling / 10 ** -17

    kernel = kernels.RBF(length_scale=300, length_scale_bounds=(.01, 2000.0))

    gp = GaussianProcessRegressor(kernel=kernel, alpha=stddev**2)

    xprime = np.atleast_2d(centers).T
    yprime = np.atleast_2d(scale_v).T

    gp.fit(xprime, yprime)
    xplot_prime = np.atleast_2d(wavelength).T
    y_pred, sigma = gp.predict(xplot_prime, return_std=True)

    y_pred = y_pred[:,0]

    sigModel = (sigma/y_pred)*scale(wavelength)

    # now scale the original variance and combine with scale factor uncertainty
    varScale = variance * pow(scale(wavelength), 2) + sigModel ** 2


    if plotFlag != False:
        figa, ax1a, ax2a = ozplot.makeFigDouble(plotFlag, "Wavelength ($\AA$)", "f$_\lambda$ (arbitrary units)",
                                      "f$_\lambda$ (10$^{-17}$ erg/s/cm$^2$/$\AA$)", [wavelength[0], wavelength[-1]])

        ax1a.plot(wavelength, flux, color='black', label="Before Calibration")
        ax1a.legend(loc=1, frameon=False, prop={'size': 20})
        ax2a.plot(wavelength, fluxScale / 10 ** -17, color='black', label="After Calibration")
        ax2a.legend(loc=1, frameon=False, prop={'size': 20})
        plt.savefig(plotFlag + "_beforeAfter.png")
        plt.close(figa)


        figb, ax1b, ax2b = ozplot.makeFigDouble(plotFlag, "Wavelength ($\AA$)", "f$_\lambda$ (10$^{-17}$ erg/s/cm$^2$/$\AA$)",
                                         "% Uncertainty", [wavelength[0], wavelength[-1]])
        ax1b.plot(wavelength, fluxScale / 10 ** -17, color='black')

        ax2b.plot(wavelength, 100*abs(pow(varScale, 0.5)/fluxScale), color='black', linestyle='-', label='Total')
        ax2b.plot(wavelength, 100*abs(sigModel/fluxScale), color='blue', linestyle='-.', label='Warping')
        ax2b.legend(loc=1, frameon=False, prop={'size': 20})
        ax2b.set_ylim([0, 50])
        plt.savefig(plotFlag + "_uncertainty.png")
        plt.close(figb)


        figc, axc = ozplot.makeFigSingle(plotFlag, "Wavelength ($\AA$)", "Scale Factor (10$^{-17}$ erg/s/cm$^2$/$\AA$/counts)")
        axc.plot(wavelength, scale(wavelength)/10**-17, color='black')
        axc.errorbar(centers, scaling/10**-17, yerr=stddev, fmt='s', color='mediumblue')
        plt.savefig(plotFlag + "_scalefactors.png")
        plt.close(figc)


    return fluxScale, varScale

# -------------------------------------------------- #
# ------------ create_output_single  --------------- #
# -------------------------------------------------- #
# Outputs the warped spectra to a new fits file.     #
# -------------------------------------------------- #


def create_output_single(obj_name, extensions, scaling, spectra, noPhotometry, badQC, spectraName, photoName, outBase,
                         redshift):

    outName = outBase + obj_name + "_scaled.fits"
    print("Saving Data to " + outName)

    hdulist = fits.HDUList(fits.PrimaryHDU())

    noPhotometryExt = []
    if len(noPhotometry) > 0:
        for i in range(len(noPhotometry)):
            noPhotometryExt.append(spectra.ext[noPhotometry[i]])

    badQCExt = []
    if len(badQC) > 0:
        for i in range(len(badQC)):
            badQCExt.append(spectra.ext[badQC[i]])

    index = 0
    # Create an HDU for each night
    for i in extensions:
        header = fits.Header()
        header['SOURCE'] = obj_name
        header['RA'] = spectra.RA
        header['DEC'] = spectra.DEC
        header['FIELD'] = spectra.field
        header['CRPIX1'] = spectra.crpix1
        header['CRVAL1'] = spectra.crval1
        header['CDELT1'] = spectra.cdelt1
        header['CTYPE1'] = 'wavelength'
        header['CUNIT1'] = 'angstrom'
        header['EPOCHS'] = len(extensions)
        header['z'] = redshift[0]

        # save the names of the input data and the extensions ignored
        header['SFILE'] = spectraName
        header['PFILE'] = photoName
        header['NOPHOTO'] = ','.join(map(str, noPhotometryExt))
        header['BADQC'] = ','.join(map(str, badQCExt))

        # save the original spectrum's extension number and some other details
        header["EXT"] = spectra.ext[i]
        header["UTMJD"] = spectra.dates[i]
        header["EXPOSE"] = spectra.exposed[i]
        header["QC"] = spectra.qc[i]

        # save scale factors/uncertainties
        header["SCALEG"] = scaling[0, i]
        header["ERRORG"] = scaling[3, i]
        header["SCALER"] = scaling[1, i]
        header["ERRORR"] = scaling[4, i]
        header["SCALEI"] = scaling[2, i]
        header["ERRORI"] = scaling[5, i]

        # save photometry/uncertainties used to calculate scale factors
        header["MAGG"] = scaling[8, i]
        header["MAGUG"] = scaling[9, i]
        header["MAGR"] = scaling[10, i]
        header["MAGUR"] = scaling[11, i]
        header["MAGI"] = scaling[12, i]
        header["MAGUI"] = scaling[13, i]
        if index == 0:
            hdulist[0].header['SOURCE'] = obj_name
            hdulist[0].header['RA'] = spectra.RA
            hdulist[0].header['DEC'] = spectra.DEC
            hdulist[0].header['CRPIX1'] = spectra.crpix1
            hdulist[0].header['CRVAL1'] = spectra.crval1
            hdulist[0].header['CDELT1'] = spectra.cdelt1
            hdulist[0].header['CTYPE1'] = 'wavelength'
            hdulist[0].header['CUNIT1'] = 'angstrom'
            hdulist[0].header['EPOCHS'] = len(extensions)

            # save the names of the input data and the extensions ignored
            hdulist[0].header['SFILE'] = spectraName
            hdulist[0].header['PFILE'] = photoName
            hdulist[0].header['NOPHOTO'] = ','.join(map(str, noPhotometryExt))
            hdulist[0].header['BADQC'] = ','.join(map(str, badQCExt))

            # save the original spectrum's extension number and some other details
            hdulist[0].header["EXT"] = spectra.ext[i]
            hdulist[0].header["UTMJD"] = spectra.dates[i]
            hdulist[0].header["EXPOSE"] = spectra.exposed[i]
            hdulist[0].header["QC"] = spectra.qc[i]

            # save scale factors/uncertainties
            hdulist[0].header["SCALEG"] = scaling[0, i]
            hdulist[0].header["ERRORG"] = scaling[3, i]
            hdulist[0].header["SCALER"] = scaling[1, i]
            hdulist[0].header["ERRORR"] = scaling[4, i]
            hdulist[0].header["SCALEI"] = scaling[2, i]
            hdulist[0].header["ERRORI"] = scaling[5, i]

            # save photometry/uncertainties used to calculate scale factors
            hdulist[0].header["MAGG"] = scaling[8, i]
            hdulist[0].header["MAGUG"] = scaling[9, i]
            hdulist[0].header["MAGR"] = scaling[10, i]
            hdulist[0].header["MAGUR"] = scaling[11, i]
            hdulist[0].header["MAGI"] = scaling[12, i]
            hdulist[0].header["MAGUI"] = scaling[13, i]
            hdulist[0].data = spectra.flux[:, i]
            hdulist.append(fits.ImageHDU(data=spectra.variance[:, i], header=header))
            hdulist.append(fits.ImageHDU(data=spectra.badpix[:, i], header=header))
            index = 2


        else:
            hdulist.append(fits.ImageHDU(data=spectra.flux[:, i], header=header))
            hdulist.append(fits.ImageHDU(data=spectra.variance[:, i], header=header))
            hdulist.append(fits.ImageHDU(data=spectra.badpix[:, i], header=header))
    hdulist.writeto(outName, overwrite=True)
    hdulist.close()

    return

# -------------------------------------------------- #
# ------------- create_output_coadd  --------------- #
# -------------------------------------------------- #
# Outputs the warped and coadded spectra to a new    #
# fits file.                                         #
# -------------------------------------------------- #


def create_output_coadd(obj_name, runList, fluxArray, varianceArray, badpixArray, extensions, scaling, spectra, redshift
                        ,badQC, noPhotometry, spectraName, photoName, outBase, coaddFlag):

    outName = outBase + obj_name + "_scaled_" + coaddFlag + ".fits"
    hdulist = fits.HDUList(fits.PrimaryHDU())

    noPhotometryExt = []
    if len(noPhotometry) > 0:
        for i in range(len(noPhotometry)):
            noPhotometryExt.append(spectra.ext[noPhotometry[i]])

    badQCExt = []
    if len(badQC) > 0:
        for i in range(len(badQC)):
            badQCExt.append(spectra.ext[badQC[i]])

    #print("Output Filename: %s \n" % (outName))
    # First save the total coadded spectrum for the source to the primary extension
    hdulist[0].data = fluxArray[:, 0]
    hdulist[0].header['CRPIX1'] = spectra.crpix1
    hdulist[0].header['CRVAL1'] = spectra.crval1
    hdulist[0].header['CDELT1'] = spectra.cdelt1
    hdulist[0].header['CTYPE1'] = 'wavelength'
    hdulist[0].header['CUNIT1'] = 'angstrom'
    hdulist[0].header['SOURCE'] = obj_name
    hdulist[0].header['RA'] = spectra.RA
    hdulist[0].header['DEC'] = spectra.DEC
    hdulist[0].header['FIELD'] = spectra.field
    hdulist[0].header['OBSNUM'] = len(runList)
    hdulist[0].header['z'] = redshift[0]
    hdulist[0].header['SFILE'] = spectraName
    hdulist[0].header['PFILE'] = photoName
    hdulist[0].header['METHOD'] = coaddFlag
    hdulist[0].header['NOPHOTO'] = ','.join(map(str, noPhotometryExt))
    hdulist[0].header['BADQC'] = ','.join(map(str, badQCExt))

    # First extension is the total coadded variance
    header = fits.Header()
    header['EXTNAME'] = 'VARIANCE'
    header['CRPIX1'] = spectra.crpix1
    header['CRVAL1'] = spectra.crval1
    header['CDELT1'] = spectra.cdelt1
    header['CTYPE1'] = 'wavelength'
    header['CUNIT1'] = 'angstrom'
    hdulist.append(fits.ImageHDU(data=varianceArray[:, 0], header=header))

    # Second Extension is the total bad pixel map
    header = fits.Header()
    header['EXTNAME'] = 'BadPix'
    header['CRPIX1'] = spectra.crpix1
    header['CRVAL1'] = spectra.crval1
    header['CDELT1'] = spectra.cdelt1
    header['CTYPE1'] = 'wavelength'
    header['CUNIT1'] = 'angstrom'
    hdulist.append(fits.ImageHDU(data=badpixArray[:, 0], header=header))

    # Create an HDU for each night
    index1 = 1
    for k in runList:
        index = 0
        date = 0
        header = fits.Header()
        header['CRPIX1'] = spectra.crpix1
        header['CRVAL1'] = spectra.crval1
        header['CDELT1'] = spectra.cdelt1
        header['CTYPE1'] = 'wavelength'
        header['CUNIT1'] = 'angstrom'
        header['RUN'] = k
        for i in extensions:
            here = False
            if coaddFlag == 'Run':
                if spectra.run[i] == k:
                    here = True

            if coaddFlag == 'Date':
                if int(spectra.dates[i]) == k:
                    here = True

            if here == True:
                head0 = "EXT" + str(index)
                header[head0] = spectra.ext[i]

                head1 = "UTMJD" + str(index)
                header[head1] = spectra.dates[i]
                date += spectra.dates[i]

                head2 = "EXPOSE" + str(index)
                header[head2] = spectra.exposed[i]

                head3 = "QC" + str(index)
                header[head3] = spectra.qc[i]

                head4 = "SCALEG" + str(index)
                header[head4] = scaling[0, i]

                head5 = "ERRORG" + str(index)
                header[head5] = scaling[3, i]

                head6 = "SCALER" + str(index)
                header[head6] = scaling[1, i]

                head7 = "ERRORR" + str(index)
                header[head7] = scaling[4, i]

                head8 = "SCALEI" + str(index)
                header[head8] = scaling[2, i]

                head9 = "ERRORI" + str(index)
                header[head9] = scaling[5, i]

                head10 = "MAGG" + str(index)
                header[head10] = scaling[8, i]

                head11 = "MAGUG" + str(index)
                header[head11] = scaling[9, i]

                head12 = "MAGR" + str(index)
                header[head12] = scaling[10, i]

                head13 = "MAGUR" + str(index)
                header[head13] = scaling[11, i]

                head14 = "MAGI" + str(index)
                header[head14] = scaling[12, i]

                head15 = "MAGUI" + str(index)
                header[head15] = scaling[13, i]

                index += 1

        if date > 0:
            header['OBSNUM'] = index
            header['AVGDATE'] = date / index

            hdu_flux = fits.ImageHDU(data=fluxArray[:, index1], header=header)
            hdu_fluxvar = fits.ImageHDU(data=varianceArray[:, index1], header=header)
            hdu_badpix = fits.ImageHDU(data=badpixArray[:, index1], header=header)
            hdulist.append(hdu_flux)
            hdulist.append(hdu_fluxvar)
            hdulist.append(hdu_badpix)
        index1 += 1

    hdulist.writeto(outName, overwrite=True)
    hdulist.close()

    return

# -------------------------------------------------- #
# ----------------- coadd_output  ------------------ #
# -------------------------------------------------- #
# Coadds the observations based on run or night.     #
# -------------------------------------------------- #


def coadd_output(obj_name, extensions, scaling, spectra, noPhotometry, badQC, spectraName, photoName, outBase, plotFlag,
                 coaddFlag, redshift):

    # Get a list of items (dates/runs) over which all observations will be coadded
    coaddOver = []

    for e in extensions:
        # OzDES runs 7,8 were close together in time and run 8 had bad weather so there was only observations of 1
        # field - coadd with run 7 to get better signal to noise
        if spectra.run[e] == 8:
            spectra.run[e] = 7

        if coaddFlag == 'Run':
            if spectra.run[e] not in coaddOver:
                coaddOver.append(spectra.run[e])

        if coaddFlag == 'Date':
            if int(spectra.dates[e]) not in coaddOver:
                coaddOver.append(int(spectra.dates[e]))


    coaddFlux = np.zeros((5000, len(coaddOver) + 1))
    coaddVar = np.zeros((5000, len(coaddOver) + 1))
    coaddBadPix = np.zeros((5000, len(coaddOver) + 1))

    speclistC = []  # For total coadd of observation
    index = 1

    for c in coaddOver:
        speclist = []
        for e in extensions:
            opt = ''
            if coaddFlag == 'Run':
                opt = spectra.run[e]
            if coaddFlag == 'Date':
                opt = int(spectra.dates[e])
            if opt == c:
                speclist.append(SingleSpec(obj_name, spectra.wavelength, spectra.flux[:,e], spectra.variance[:,e],
                                           spectra.badpix[:,e]))
                speclistC.append(SingleSpec(obj_name, spectra.wavelength, spectra.flux[:,e], spectra.variance[:,e],
                                            spectra.badpix[:,e]))

        if len(speclist) > 1:
            runCoadd = outlier_reject_and_coadd(obj_name, speclist)
            coaddFlux[:, index] = runCoadd.flux
            coaddVar[:, index] = runCoadd.fluxvar
            coaddVar[:, index] = runCoadd.fluxvar
            coaddBadPix[:,index] = runCoadd.isbad.astype('uint8')
        if len(speclist) == 1:
            coaddFlux[:, index] = speclist[0].flux
            coaddVar[:, index] = speclist[0].fluxvar
            coaddBadPix[:, index] = speclist[0].isbad.astype('uint8')
        index += 1

    if len(speclistC) > 1:
        allCoadd = outlier_reject_and_coadd(obj_name, speclistC)
        coaddFlux[:, 0] = allCoadd.flux
        coaddVar[:, 0] = allCoadd.fluxvar
        coaddBadPix[:, 0] = allCoadd.isbad.astype('uint8')
    if len(speclistC) == 1:
        coaddFlux[:, 0] = speclistC[0].flux
        coaddVar[:, 0] = speclistC[0].fluxvar
        coaddBadPix[:, 0] = speclistC[0].isbad.astype('uint8')

    mark_as_bad(coaddFlux, coaddVar)

    create_output_coadd(obj_name, coaddOver, coaddFlux, coaddVar, coaddBadPix, extensions, scaling, spectra, redshift,
                        badQC, noPhotometry, spectraName, photoName, outBase, coaddFlag)


    return

# -------------------------------------------------- #
# Modified from code originally provided by          #
# Harry Hobson                                       #
# -------------------------------------------------- #
# ------------------ mark_as_bad ------------------- #
# -------------------------------------------------- #
# Occasionally you get some big spikes in the data   #
# that you do not want messing with your magnitude   #
# calculations.  Remove these by looking at single   #
# bins that have a significantly 4.5 larger than     #
# average fluxes or variances and change those to    #
# nans. Nans will be interpolated over.  The         #
# threshold should be chosen to weigh removing       #
# extreme outliers and removing noise.               #
# -------------------------------------------------- #

def mark_as_bad(fluxes, variances):
    number = int(fluxes.size/fluxes.shape[0])
    for epoch in range(number):
        if number == 1:
            flux = fluxes[:]
            variance = variances[:]
        else:
            flux = fluxes[:, epoch]
            variance = variances[:, epoch]

        nBins = len(flux)
        # define the local average in flux and variance to compare outliers to
        for i in range(nBins):
            if i < 50:
                avg = np.nanmean(variance[0:99])
                avgf = np.nanmean(flux[0:99])
            elif i > nBins - 50:
                avg = np.nanmean(variance[i-50:nBins-1])
                avgf = np.nanmean(flux[i-50:nBins-1])
            else:
                avg = np.nanmean(variance[i-50:i+50])
                avgf = np.nanmean(flux[i-50:i+50])

            # find outliers and set that bin and the neighbouring ones to nan.

            if np.isnan(variance[i]) == False and variance[i] > 4.5*avg:

                flux[i] = np.nan
                if i > 2 and i < 4996:
                    flux[i - 1] = np.nan
                    flux[i - 2] = np.nan
                    flux[i - 3] = np.nan
                    flux[i + 1] = np.nan
                    flux[i + 2] = np.nan
                    flux[i + 3] = np.nan

            if np.isnan(flux[i]) == False and flux[i] > 4.5 * avgf:

                flux[i] = np.nan
                if i > 2 and i < 4996:
                    flux[i-1] = np.nan
                    flux[i-2] = np.nan
                    flux[i-3] = np.nan
                    flux[i+1] = np.nan
                    flux[i+2] = np.nan
                    flux[i+3] = np.nan

            if np.isnan(flux[i]) == False and flux[i] < -4.5 * avgf:

                flux[i] = np.nan
                if i > 2 and i < 4996:
                    flux[i-1] = np.nan
                    flux[i-2] = np.nan
                    flux[i-3] = np.nan
                    flux[i+1] = np.nan
                    flux[i+2] = np.nan
                    flux[i+3] = np.nan

        # interpolates nans (added here and bad pixels in the data)
        filter_bad_pixels(flux, variance)
    return

# -------------------------------------------------- #
# Modified from code originally provided by          #
# Harry Hobson                                       #
# -------------------------------------------------- #
# --------------- filter_bad_pixels ---------------- #
# -------------------------------------------------- #
# Interpolates over nans in the spectrum.            #
# -------------------------------------------------- #

def filter_bad_pixels(fluxes, variances):
    number = int(fluxes.size/fluxes.shape[0])
    for epoch in range(number):
        if (number == 1):
            flux = fluxes[:]
            variance = variances[:]
        else:
            flux = fluxes[:, epoch]
            variance = variances[:, epoch]

        nBins = len(flux)

        flux[0] = 0.0
        flux[-1] = 0.0
        variance[0] = 100*np.nanmean(variance)
        variance[-1] = 100*np.nanmean(variance)

        bad_pixels = np.logical_or.reduce((np.isnan(flux), np.isnan(variance), variance < 0))

        bin = 0
        binEnd = 0

        while (bin < nBins):
            if (bad_pixels[bin] == True):
                binStart = bin
                binNext = bin + 1
                while (binNext < nBins):
                    if bad_pixels[binNext] == False:
                        binEnd = binNext - 1
                        binNext = nBins
                    binNext = binNext + 1

                ya = float(flux[binStart - 1])
                xa = float(binStart - 1)
                sa = variance[binStart - 1]
                yb = flux[binEnd + 1]
                xb = binEnd + 1
                sb = variance[binEnd + 1]

                step = binStart
                while (step < binEnd + 1):
                    flux[step] = ya + (yb - ya) * (step - xa) / (xb - xa)
                    variance[step] = sa + (sb + sa) * ((step - xa) / (xb - xa)) ** 2
                    step = step + 1
                bin = binEnd
            bin = bin + 1
    return


# -------------------------------------------------- #
#  The following 4 functions were written by Chris   #
# Lidman, Mike Childress, and maybe others for the   #
# initial processing of the OzDES spectra.  They     #
# were taken from the DES_coaddSpectra.py functions. #
# -------------------------------------------------- #
# -------------------- OzExcept -------------------- #
# -------------------------------------------------- #
# -------------------------------------------------- #
# A simple exception class                           #
# -------------------------------------------------- #


class OzExcept(Exception):
    """
    Simple exception class
    """

    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return "{0}: {1}".format(self.__class__.__name__, msg)


# -------------------------------------------------- #
# ----------------- VerboseMessager ---------------- #
# -------------------------------------------------- #
# -------------------------------------------------- #
# Verbose messaging for routines below.              #
# -------------------------------------------------- #


class VerboseMessager(object):
    """
    Verbose messaging for routines below
    """

    def __init__(self, verbose=False):
        self.verbose = verbose

    def __call__(self, *args):
        if self.verbose:
            print("Something strange is happening")
            sys.stdout.flush()

# -------------------------------------------------- #
# ------------------- SingleSpec ------------------- #
# -------------------------------------------------- #
# -------------------------------------------------- #
# Class representing a single spectrum for analysis. #
# -------------------------------------------------- #
class SingleSpec(object):
    """
    Class representing a single spectrum for analysis
    """

    ## Added filename to SingleSpec
    def __init__(self, obj_name, wl, flux, fluxvar, badpix):

        self.name = obj_name
        # ---------------------------
        # self.pivot = int(fibrow[9])
        # self.xplate = int(fibrow[3])
        # self.yplate = int(fibrow[4])
        # self.ra = np.degrees(fibrow[1])
        # self.dec = np.degrees(fibrow[2])
        # self.mag=float(fibrow[10])
        # self.header=header

        self.wl = np.array(wl)
        self.flux = np.array(flux)
        self.fluxvar = np.array(fluxvar)

        # If there is a nan in either the flux, or the variance, mark it as bad

        # JKH: this was what was here originally, my version complains about it
        # self.fluxvar[fluxvar < 0] = np.nan

        for i in range(5000):
            if (self.fluxvar[i] < 0):
                self.fluxvar[i] = np.nan

        # The following doesn't take into account
        #self.isbad = np.any([np.isnan(self.flux), np.isnan(self.fluxvar)], axis=0)
        self.isbad = badpix.astype(bool)

# -------------------------------------------------- #
# ------------ outlier_reject_and_coadd ------------ #
# -------------------------------------------------- #
# -------------------------------------------------- #
# OzDES coadding function to reject outliers and     #
# coadd all of the spectra in the inputted list.     #
# -------------------------------------------------- #
def outlier_reject_and_coadd(obj_name, speclist):
    """
    Reject outliers on single-object spectra to be coadded.
    Assumes input spectra have been resampled to a common wavelength grid,
    so this step needs to be done after joining and resampling.

    Inputs
        speclist:  list of SingleSpec instances on a common wavelength grid
        show:  boolean; show diagnostic plot?  (debug only; default=False)
        savefig:  boolean; save diagnostic plot?  (debug only; default=False)
    Output
        result:  SingleSpec instance of coadded spectrum, with bad pixels
            set to np.nan (runz requires this)
    """

    # Edge cases
    if len(speclist) == 0:
        print("outlier_reject:  empty spectrum list")
        return None
    elif len(speclist) == 1:
        tgname = speclist[0].name
        vmsg("Only one spectrum, no coadd needed for {0}".format(tgname))
        return speclist[0]

    # Have at least two spectra, so let's try to reject outliers
    # At this stage, all spectra have been mapped to a common wavelength scale
    wl = speclist[0].wl
    tgname = speclist[0].name
    # Retrieve single-object spectra and variance spectra.
    flux_2d = np.array([s.flux for s in speclist])
    fluxvar_2d = np.array([s.fluxvar for s in speclist])
    badpix_2d = np.array([s.isbad for s in speclist])


    # Baseline parameters:
    #    outsig     Significance threshold for outliers (in sigma)
    #    nbin       Bin width for median rebinning
    #    ncoinc     Maximum number of spectra in which an artifact can appear
    outsig, nbin, ncoinc = 5, 25, 1
    nspec, nwl = flux_2d.shape

    # Run a median filter of the spectra to look for n-sigma outliers.
    # These incantations are kind of complicated but they seem to work
    # i) Compute the median of a wavelength section (nbin) along the observation direction
    # 0,1 : observation,wavelength, row index, column index
    # In moving to numpy v1.10.2, we replaced median with nanmedian
    fmed = np.reshape([np.nanmedian(flux_2d[:, j:j + nbin], axis=1)
                       for j in np.arange(0, nwl, nbin)], (-1, nspec)).T

    # Now expand fmed and flag pixels that are more than outsig off
    fmed_2d = np.reshape([fmed[:, int(j / nbin)] for j in np.arange(nwl)], (-1, nspec)).T

    resid = (flux_2d - fmed_2d) / np.sqrt(fluxvar_2d)
    # If the residual is nan, set flag_2d to 1
    nans = np.isnan(resid)

    flag_2d = np.zeros(nspec * nwl).reshape(nspec, nwl)
    flag_2d[nans] = 1
    flag_2d[~nans] = (np.abs(resid[~nans]) > outsig)

    # If a pixel is flagged in only one spectrum, it's probably a cosmic ray
    # and we should mark it as bad and add ito to badpix_2d.  Otherwise, keep it.
    # This may fail if we coadd many spectra and a cosmic appears in 2 pixels
    # For these cases, we could increase ncoinc
    flagsum = np.tile(np.sum(flag_2d, axis=0), (nspec, 1))
    # flag_2d, flagsum forms a tuple of 2 2d arrays
    # If flag_2d is true and if and flagsum <= ncoinc then set that pixel to bad.
    badpix_2d[np.all([flag_2d, flagsum <= ncoinc], axis=0)] = True


    # Remove bad pixels in the collection of spectra.  In the output they
    # must appear as NaN, but any wavelength bin which is NaN in one spectrum
    # will be NaN in the coadd.  So we need to set the bad pixel values to
    # something innocuous like the median flux, then set the weights of the
    # bad pixels to zero in the coadd.  If a wavelength bin is bad in all
    # the coadds, it's just bad and needs to be marked as NaN in the coadd.
    # In moving to numpy v1.10.2, we replaced median with nanmedian
    flux_2d[badpix_2d] = np.nanmedian(fluxvar_2d)
    fluxvar_2d[badpix_2d] = np.nanmedian(fluxvar_2d)
    badpix_coadd = np.all(badpix_2d, axis=0)
    # Derive the weights
    ## Use just the variance
    wi = 1.0 / (fluxvar_2d)
    # Set the weights of bad data to zero
    wi[badpix_2d] = 0.0
    # Why set the weight of the just first spectrum to np.nan?
    # If just one of the mixels is nan, then the result computed below is nan as well
    for i, val in enumerate(badpix_coadd):
        if val:  wi[0, i] = np.nan

    # Some coadd
    coaddflux = np.average(flux_2d, weights=wi, axis=0)
    coaddfluxvar = np.average(fluxvar_2d, weights=wi, axis=0) / nspec

    coaddflux[badpix_coadd] = np.nan
    coaddfluxvar[badpix_coadd] = np.nan

    # Return the coadded spectrum in a SingleSpectrum object
    return SingleSpec(obj_name, wl, coaddflux, coaddfluxvar, badpix_coadd)


# -------------------------------------------------- #
# --------------------- perVar --------------------- #
# -------------------------------------------------- #
# -------------------------------------------------- #
# Calculate percent variation of light curve.        #
# -------------------------------------------------- #
def perVar(mags):
    minM = min(mags)
    maxM = max(mags)

    return 100*(maxM-minM)/minM


# -------------------------------------------------- #
# --------------------- diffVar -------------------- #
# -------------------------------------------------- #
# -------------------------------------------------- #
# Calculate total variation of light curve.          #
# -------------------------------------------------- #
def diffVar(mags):
    minM = min(mags)
    maxM = max(mags)

    return maxM-minM