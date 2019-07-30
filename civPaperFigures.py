# ---------------------------------------------------------- #
# -------------------- civPaperFigures.py ------------------ #
# --------- https://github.com/jhoormann/RMCodeDump -------- #
# ---------------------------------------------------------- #
# This is all of the code I used to create the figures in    #
# OzDES Y4 CIV paper as well as the machine readable tables. #
# ---------------------------------------------------------- #
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.integrate import fixed_quad
import pickle
from scipy.stats import mode
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FormatStrFormatter
from scipy import stats
import matplotlib.cm as cm


class Spectrum(object):
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
                self._flux[:, i] = self.data[i * 3 + 3].data / 10 ** -16
        return self._flux

    @property
    def variance(self):
        if getattr(self, '_variance', None) is None:
            self._variance = np.zeros((5000, self.numEpochs), dtype=float)
            for i in range(self.numEpochs):
                self._variance[:, i] = self.data[i * 3 + 4].data / 10 ** -32
        return self._variance

    @property
    def fluxCoadd(self):
        if getattr(self, '_fluxCoadd', None) is None:
            self._fluxCoadd = np.zeros(5000, dtype=float)
            self._fluxCoadd[:] = self.data[0].data / 10 ** -16
        return self._fluxCoadd

    @property
    def varianceCoadd(self):
        if getattr(self, '_varianceCoadd', None) is None:
            self._varianceCoadd = np.zeros(5000, dtype=float)
            self._varianceCoadd[:] = self.data[1].data / 10 ** -32
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

def findBin(line, wavelength):
    bin = 0
    for i in range(4999):
        if(line >= wavelength[i] and line <= wavelength[i+1]):
            bin = i
            i = 5000
        if(line > wavelength[4999]):
            bin = 4999
            i=5000
    return bin


def meanUncert(variance):
    length = len(variance)

    var = 0
    num = 0
    for i in range(length):
        if np.isnan(variance[i]) == False:
            var = var + variance[i]
            num += 1

    sigma2 = (var / (num ** 2))

    return sigma2**0.5

def comoving_distance(z):
    # returns the comoving distance in Mpc
    # c in km/s
    c = 299792.458
    # H0 in km/s/Mpc
    H0 = 70.0

    f_E = lambda x: 1.0 / np.sqrt(0.3 * (1 + x) ** 3 + 0.7)
    d_C = c / H0 * fixed_quad(f_E, 0.0, z, n=500)[0]

    return d_C

def exponential_smooth(fluxes):
    """"Applies an exponential smoothing kernel to the data"""

    number = int(fluxes.size/fluxes.shape[0])

    #search_pixels = 3
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
def luminosity(flux, z, lum, flux_std):

    flux = flux*lum
    # flux should be in erg/s/cm^2 the above statement gets rid of the angstroms
    d_C = comoving_distance(z)
    d_L = (1.0 + z) * d_C
    # convert d_L from Mpc to cm
    d_L *= 3.0857E24
    # scale factor used for uncertainty propogation
    scale = 4. * np.pi * d_L ** 2
    L = flux * scale
    L_std = flux_std * scale

    err = np.log10(L)- np.log10(L-L_std)
    return np.log10(L), err

class Spectrumv14(object):
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
        self._redzp = None
        self._bluezp = None
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
    def redzp(self):
        if getattr(self, '_redzp', None) is None:
            self._redzp = []
            for i in range(self.numEpochs):
                self._redzp.append(self.data[i * 3 + 3].header['REDZP'])
                # this gives if the ZP for the red arm
        return self._redzp

    @property
    def bluezp(self):
        if getattr(self, '_bluezp', None) is None:
            self._bluezp = []
            for i in range(self.numEpochs):
                self._bluezp.append(self.data[i * 3 + 3].header['BLUZP'])
                # this gives if the ZP for the blue arm
        return self._bluezp

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

title_font = {'size':'22', 'color':'black', 'weight':'normal', 'verticalalignment':'bottom'}
axis_font = {'size':'22'}
def makeFigDouble(title, xlabel, ylabel1, ylabel2, xlim=[0, 0], ylim1=[0, 0], ylim2=[0, 0]):
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig = plt.gcf()
    fig.set_size_inches(10, 10, forward=True)
    fig.subplots_adjust(hspace=0)

    for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
        label.set_fontsize(23)
    for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
        label.set_fontsize(23)

    ax1.set_ylabel(ylabel1, **axis_font)
    if ylim1 != [0, 0] and ylim1[0] < ylim1[1]:
        ax1.set_ylim(ylim1)

    ax2.set_ylabel(ylabel2, **axis_font)
    if ylim2 != [0, 0] and ylim2[0] < ylim2[1]:
        ax2.set_ylim(ylim2)

    ax2.set_xlabel(xlabel, **axis_font)
    if xlim != [0, 0] and xlim[0] < xlim[1]:
        ax2.set_xlim(xlim)

    ax1.set_title(title, **title_font)
    return fig, ax1, ax2


def makeFigSingle(title, xlabel, ylabel, xlim=[0, 0], ylim=[0, 0]):
    fig = plt.figure()
    fig = plt.gcf()
    fig.set_size_inches(10, 10, forward=True)

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

def makeFigTriple(title, xlabel, ylabel1, ylabel2, ylabel3, xlim=[0, 0], ylim1=[0, 0], ylim2=[0, 0], ylim3=[0, 0]):
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    fig = plt.gcf()
    fig.set_size_inches(10, 12, forward=True)
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
def cont_fit_reject(wavelength, fluxes, variances, minWin, maxWin):
    wave = np.array([np.nanmean(minWin), np.nanmean(maxWin)])

    nBins = len(wavelength)

    number = int(fluxes.size / nBins)
    allConts = np.zeros((number, nBins))

    for epoch in range(number):

        if number == 1:
            flux = fluxes
            variance = variances
        else:
            flux = fluxes[:, epoch]
            variance = variances[:, epoch]

        fvals = np.array([np.nanmean(flux[findBin(minWin[0], wavelength):findBin(minWin[1], wavelength)]),
                          np.nanmean(flux[findBin(maxWin[0], wavelength):findBin(maxWin[1], wavelength)])])
        svals = np.array([meanUncert(variance[findBin(minWin[0], wavelength):findBin(minWin[1], wavelength)]),
                          meanUncert(variance[findBin(maxWin[0], wavelength):findBin(maxWin[1], wavelength)])])

        cont = np.zeros(nBins)
        contVar = np.zeros(nBins)

        for i in range(nBins):
            cont[i], contVar[i] = interpolateVals(wave, fvals, svals, wavelength[i])

        allConts[epoch, :] = cont

        flux -= cont
        variance += contVar

        minMid = findBin(wave[0], wavelength)
        maxMid = findBin(wave[1], wavelength)

    return wavelength[minMid:maxMid], cont[minMid:maxMid]

def interpolateVals(x, y, s, val):
    # uncertainty is variance

    interp = y[0] + (val - x[0]) * (y[1] - y[0]) / (x[1] - x[0])

    interp_var = s[0] + (s[0] + s[1]) * ((val - x[0]) / (x[1] - x[0])) ** 2.

    return interp, interp_var
def photoSplit(photo):

        gDate = []
        gMag = []
        gErr = []

        rDate = []
        rMag = []
        rErr = []

        iDate = []
        iMag = []
        iErr = []

        for i in range(len(photo['date'][:])):
            if photo['band'][i] == 'g':
                gDate.append(photo['date'][i])
                gMag.append(photo['mag'][i])
                gErr.append(photo['err'][i])
            if photo['band'][i] == 'r':
                rDate.append(photo['date'][i])
                rMag.append(photo['mag'][i])
                rErr.append(photo['err'][i])
            if photo['band'][i] == 'i':
                iDate.append(photo['date'][i])
                iMag.append(photo['mag'][i])
                iErr.append(photo['err'][i])


        return np.array(gDate), np.array(gMag), np.array(gErr), np.array(rDate), np.array(rMag), np.array(
            rErr), np.array(iDate), np.array(iMag), np.array(iErr)

def meanSpec(flux):

    length = len(flux[:,0])
    epochs = len(flux[0,:])

    meanFlux = np.zeros(length)

    for i in range(length):
        meanFlux[i] = np.nanmean(flux[i,:])

    return meanFlux

def rmsSpec(flux):
    length = len(flux[:, 0])
    epochs = len(flux[0, :])

    mean = meanSpec(flux)
    rms = np.zeros(length)

    for b in range(length):
        for e in range(epochs):
            rms[b] += (flux[b, e] - mean[b]) ** 2
        rms[b] = (rms[b] / (epochs - 1)) ** 0.5

    return rms
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

            if np.isnan(variance[i]) == False and variance[i] > 3.5*avg:
                flux[i] = np.nan
                if i > 2 and i < 4996:
                    flux[i - 1] = np.nan
                    flux[i - 2] = np.nan
                    flux[i - 3] = np.nan
                    flux[i + 1] = np.nan
                    flux[i + 2] = np.nan
                    flux[i + 3] = np.nan

            if np.isnan(flux[i]) == False and flux[i] > 4.5*avgf:
                flux[i] = np.nan
                if i > 2 and i < 4996:
                    flux[i-1] = np.nan
                    flux[i-2] = np.nan
                    flux[i-3] = np.nan
                    flux[i+1] = np.nan
                    flux[i+2] = np.nan
                    flux[i+3] = np.nan

            if np.isnan(flux[i]) == False and avgf > 4.5*flux[i]:
                flux[i] = np.nan
                if i > 2 and i < 4996:
                    flux[i-1] = np.nan
                    flux[i-2] = np.nan
                    flux[i-3] = np.nan
                    flux[i+1] = np.nan
                    flux[i+2] = np.nan
                    flux[i+3] = np.nan


        filter_bad_pixels(flux, variance)
    return
'''
def filter_bad_pixels(fluxes, variances, number):
    nBins = 5000
    for epoch in range(number):
        if (number == 1):
            flux = fluxes[:]
            variance = variances[:]
        else:
            flux = fluxes[:, epoch]
            variance = variances[:, epoch]

        flux[0] = 0.0
        flux[-1] = 0.0
        variance[0] = 0
        variance[-1] = 0

        bad_pixels = np.logical_or.reduce((np.isnan(flux), np.isnan(variance), variance < 0))

        bin = 0
        binStart = 0
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
'''


def filter_bad_pixels(fluxes, variances):
    number = int(fluxes.size / fluxes.shape[0])
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
        variance[0] = 100
        variance[-1] = 100

        bad_pixels = np.logical_or.reduce((np.isnan(flux), np.isnan(variance), variance < 0))

        bin = 0
        binStart = 0
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
# ---------------------------------- #
# ------- Redshift Histogram ------- #
# ---------------------------------- #
'''
sources, redshifts = np.loadtxt("RM_Quasars_z.txt", unpack=True)

fig, ax = makeFigSingle("", "Redshift", "Number", [0,4.5])

zs = np.arange(0., 4.5, 0.2)
#ax.axvspan(0, 0.81, alpha =1, hatch=3*'/', edgecolor='darkred', facecolor='none', label=r'H$\beta$')
#ax.axvspan(0.32, 2.15, alpha = 1, hatch=3*'.', edgecolor='mediumblue', facecolor='none', label='MgII')
#ax.axvspan(1.39, 4.5, alpha = 1, hatch=3*'X', edgecolor='forestgreen', facecolor='none', label='CIV')
ax.hist(redshifts, bins=zs, color = 'grey', edgecolor = 'black')
plt.legend(loc='upper right', prop={'size':20})

count = 0
for i in range(len(redshifts)):
    if redshifts[i] > 3:
        count += 1
print(count)

plt.show()
'''

# ---------------------------------- #
# ------- Magnitude Histogram ------ #
# ---------------------------------- #
'''
sources = np.loadtxt('../../OzDES_ReverberationMapping/OzDES_Data/RM_IDs.txt')
rMags = []

for i in range(len(sources)):
    agn = str(int(sources[i]))

    photoName = '../../OzDES_ReverberationMapping/OzDES_Data/photometryY4/' + agn + '_lc.dat'
    photo = np.loadtxt(photoName, dtype={'names': ('date', 'mag', 'err', 'band'),
                                         'formats': (np.float, np.float, np.float, '|S15')}, skiprows=1)

    gDate, gMag, gErr, rDate, rMag, rErr, iDate, iMag, iErr = photoSplit(photo)

    rMags.append(np.nanmean(gMag))

fig, ax = makeFigSingle("", "g-Band Magnitude", "Number", [16, 23])
rs = np.arange(16., 23, 0.3)
ax.hist(rMags, bins=rs, color = 'grey', edgecolor = 'black')
plt.show()
'''

# ---------------------------------- #
# ----- Photo Epochs Histogram ----- #
# ---------------------------------- #
'''
sources = np.loadtxt('../../OzDES_ReverberationMapping/OzDES_Data/RM_IDs.txt')
epochs = []


for i in range(len(sources)):
    agn = str(int(sources[i]))

    photoName = '../../OzDES_ReverberationMapping/OzDES_Data/photometryY4/' + agn + '_lc.dat'
    photo = np.loadtxt(photoName, dtype={'names': ('date', 'mag', 'err', 'band'),
                                         'formats': (np.float, np.float, np.float, '|S15')}, skiprows=1)

    gDate, gMag, gErr, rDate, rMag, rErr, iDate, iMag, iErr = photoSplit(photo)

    epochs.append(len(gMag))

fig, ax = makeFigSingle("", "Photometric Epochs", "Number", [0, 250])
rs = np.arange(0., 250, 10)
ax.hist(epochs, bins=rs, color = 'grey', edgecolor = 'black')
plt.show()

'''
# ---------------------------------- #
# ---- Spectro Epochs Histogram ---- #
# ---------------------------------- #
'''
sources = np.loadtxt('../../OzDES_ReverberationMapping/OzDES_Data/RM_IDs.txt')
epochs = []

for i in range(len(sources)):
    agn = str(int(sources[i]))

    spectraName = '../../OzDES_ReverberationMapping/OzDES_Data/processedSpectraY4/' + agn + '_scaled.fits'
    spectra = Spectrum(spectraName)

    epochs.append(spectra.numEpochs)

print(min(epochs))
print(max(epochs))
print(mode(epochs))

fig, ax = makeFigSingle("", "Spectroscopic Epochs", "Number", [0, 23])
rs = np.arange(0., 25, 1)
ax.hist(epochs, bins=rs, color = 'grey', edgecolor = 'black')
plt.show()
'''

# ---------------------------------- #
# ----- Photo Epochs Histogram ----- #
# ---------------------------------- #
'''
sources = ['2937961955', '2971212466']

for i in range(2):
    spectraName = '../../OzDES_ReverberationMapping/OzDES_Data/processedSpectraY4/' + sources[i] + '_scaled.fits'
    spectra = Spectrum(spectraName)

    fig, ax = makeFigSingle("", r'Wavelength [$\mathrm{\AA}$]', "Flux [10$^{-16}$ erg/s/cm$^2$/$\AA$]",
                            [spectra.wavelength[0], spectra.wavelength[4999]])
    fig.set_size_inches(12, 8, forward=True)

    ax.plot(spectra.wavelength, spectra.fluxCoadd, color = 'black')
    plt.show()
'''

# ---------------------------------- #
# --------------- CCF -------------- #
# ---------------------------------- #
#
# sources = ['2937961955', '2971212466']
# name_us, cent, lag_us, lag_max_err_us, lag_min_err_us, z_us = np.loadtxt("CIV_lags_new.txt",
#     dtype={'names':('name', 'lum', 'lumerr', 'lag', 'lagerrmin', 'lagerrmax'),'formats':('|S100', '|S100', np.float, np.float,
#         np.float, np.float)},unpack=True)
# for i in range(2):
#     fileName = sources[i] + "_ccf.dat"
#     x, y = np.loadtxt(fileName, unpack=True)
#     fig, ax = makeFigSingle("", "Observed Lag [days]", "CCF")
#     ax.plot(x,y, color = 'black', linewidth=2)
#
#     if i == 0:
#         ax.annotate('lag = ' + str(lag_us[i]) + " days", size = 20, ha="center",  xy=(lag_us[i], 0.83), xytext=(lag_us[i], 0.63),
#                     arrowprops=dict(facecolor='black', shrink=0.05))
#     if i == 1:
#         ax.annotate('lag = ' + str(lag_us[i]) + " days", size = 20, ha="center", xy=(lag_us[i], 0.9), xytext=(lag_us[i], 0.6),
#                     arrowprops=dict(facecolor='black', shrink=0.05))
#
#     plt.show()


# ---------------------------------- #
# --------------- CCF -------------- #
# ---------------------------------- #
#
# sources = ['2937961955', '2971212466']
# names = ['DES J0228-04', 'DES J0033-42']
# #names = ['DES J022828.19-040044.30', 'DES J003352.72-425452.60']
#
# for i in range(2):
#     fileName2 = "../thisIsIt/data/" + sources[i] + "_centtab.dat"
#     lag = np.loadtxt(fileName2)
#     name_us, cent, lag_us, lag_max_err_us, lag_min_err_us, z_us = np.loadtxt("../thisIsIt/data/" + "CIV_lags_new.txt",
#                                                                              dtype={'names': (
#                                                                              'name', 'cent', 'lag',
#                                                                              'lagerrmin', 'lagerrmax', 'z'), 'formats': (
#                                                                              '|S100', '|S100', np.float,
#                                                                              np.float, np.float, np.float)}, unpack=True)
#     lags = '$%s^{+%s}_{-%s}$'%(int(lag_us[i]),int(lag_max_err_us[i]),int(lag_min_err_us[i]))
#
#     if i == 1:
#         fileName = "../thisIsIt/data/" + sources[i] + "_ccf.dat"
#     if i == 0:
#         #fileName = "../thisIsIt/data/" + "2937961955_ccf_76" +  ".dat"
#         fileName = "../thisIsIt/data/" + sources[i] + "_ccf.dat"
#
#     x, y = np.loadtxt(fileName, unpack=True)
#
#     peak = 0.8*max(y)
#
#     fileName1500 = sources[i] + "_centtab.dat"
#     lags1500 = np.loadtxt(fileName1500)
#     rs1500t = np.arange(0, 850, 15)
#
#     if i == 0:
#         lims = [0,900]
#     if i == 1:
#         lims = [0, 1400]
#
#     lc1 = "../thisIsIt/data/" + sources[i] + "_gBand.txt"
#     lc2 = "../thisIsIt/data/" + sources[i] + "_CIV.txt"
#     mjd1, flux1, err1 = np.loadtxt(lc1, unpack=True, usecols=[0, 1, 2])
#     mjd2, flux2, err2 = np.loadtxt(lc2, unpack=True, usecols=[0, 1, 2])
#     SV = [q for q in np.logical_and(mjd1 < 56400, mjd1 > 55000)]
#     Y1 = [q for q in np.logical_and(mjd1 > 56400, mjd1 < 56800)]
#     Y2 = [q for q in np.logical_and(mjd1 > 56800, mjd1 < 57150)]
#     Y3 = [q for q in np.logical_and(mjd1 > 57150, mjd1 < 57500)]
#     Y4 = [q for q in np.logical_and(mjd1 > 57500, mjd1 < 57850)]
#
#     SVr = [min(mjd1[SV]), max(mjd1[SV])]
#     Y1r = [min(mjd1[Y1]), max(mjd1[Y1])]
#     Y2r = [min(mjd1[Y2]), max(mjd1[Y2])]
#     Y3r = [min(mjd1[Y3]), max(mjd1[Y3])]
#     Y4r = [min(mjd1[Y4]), max(mjd1[Y4])]
#
#     N0 = len(mjd2)
#     weights = np.zeros(len(lags1500))
#     for e in range(len(lags1500)):
#         mjd = np.copy(mjd2)
#         mjd = mjd - lags1500[e]
#
#         overlap = len(np.where(np.logical_and(mjd > Y1r[0], mjd < Y1r[1]))[0]) + \
#                   len(np.where(np.logical_and(mjd > Y2r[0], mjd < Y2r[1]))[0]) + \
#                   len(np.where(np.logical_and(mjd > Y3r[0], mjd < Y3r[1]))[0]) + \
#                   len(np.where(np.logical_and(mjd > Y4r[0], mjd < Y4r[1]))[0]) + \
#                   len(np.where(np.logical_and(mjd > SVr[0], mjd < SVr[1]))[0])
#
#
#         weights[e] = pow(overlap/N0,2)
#
#
#     fig, ax, ax1 = makeFigDouble("", "Observed Lag [days]", "CCF", "Number of Realizations", [0, 850], [-0.15, 1.2], lims)
#     ax.plot(x,y, color = 'grey', linewidth=2)
#     ax.axhline(peak, color = 'firebrick', linewidth=1)
#     rs = np.arange(0., 850, 15)
#     ax1.hist(lag, bins=rs, color = 'white', edgecolor = 'black')
#     ax1.hist(lags1500, bins=rs1500t, weights = weights, color = (.19,.19,.19), edgecolor = 'black',)
#     ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#     ax.tick_params(axis='y', pad=10)
#     ax.axvline(lag_us[i], color = 'black', linewidth = 4)
#     ax.axvline(lag_us[i] - lag_min_err_us[i], color = 'black', linestyle = '--', alpha = 0.8, linewidth = 2)
#     ax.axvline(lag_us[i] + lag_max_err_us[i], color = 'black', linestyle = '--', alpha = 0.8, linewidth = 2)
#     ax1.axvline(lag_us[i], color = 'black', linewidth = 4)
#     ax1.axvline(lag_us[i] - lag_min_err_us[i], color = 'black', linestyle = '--', alpha = 0.8, linewidth = 2)
#     ax1.axvline(lag_us[i] + lag_max_err_us[i], color = 'black', linestyle = '--', alpha = 0.8, linewidth = 2)
#     t=ax.text(645, 1.05, names[i], fontdict={'color': 'black', 'size': 20})
#     t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='white'))
#
#
#
#     # if i == 0:
#     #     ax1.set_ylim([0, 1150])
#     #     ax1.annotate(r'$\tau_{obs}$ = ' + lags + ' days', size = 20, ha="center",  xy=(lag_us[i], 800), xytext=(lag_us[i],1000),
#     #                 arrowprops=dict(facecolor='black', shrink=0.05))
#     #
#     # if i == 1:
#     #     ax1.set_ylim([0, 1550])
#     #     ax1.annotate(r'$\tau_{obs}$ = ' + lags + ' days', size = 20, ha="center", xy=(lag_us[i], 810), xytext=(lag_us[i], 1400),
#     #                 arrowprops=dict(facecolor='black', shrink=0.05))
#
#
#     plt.tight_layout()
#     fig.subplots_adjust(hspace=0)
#     plt.savefig('../thisIsIt/' + sources[i] + "_iccf.pdf", format='pdf', dpi=1000)
#     plt.savefig('../thisIsIt/' + sources[i] + "_iccf.png")
#     plt.show()


# ---------------------------------- #
# --------------- Lag -------------- #
# ---------------------------------- #
'''
sources = ['2937961955', '2971212466']

for i in range(2):
    fileName = sources[i] + "_centtab.dat"
    #fileName = sources[i] + "_peaktab.dat"
    lag = np.loadtxt(fileName)
    fig, ax = makeFigSingle("", "Observed Lag [days]", "Number", [0, 850])
    rs = np.arange(0., 850, 15)
    ax.hist(lag, bins=rs, color = 'grey', edgecolor = 'black')
    plt.show()
'''
# ---------------------------------- #
# ------------- Spectra ------------ #
# ---------------------------------- #
sources = ['2937961955', '2971212466']
redshifts = [1.905, 2.593]
#names = ['DES J022828.19-040044.30', 'DES J003352.72-425452.60']
names = ['DES J0228-04', 'DES J0033-42']

loc = "../../OzDES_ReverberationMapping/OzDES_Data/processedSpectraY4/"

Lya = 1215
SiIV = 1397
CIV = 1549
CIII = 1908
MgII = 2798



for i in range(2):
    fileName = loc + sources[i] + "_scaled.fits"
    spectra = Spectrum(fileName)
    flux = spectra.fluxCoadd
    variance = spectra.varianceCoadd
    mark_as_bad(flux, variance)

    integration = [1470*(1+redshifts[i]), 1595*(1+redshifts[i])]
    contMin = [1450*(1+redshifts[i]), 1460*(1+redshifts[i])]
    contMax = [1780*(1+redshifts[i]), 1790*(1+redshifts[i])]
    contMinBS = [1435 * (1 + redshifts[i]), 1480 * (1 + redshifts[i])]
    contMaxBS = [1695 * (1 + redshifts[i]), 1820 * (1 + redshifts[i])]

    #exponential_smooth(spectra.fluxCoadd)
    if i == 0:
        fig, ax = makeFigSingle("", "Observed " + r'Wavelength [$\mathrm{\AA}$]', "Flux [10$^{-16}$ ergs s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$]",
                            [spectra.wavelength[0], spectra.wavelength[-1]], [0, 8])
        # fig2, ax1, ax2 = makeFigDouble("", "Observed " + r'Wavelength [$\mathrm{\AA}$]', "Flux Before", "Flux After",
        #                     [contMinBS[0]-200, contMaxBS[1]+200], [-2, 8], [-2,8])

    if i == 1:
        fig, ax = makeFigSingle("", "Observed " + r'Wavelength [$\mathrm{\AA}$]', "Flux [10$^{-16}$ ergs s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$]",
                            [spectra.wavelength[0], spectra.wavelength[-1]], [0, 5.5])
        # fig2, ax1, ax2 = makeFigDouble("", "Observed " + r'Wavelength [$\mathrm{\AA}$]',  "Flux Before", "Flux After",
        #                     [contMinBS[0]-200, contMaxBS[1]+200], [-1, 5.5], [-1,5.5])
    fig.set_size_inches(14, 10, forward=True)

    # ax.plot(spectra.wavelength, flux, color = 'black')
    ax.plot(spectra.wavelength, flux, color = 'black')

    wave, cont = cont_fit_reject(spectra.wavelength,flux, variance, contMin, contMax)

    wavelengthRF = spectra.wavelength/(1+redshifts[i])

    #ax.plot(spectra.wavelength, 100*(spectra.variance[:,0]**0.5)/spectra.flux[:,0], color = 'black', linewidth = 2)
    if i == 1:
        t0=ax.text((1+redshifts[i])*CIV-200, 2.2, "C IV", fontdict={'color': 'black', 'size': 20})
        t1=ax.text((1+redshifts[i])*SiIV-130, 1.6, "Si IV", fontdict={'color': 'black', 'size': 20})
        t2=ax.text((1+redshifts[i])*CIII-120, 1.2, "C III", fontdict={'color': 'black', 'size': 20})
        t3=ax.text((1+redshifts[i])*Lya-120, 5.1, "Ly" + r'$\alpha$', fontdict={'color': 'black', 'size': 20})
        t0.set_bbox(dict(facecolor='white', alpha=0.0, edgecolor='white'))
        t1.set_bbox(dict(facecolor='white', alpha=0.0, edgecolor='white'))
        t2.set_bbox(dict(facecolor='white', alpha=0.0, edgecolor='white'))
        t3.set_bbox(dict(facecolor='white', alpha=0.0, edgecolor='white'))

    if i == 0:
        t0=ax.text((1+redshifts[i])*CIV-185, 6.7, "C IV", fontdict={'color': 'black', 'size': 20})
        t1=ax.text((1+redshifts[i])*SiIV-150, 4.2, "Si IV", fontdict={'color': 'black', 'size': 20})
        t2=ax.text((1+redshifts[i])*CIII-120, 3, "C III", fontdict={'color': 'black', 'size': 20})
        t3=ax.text((1+redshifts[i])*MgII-120, 2.2, "Mg II", fontdict={'color': 'black', 'size': 20})
        t0.set_bbox(dict(facecolor='white', alpha=0.0, edgecolor='white'))
        t1.set_bbox(dict(facecolor='white', alpha=0.0, edgecolor='white'))
        t2.set_bbox(dict(facecolor='white', alpha=0.0, edgecolor='white'))
        t3.set_bbox(dict(facecolor='white', alpha=0.0, edgecolor='white'))

    ax.axvline(x = integration[0], color = 'firebrick', linestyle='--', linewidth=2)
    ax.axvline(x = integration[1], color = 'firebrick', linestyle='--', linewidth=2)
    # ax2.axvline(x = integration[0], color = 'firebrick')
    # ax2.axvline(x = integration[1], color = 'firebrick')

    ax.axvspan(contMinBS[0], contMinBS[1], color='forestgreen', alpha=0.2)
    ax.axvspan(contMaxBS[0], contMaxBS[1], color='forestgreen', alpha=0.2)
    ax.axvspan(contMin[0], contMin[1], color='forestgreen', alpha=0.5)
    ax.axvspan(contMax[0], contMax[1], color='forestgreen', alpha=0.5)
    # ax2.axvspan(contMin[0], contMin[1], color='forestgreen', alpha=0.5)
    # ax2.axvspan(contMax[0], contMax[1], color='forestgreen', alpha=0.5)

    ax.plot(wave, cont, color = 'mediumblue', linewidth=2)

    # ax2.plot(spectra.wavelength, flux, color = 'black')


    # for 466: fig 0-5.5 name 5.1 z 4.8
    # for 955: fig 0-8 name 7.48 z 6.98
    if i == 0:
        t4=ax.text(7905, 7.48, names[i], fontdict={'color': 'black', 'size': 20})
        t5=ax.text(8160, 6.98, "z = " + str(redshifts[i]), fontdict={'color': 'black', 'size': 20})
        t4.set_bbox(dict(facecolor='white', alpha=0.0, edgecolor='white'))
        t5.set_bbox(dict(facecolor='white', alpha=0.0, edgecolor='white'))

    if i == 1:
        t4=ax.text(7905, 5.1, names[i], fontdict={'color': 'black', 'size': 20})
        t5=ax.text(8160, 4.8, "z = " + str(redshifts[i]), fontdict={'color': 'black', 'size': 20})
        t4.set_bbox(dict(facecolor='white', alpha=0.0, edgecolor='white'))
        t5.set_bbox(dict(facecolor='white', alpha=0.0, edgecolor='white'))


    ax3 = ax.twiny()
    title = ax.set_title("Rest Frame " + r'Wavelength [$\mathrm{\AA}$]', **axis_font)
    title.set_y(1.1)
    for label in (ax3.get_xticklabels() + ax3.get_yticklabels()):
        label.set_fontsize(23)

    if i == 1:
        ticksOBS = (1+redshifts[i])*np.array([1100, 1400, 1700, 2000, 2300])
        tickslab = ['1100', '1400', '1700', '2000', '2300']
    if i == 0:
        ticksOBS = (1+redshifts[i])*np.array([1400, 1700, 2000, 2300, 2600, 2900])
        tickslab = ['1400', '1700', '2000', '2300', '2600', '2900']

    bound = (spectra.wavelength[0], spectra.wavelength[-1])
    ax3.set_xbound(bound)
    ax3.set_xticks(ticksOBS)
    ax3.set_xticklabels(tickslab)

    # ax2Xs = []
    # for X in ax.get_xticks():
    #     ax2Xs.append(int(X/(1+redshifts[i])))
    #
    # ax2.set_xticks(ax.get_xticks())
    # ax2.set_xbound(ax.get_xbound())
    # ax2.set_xticklabels(ax2Xs)

    fig.subplots_adjust(top=0.85)

    plt.savefig('../thisIsIt/' + sources[i] + "_spectrum.pdf", format='pdf', dpi=1000)
    plt.savefig('../thisIsIt/' + sources[i] + "_spectrum.png")

    plt.show()

# ---------------------------------- #
# ------------- Spectra ------------ #
# ---------------------------------- #
# sources = ['2937961955', '2971212466']
# redshifts = [1.905, 2.593]
# #names = ['DES J022828.19-040044.30', 'DES J003352.72-425452.60']
# names = ['DES J0228-04', 'DES J0033-42']
#
# loc = "../../OzDES_ReverberationMapping/OzDES_Data/processedSpectraY4/"
#
# Lya = 1215
# SiIV = 1397
# CIV = 1549
# CIII = 1908
# MgII = 2798
#
#
#
# for i in range(1,2):
#     fileName = loc + sources[i] + "_scaled.fits"
#     spectra = Spectrum(fileName)
#     flux = spectra.flux
#     variance = spectra.variance
#
#     integration = [1470*(1+redshifts[i]), 1595*(1+redshifts[i])]
#     contMin = [1450*(1+redshifts[i]), 1460*(1+redshifts[i])]
#     contMax = [1780*(1+redshifts[i]), 1790*(1+redshifts[i])]
#     contMinBS = [1435 * (1 + redshifts[i]), 1480 * (1 + redshifts[i])]
#     contMaxBS = [1695 * (1 + redshifts[i]), 1820 * (1 + redshifts[i])]
#
#     rangeP = [contMinBS[1], contMaxBS[0]]
#     #rangeP = [spectra.wavelength[0], spectra.wavelength[-1]]
#     #exponential_smooth(spectra.fluxCoadd)
#     if i == 0:
#         # fig, ax = makeFigSingle("", "Observed " + r'Wavelength [$\mathrm{\AA}$]', "Flux [10$^{-16}$ ergs s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$]",
#         #                             [spectra.wavelength[0], spectra.wavelength[-1]], [0, 8])
#         fig, ax = makeFigSingle("", "Observed " + r'Wavelength [$\mathrm{\AA}$]', "Flux [10$^{-16}$ ergs s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$]",
#                                     rangeP, [-0.5, 6])
#         # fig, ax = makeFigSingle("", "Observed " + r'Wavelength [$\mathrm{\AA}$]', "Flux [10$^{-16}$ ergs s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$]",
#         #                     rangeP, [-0.5, 6])
#         # fig2, ax1, ax2 = makeFigDouble("", "Observed " + r'Wavelength [$\mathrm{\AA}$]', "Flux Before", "Flux After",
#         #                     [contMinBS[0]-200, contMaxBS[1]+200], [-2, 8], [-2,8])
#
#     if i == 1:
#         # fig, ax = makeFigSingle("", "Observed " + r'Wavelength [$\mathrm{\AA}$]', "Flux [10$^{-16}$ ergs s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$]",
#         #                             [spectra.wavelength[0], spectra.wavelength[-1]], [0, 5.5])
#         fig, ax = makeFigSingle("", "Observed " + r'Wavelength [$\mathrm{\AA}$]', "Flux [10$^{-16}$ ergs s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$]",
#                                     rangeP, [-0.2, 1.6])
#         # fig, ax = makeFigSingle("", "Observed " + r'Wavelength [$\mathrm{\AA}$]', "Flux [10$^{-16}$ ergs s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$]",
#         #                     rangeP, [-0.2, 1.7])
#         # fig2, ax1, ax2 = makeFigDouble("", "Observed " + r'Wavelength [$\mathrm{\AA}$]',  "Flux Before", "Flux After",
#         #                     [contMinBS[0]-200, contMaxBS[1]+200], [-1, 5.5], [-1,5.5])
#         ax.yaxis.major.locator.set_params(nbins=4)
#     #fig.set_size_inches(14, 10, forward=True)
#     fig.set_size_inches(10, 10, forward=True)
#
#     # ax.plot(spectra.wavelength, flux, color = 'black')
#     # ax.plot(spectra.wavelength, flux, color = 'black')
#
#     exponential_smooth(flux)
#     colors = cm.rainbow(np.linspace(0, 1, spectra.numEpochs))
#     wave, cont = cont_fit_reject(spectra.wavelength,flux, variance, contMin, contMax)
#
#
#     for e in range(spectra.numEpochs):
#         ax.plot(spectra.wavelength, spectra.flux[:,e], color=colors[e])
#
#     wavelengthRF = spectra.wavelength/(1+redshifts[i])
#
#     #ax.plot(spectra.wavelength, 100*(spectra.variance[:,0]**0.5)/spectra.flux[:,0], color = 'black', linewidth = 2)
#     # if i == 1:
#     #     t0=ax.text((1+redshifts[i])*CIV-200, 2.2, "C IV", fontdict={'color': 'black', 'size': 20})
#     #     t1=ax.text((1+redshifts[i])*SiIV-130, 1.6, "Si IV", fontdict={'color': 'black', 'size': 20})
#     #     t2=ax.text((1+redshifts[i])*CIII-120, 1.2, "C III", fontdict={'color': 'black', 'size': 20})
#     #     t3=ax.text((1+redshifts[i])*Lya-120, 5.1, "Ly" + r'$\alpha$', fontdict={'color': 'black', 'size': 20})
#     #     t0.set_bbox(dict(facecolor='white', alpha=0.0, edgecolor='white'))
#     #     t1.set_bbox(dict(facecolor='white', alpha=0.0, edgecolor='white'))
#     #     t2.set_bbox(dict(facecolor='white', alpha=0.0, edgecolor='white'))
#     #     t3.set_bbox(dict(facecolor='white', alpha=0.0, edgecolor='white'))
#     #
#     # if i == 0:
#     #     t0=ax.text((1+redshifts[i])*CIV-185, 7, "C IV", fontdict={'color': 'black', 'size': 20})
#     #     t1=ax.text((1+redshifts[i])*SiIV-150, 4.8, "Si IV", fontdict={'color': 'black', 'size': 20})
#     #     t2=ax.text((1+redshifts[i])*CIII-120, 3, "C III", fontdict={'color': 'black', 'size': 20})
#     #     t3=ax.text((1+redshifts[i])*MgII-120, 2.2, "Mg II", fontdict={'color': 'black', 'size': 20})
#     #     t0.set_bbox(dict(facecolor='white', alpha=0.0, edgecolor='white'))
#     #     t1.set_bbox(dict(facecolor='white', alpha=0.0, edgecolor='white'))
#     #     t2.set_bbox(dict(facecolor='white', alpha=0.0, edgecolor='white'))
#     #     t3.set_bbox(dict(facecolor='white', alpha=0.0, edgecolor='white'))
#
#     # ax.axvline(x = integration[0], color = 'firebrick', linestyle='--', linewidth=2)
#     # ax.axvline(x = integration[1], color = 'firebrick', linestyle='--', linewidth=2)
#     # # ax2.axvline(x = integration[0], color = 'firebrick')
#     # # ax2.axvline(x = integration[1], color = 'firebrick')
#     #
#     # ax.axvspan(contMinBS[0], contMinBS[1], color='forestgreen', alpha=0.2)
#     # ax.axvspan(contMaxBS[0], contMaxBS[1], color='forestgreen', alpha=0.2)
#     # ax.axvspan(contMin[0], contMin[1], color='forestgreen', alpha=0.5)
#     # ax.axvspan(contMax[0], contMax[1], color='forestgreen', alpha=0.5)
#     # ax2.axvspan(contMin[0], contMin[1], color='forestgreen', alpha=0.5)
#     # ax2.axvspan(contMax[0], contMax[1], color='forestgreen', alpha=0.5)
#
#     # ax.plot(wave, cont, color = 'mediumblue', linewidth=2)
#
#     # ax2.plot(spectra.wavelength, flux, color = 'black')
#
#
#     # for 466: fig 0-5.5 name 5.1 z 4.8
#     # for 955: fig 0-8 name 7.48 z 6.98
#
#     if i == 0:
#         t4=ax.text(7905, 7.48, names[i], fontdict={'color': 'black', 'size': 20})
#         t5=ax.text(8160, 6.98, "z = " + str(redshifts[i]), fontdict={'color': 'black', 'size': 20})
#         t4.set_bbox(dict(facecolor='white', alpha=0.0, edgecolor='white'))
#         t5.set_bbox(dict(facecolor='white', alpha=0.0, edgecolor='white'))
#
#     if i == 1:
#         t4=ax.text(7905, 5.1, names[i], fontdict={'color': 'black', 'size': 20})
#         t5=ax.text(8160, 4.8, "z = " + str(redshifts[i]), fontdict={'color': 'black', 'size': 20})
#         t4.set_bbox(dict(facecolor='white', alpha=0.0, edgecolor='white'))
#         t5.set_bbox(dict(facecolor='white', alpha=0.0, edgecolor='white'))
#
#     if i == 0:
#         ax.set_xticks((np.array([4300, 4500, 4700, 4900])))
#         ax.set_xticklabels(['4300', '4500', '4700', '4900'])
#     if i == 1:
#         ax.set_xticks((np.array([5400, 5600, 5800, 6000])))
#         ax.set_xticklabels(['5400', '5600', '5800', '6000'])
#
#     ax3 = ax.twiny()
#     title = ax.set_title("Rest Frame " + r'Wavelength [$\mathrm{\AA}$]', **axis_font)
#     title.set_y(1.1)
#     for label in (ax3.get_xticklabels() + ax3.get_yticklabels()):
#         label.set_fontsize(23)
#
#     # if i == 1:
#     #     ticksOBS = (1+redshifts[i])*np.array([1100, 1400, 1700, 2000, 2300])
#     #     tickslab = ['1100', '1400', '1700', '2000', '2300']
#     # if i == 0:
#     #     ticksOBS = (1+redshifts[i])*np.array([1400, 1700, 2000, 2300, 2600, 2900])
#     #     tickslab = ['1400', '1700', '2000', '2300', '2600', '2900']
#
#     ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#
#     bound = [contMinBS[1], contMaxBS[0]]
#     ax3.set_xbound(bound)
#
#     if i == 1:
#         ax3.set_xticks((1+redshifts[i])*(np.array([1500, 1550, 1600, 1650])))
#         ax3.set_xticklabels(['1500', '1550', '1600', '1650'])
#     if i == 0:
#         ax3.set_xticks((1+redshifts[i])*(np.array([1500, 1550, 1600, 1650])))
#         ax3.set_xticklabels(['1500', '1550', '1600', '1650'])
#
#     # bound = (spectra.wavelength[0], spectra.wavelength[-1])
#     # bound = [contMinBS[1], contMaxBS[0]]
#     # ax3.set_xbound(bound)
#     # ax3.set_xticks(ticksOBS)
#     # ax3.set_xticklabels(tickslab)
#
#     # ax2Xs = []
#     # for X in ax.get_xticks():
#     #     ax2Xs.append(int(X/(1+redshifts[i])))
#     #
#     # ax3.set_xticks(ax.get_xticks())
#     # ax3.set_xbound(ax.get_xbound())
#     # ax3.set_xticklabels(ax2Xs)
#
#     fig.subplots_adjust(top=0.85)
#     #
#     # plt.savefig('../thisIsIt/' + sources[i] + "_colSpec.pdf", format='pdf', dpi=1000)
#     # plt.savefig('../thisIsIt/' + sources[i] + "_colSpec.png")
#
#     plt.savefig('../thisIsIt/' + sources[i] + "_colLine.pdf", format='pdf', dpi=1000)
#     plt.savefig('../thisIsIt/' + sources[i] + "_colLine.png")
#     plt.show()

# ---------------------------------- #
# ----------Lots Spectra ----------- #
# ---------------------------------- #

#
# sources = ['2937961955', '2971212466']
# redshifts = [1.905, 2.593]
# names = ['DES J022828.19-040044.30', 'DES J003352.72-425452.60']
# loc = "../../OzDES_ReverberationMapping/OzDES_Data/processedSpectraY4/"
# axis_font2 = {'size':'15'}
# for i in range(1,2):
#     fileName = loc + sources[i] + "_scaled.fits"
#     spectra = Spectrum(fileName)
#     flux = spectra.flux
#     variance = spectra.variance
#     wave = spectra.wavelength
#
#     exponential_smooth(flux)
#
#     numEpochs = spectra.numEpochs
#     print(numEpochs)
#
#     fig, ax_array = plt.subplots(numEpochs, sharex=True)
#     fig = plt.gcf()
#     fig.set_size_inches(10, 13.93, forward=True)
#     ax = fig.add_subplot(111, frameon=False)
#     fig.subplots_adjust(hspace=0)
#     textarray = np.zeros(numEpochs)
#     ax_array[numEpochs-1].set_xlabel(r'Wavelength [$\mathrm{\AA}$]', **axis_font2)
#     ax.set_ylabel("Flux [10$^{-16}$ erg/s/cm$^{2}$/$\AA$]", labelpad=20, **axis_font2)
#     ax.yaxis.set_major_locator(plt.NullLocator())
#     ax.yaxis.set_major_formatter(plt.NullFormatter())
#     ax.xaxis.set_major_locator(plt.NullLocator())
#     ax.xaxis.set_major_formatter(plt.NullFormatter())
#
#     if i == 0:
#         t = ax_array[0].text(7595, 4, names[i],
#                              fontdict={'color': 'black', 'size': 10})
#         t.set_bbox(dict(facecolor='white', alpha=0.0, edgecolor='white'))
#
#     if i == 1:
#         t = ax_array[0].text(7595, 3.2, names[i],
#                              fontdict={'color': 'black', 'size': 10})
#         t.set_bbox(dict(facecolor='white', alpha=0.0, edgecolor='white'))
#     for e in range(numEpochs):
#         ax_array[e].set_xlim([wave[0], wave[-1]])
#         #ax_array[e].set_ylabel('f$_\lambda$', **axis_font)
#         if i == 0:
#             ax_array[e].set_ylim([0,9])
#             t = ax_array[e].text(8215, 6, "MJD " + str(round(spectra.dates[e], 2)),
#                                  fontdict={'color': 'black', 'size': 10})
#
#         if i == 1:
#             ax_array[e].set_ylim([0,7])
#             t = ax_array[e].text(8215, 5, "MJD " + str(round(spectra.dates[e], 2)),
#                                  fontdict={'color': 'black', 'size': 10})
#
#         # for label in (ax_array[e].get_xticklabels() + ax_array[e].get_yticklabels()):
#         #     label.set_fontsize(20)
#         t.set_bbox(dict(facecolor='white', alpha=0.0, edgecolor='white'))
#         for label in (ax_array[e].get_xticklabels()):
#             label.set_fontsize(15)
#         for label in (ax_array[e].get_yticklabels()):
#             label.set_fontsize(15)
#         ax_array[e].plot(wave, flux[:,e], color='black')
#
#     wavelengthRF = spectra.wavelength/(1+redshifts[i])
#     ax3 = ax_array[0].twiny()
#     title = ax_array[0].set_title("Rest Frame " + r'Wavelength [$\mathrm{\AA}$]', **axis_font2)
#     title.set_y(1.7)
#     for label in (ax3.get_xticklabels() + ax3.get_yticklabels()):
#         label.set_fontsize(15)
#
#     if i == 1:
#         ticksOBS = (1+redshifts[i])*np.array([1100, 1400, 1700, 2000, 2300])
#         tickslab = ['1100', '1400', '1700', '2000', '2300']
#     if i == 0:
#         ticksOBS = (1+redshifts[i])*np.array([1400, 1700, 2000, 2300, 2600, 2900])
#         tickslab = ['1400', '1700', '2000', '2300', '2600', '2900']
#
#     bound = (spectra.wavelength[0], spectra.wavelength[-1])
#     ax3.set_xbound(bound)
#     ax3.set_xticks(ticksOBS)
#     ax3.set_xticklabels(tickslab)
#     plt.show()
# ---------------------------------- #
# ------------- F-Stars ------------ #
# ---------------------------------- #
'''
colors = pickle.load(open("colors.pkl", "rb"))
spectra = Spectrum("FSC0225m0444_scaled.fits")
filter_bad_pixels(spectra.flux, spectra.variance, spectra.numEpochs)
fig, ax, ax2 = makeFigDouble("", r'Wavelength [$\mathrm{\AA}$]', "Flux [10$^{-16}$ erg/s/cm$^{2}$/$\AA$]", "RMS Flux/Mean Flux",
                             [spectra.wavelength[0], spectra.wavelength[-1]], [0, 6], [0, 0.23])
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

for i in range(spectra.numEpochs):
    ax.plot(spectra.wavelength, spectra.flux[:,i], alpha = 0.7, color = colors['grey'][i%6])
rmsFlux = rmsSpec(spectra.flux)
meanFlux = meanSpec(spectra.flux)
ax2.plot(spectra.wavelength, rmsFlux/meanFlux, color = 'black')
plt.show()
'''

# # ---------------------------------- #
# # ------------- F-Stars ------------ #
# # ---------------------------------- #

# colors = pickle.load(open("colors.pkl", "rb"))
# spectra = Spectrum("FSC0225m0444_scaled.fits")
# length = len(spectra.dates[0:13])
# mark_as_bad(spectra.flux, spectra.variance)
#
# fig, ax, ax2, ax3 = makeFigTriple("", r'Wavelength [$\mathrm{\AA}$]', "Flux", "RMS Flux", "% Variation",
#                              [spectra.wavelength[0], spectra.wavelength[-1]], [0, 6], [0, 0.79], [0, 24])
# ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#
# for i in range(length):
#     ax.plot(spectra.wavelength, spectra.flux[:,i], alpha = 0.7, color = colors['grey'][(i+1)%6])
# rmsFlux = rmsSpec(spectra.flux[:,0:13])
# meanFlux = meanSpec(spectra.flux[:,0:13])
# ax2.plot(spectra.wavelength, rmsFlux, color = 'black')
# ax3.plot(spectra.wavelength, 100*rmsFlux/meanFlux, color = 'black')
# ax3.annotate("Red/Blue Splice", xy=(5700,7), xytext=(5700, 10),arrowprops=dict(facecolor='grey', shrink=0.02))
# t = ax3.text(5450, 11, "Red/Blue Splice",
#                              fontdict={'color': 'black', 'size': 14}, bbox={'facecolor':'white', 'alpha':1, 'pad':10, 'edgecolor':'white'})
# # #ax3.plot(spectra.wavelength, 100*(spectra.variance[:,0]**0.5)/spectra.flux[:,0])
# plt.savefig('../thisIsIt/FSC0225m0444_calibTest_3plot.pdf', format='pdf', dpi=1000)
# plt.show()

# colors = pickle.load(open("colors.pkl", "rb"))
# spectra = Spectrum("FSC0225m0444_scaled.fits")
# length = len(spectra.dates[0:13])
# mark_as_bad(spectra.flux, spectra.variance)
#
# fig, ax = makeFigSingle("After Scaling",  r'Wavelength [$\mathrm{\AA}$]', "Flux[10$^{-16}$ ergs s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$]",
#                         [spectra.wavelength[0], spectra.wavelength[-1]], [0, 6])
#
# for i in range(length):
#     ax.plot(spectra.wavelength, spectra.flux[:,i], alpha = 0.7, color = colors['grey'][(i+1)%6])
# plt.show()
#
# colors = pickle.load(open("colors.pkl", "rb"))
# spectra = Spectrum("FSC0225m0444_scaled_not.fits")
# length = len(spectra.dates[0:13])
# mark_as_bad(spectra.flux, spectra.variance)
#
# fig, ax = makeFigSingle("Before Scaling",  r'Wavelength [$\mathrm{\AA}$]', "Flux [arbitrary units]",
#                         [spectra.wavelength[0], spectra.wavelength[-1]], [0, 4])
#
# for i in range(length):
#     ax.plot(spectra.wavelength, spectra.flux[:,i]*10**-16, alpha = 0.7, color = colors['grey'][(i+1)%6])
#
# plt.show()


# ---------------------------------- #
# ---------- Light Curves ---------- #
# ---------------------------------- #
'''
names = ['DES J022828.19-040044.30', 'DES J003352.72-425452.60']
source = ['2937961955', '2971212466']
name_us, cent, lag_us, lag_max_err_us, lag_min_err_us, z_us = np.loadtxt("CIV_lags_new.txt",
    dtype={'names':('name', 'lum', 'lumerr', 'lag', 'lagerrmin', 'lagerrmax'),'formats':('|S100', '|S100', np.float, np.float,
        np.float, np.float)},unpack=True)

for s in range(2):
    date_line, flux_line, err_line = np.loadtxt(source[s] + "_CIV.txt", unpack=True)
    date_cont, flux_cont, err_cont = np.loadtxt(source[s] + "_gBand.txt", unpack=True)

    lags = '$%s^{+%s}_{-%s}$'%(int(lag_us[s]),int(lag_max_err_us[s]),int(lag_min_err_us[s]))
    title = "SVA1_COADD-" + source[s] + "\nz = " + str(round(z_us[s],3)) + "   lag = " + lags + " days"

    if s == 0:
        fig, ax1, ax2 = makeFigDouble("", "Date [MJD]", "Continuum Flux",
                                      "CIV Line Flux", [56100, 57900], [18, 35], [min(flux_line)-0.6, max(flux_line)+0.5])

    if s == 1:
        fig, ax1, ax2 = makeFigDouble("", "Date [MJD]", "Continuum Flux",
                                      "CIV Line Flux", [56100, 57900], [10, 17.5], [min(flux_line)-0.4, max(flux_line)+0.5])

    ax1.yaxis.set_major_locator(MaxNLocator(prune='upper'))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    ax2.yaxis.set_major_locator(MaxNLocator(prune='upper'))
    ax1.yaxis.major.locator.set_params(nbins=5)
    ax2.yaxis.major.locator.set_params(nbins=5)
    ax2.tick_params(axis='x', pad=10)
    ax1.errorbar(date_cont, flux_cont, yerr=err_cont, color='black', fmt='o', ms=6, elinewidth=1.5)
    ax2.errorbar(date_line, flux_line, yerr=err_line, color='black', fmt='o', ms=6, elinewidth=1.5)
    if s == 0:
        ax1.text(56200, 33, names[s], fontdict={'color': 'black', 'size': 20})
        ax1.text(56200, 31.5, "z = " + str(round(z_us[s],3)) + "; " + r'$\tau_{obs}$ = ' + lags + ' days',
                 fontdict={'color': 'black', 'size': 20})
    if s == 1:
        ax1.text(56200, 16.615, names[s], fontdict={'color': 'black', 'size': 20})
        ax1.text(56200, 15.958, "z = " + str(round(z_us[s],3)) + "; " + r'$\tau_{obs}$ = ' + lags + ' days',
                 fontdict={'color': 'black', 'size': 20})
    plt.show()
'''
# source = ['2937961955', '2971212466']
# for s in range(2):
#     date_line, flux_line, err_line = np.loadtxt(source[s] + "_gBand_test.txt", unpack=True)
#     date_cont, flux_cont, err_cont = np.loadtxt("../thisIsIt/data/" + source[s] + "_gBand.txt", unpack=True)
#     date_line2, flux_line2, err_line2 = np.loadtxt(source[s] + "_gBand_testAp.txt", unpack=True)
#     fig, ax1 = makeFigSingle("", "Date [MJD]", "Continuum Flux")
#     ax1.errorbar(date_line2, flux_line2, yerr=err_line2,  fmt='o',
#                 mfc = 'none', mec = 'black', color = 'black', markersize = 9, markeredgewidth=1.5, elinewidth=1)
#     ax1.errorbar(date_line, flux_line, yerr=err_line,   fmt='s', color = 'firebrick',
#                 markersize = 5, elinewidth=2)
#     ax1.errorbar(date_cont, flux_cont, yerr=err_cont, fmt='s', color = 'mediumblue',
#                 markersize = 5, elinewidth=2)
#     plt.show()


# ---------------------------------- #
# ---------- Light Curves ---------- #
# ---------------------------------- #

# #names = ['DES J022828.19-040044.30', 'DES J003352.72-425452.60']
# names = ['DES J0228-04', 'DES J0033-42']
#
# source = ['2937961955', '2971212466']
# name_us, cent, lag_us, lag_max_err_us, lag_min_err_us, r_us = np.loadtxt("../thisIsIt/data/CIV_lags_new.txt",
#     dtype={'names':('name', 'lum', 'lumerr', 'lag', 'lagerrmin', 'lagerrmax'),'formats':('|S100', '|S100', np.float, np.float,
#         np.float, np.float)},unpack=True)
#
# z_us = [1.905, 2.593]
# for s in range(2):
#     date_line, flux_line, err_line = np.loadtxt("../thisIsIt/data/" + source[s] + "_CIV.txt", unpack=True)
#     date_cont, flux_cont, err_cont = np.loadtxt("../thisIsIt/data/" + source[s] + "_gBand.txt", unpack=True)
#
#     lags = '$%s^{+%s}_{-%s}$'%(int(round(lag_us[s],0)),int(round(lag_max_err_us[s],0)),int(round(lag_min_err_us[s],0)))
#     title = "SVA1_COADD-" + source[s] + "\nz = " + str(round(z_us[s],3)) + "   lag = " + lags + " days"
#
#     if s == 0:
#         fig, ax1, ax2, ax3 = makeFigTriple("", "Date [MJD]", "Continuum Flux",
#                                       "CIV Line Flux","Lag Adjusted Flux", [56100, 57900], [18, 35],
#                                            [min(flux_line)-0.6, max(flux_line)+0.5], [min(flux_line)-0.6, max(flux_line)+0.5])
#         fig.set_size_inches(10, 12, forward=True)
#
#     if s == 1:
#         fig, ax1, ax2, ax3 = makeFigTriple("", "Date [MJD]", "Continuum Flux",
#                                       "CIV Line Flux","Lag Adjusted Flux", [56100, 57900], [10, 17.5],
#                                            [min(flux_line)-0.4, max(flux_line)+0.6], [min(flux_line)-0.4, max(flux_line)+0.6])
#         fig.set_size_inches(10, 12, forward=True)
#
#     ax1.yaxis.set_major_locator(MaxNLocator(prune='lower'))
#     ax1.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
#
#     ax2.yaxis.set_major_locator(MaxNLocator(prune='upper'))
#     ax3.yaxis.set_major_locator(MaxNLocator(prune='upper'))
#     ax1.yaxis.major.locator.set_params(nbins=5)
#     ax2.yaxis.major.locator.set_params(nbins=5)
#     ax2.tick_params(axis='x', pad=10)
#     ax3.yaxis.major.locator.set_params(nbins=5)
#     ax3.tick_params(axis='x', pad=10)
#     ax3.xaxis.major.locator.set_params(nbins=5)
#
#     ax1.errorbar(date_cont, flux_cont, yerr=err_cont, color='black', fmt='o', ms=6, elinewidth=1.5)
#     ax2.errorbar(date_line, flux_line, yerr=err_line, color='black', fmt='o', ms=6, elinewidth=1.5)
#     ax3.errorbar(date_line-lag_us[s], flux_line, yerr=err_line, color='black', fmt='o', ms=6, elinewidth=1.5)
#     if s == 0:
#         ax1.text(56200, 33, names[s], fontdict={'color': 'black', 'size': 18})
#         ax1.text(56200, 31.3, "z = " + str(round(z_us[s],3)) + "; " + r'$\tau_{obs}$ = ' + lags + ' days',
#                  fontdict={'color': 'black', 'size': 18})
#         ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#         ax3.text(57310, 13.7, 'Date - ' + str(int(round(lag_us[0]))) + ' days', fontdict={'color': 'black', 'size': 16})
#         ax3.arrow(57700, 13.5, -lag_us[s], 0, width=0.05, head_width=0.15, head_length=25, ec='grey', fc='grey')
#
#     if s == 1:
#         ax1.text(56200, 16.615, names[s], fontdict={'color': 'black', 'size': 18})
#         ax1.text(56200, 15.9, "z = " + str(round(z_us[s],3)) + "; " + r'$\tau_{obs}$ = ' + lags + ' days',
#                  fontdict={'color': 'black', 'size': 18})
#         ax1.tick_params(axis='y', pad=14)
#         ax3.text(56620, 3.6, 'Date - ' + str(int(round(lag_us[s]))) + ' days', fontdict={'color': 'black', 'size': 16})
#         ax3.arrow(57000, 3.48, -lag_us[s], 0, width=0.05, head_width=0.15, head_length=25, ec='grey', fc='grey')
#
#     plt.savefig('../thisIsIt/' + source[s] + '_lightCurve.pdf', format='pdf', dpi=1000)
#     plt.savefig('../thisIsIt/' + source[s] + '_lightCurve.png')
#
#
#     plt.show()

# ---------------------------------- #
# ------ Light Curves for Talk ----- #
# ---------------------------------- #
'''
source = ['2937961955', '2971212466']
name_us, lum_us, lum_err_us, lag_us, lag_min_err_us, lag_max_err_us, z_us = np.loadtxt("OzDES_CIV_Lags.txt",
    dtype={'names':('name', 'lum', 'lumerr', 'lag', 'lagerrmin', 'lagerrmax', 'z'),'formats':('|S100', np.float, np.float,
        np.float, np.float, np.float, np.float)},skiprows=1,unpack=True)

for s in range(2):
    date_line, flux_line, err_line = np.loadtxt(source[s] + "_CIV.txt", unpack=True)
    date_cont, flux_cont, err_cont = np.loadtxt(source[s] + "_gBand.txt", unpack=True)
    
    lags = '$%s^{+%s}_{-%s}$'%(int(lag_us[s]),int(lag_max_err_us[s]),int(lag_min_err_us[s]))
    title = "SVA1_COADD-" + source[s] + "\nz = " + str(round(z_us[s],3)) + "   lag = " + lags + " days"

    if s == 0:
        fig, ax1 = makeFigSingle("", "Date [MJD]", "Continuum Flux [10$^{-17}$ erg/s/cm$^{2}$/$\AA$]", [56100, 57900], [10.5, 32])
        #fig, ax1, ax2 = makeFigDouble("", "Date [MJD]", "Continuum Flux",
        #                              "CIV Line Flux", [56100, 57900], [14, 35], [min(flux_line) - 0.4, max(flux_line) + 0.9])
    if s == 1:
        fig, ax1 = makeFigSingle("", "Date [MJD]", "Continuum Flux [10$^{-17}$ erg/s/cm$^{2}$/$\AA$]", [56100, 57900], [6.5, 16])
        #fig, ax1, ax2 = makeFigDouble("", "Date [MJD]", "Continuum Flux",
        #                              "CIV Line Flux", [56100, 57900], [10, 16], [min(flux_line) - 0.4, max(flux_line) + 0.8])


    ax2 = ax1.twinx()
    ax2.set_ylabel("CIV Line Flux [10$^{-17}$ erg/s/cm$^{2}$/$\AA$]", **axis_font, color = 'firebrick')
    ax2.xaxis.major.locator.set_params(nbins=4)
    if s == 0:
        ax2.set_ylim([9.5, 19])
    if s == 1:
        ax2.set_ylim([1.2, 5.5])
    for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
        label.set_fontsize(25)
    ax2.tick_params(axis='y', labelcolor = 'firebrick')

    #fig.set_size_inches(12, 10, forward=True)

    ax1.errorbar(date_cont, flux_cont, yerr=err_cont, color='black', fmt='o', ms=6, elinewidth=1.5)
    ax2.errorbar(date_line, flux_line, yerr=err_line, color='firebrick', fmt='s', ms=6, elinewidth=1.5)

    plt.show()
'''
# ---------------------------------- #
# ----------- R-L Diagram ---------- #
# ---------------------------------- #
#
# name_prev, lum_prev, lum_err_prev, lag_prev, lag_min_err_prev, lag_max_err_prev, z_prev = np.loadtxt("exisitingLags_all.txt",
#     dtype={'names':('name', 'lum', 'lumerr', 'lag', 'lagerrmin', 'lagerrmax', 'z'),'formats':('|S100', np.float, np.float,
#         np.float, np.float, np.float, np.float)},skiprows=1,unpack=True)
#
# name_us, lum_us, lum_err_us, lag_us, lag_min_err_us, lag_max_err_us, z_us = np.loadtxt("OzDES_CIV_Lags.txt",
#     dtype={'names':('name', 'lum', 'lumerr', 'lag', 'lagerrmin', 'lagerrmax', 'z'),'formats':('|S100', np.float, np.float,
#         np.float, np.float, np.float, np.float)},skiprows=1,unpack=True)
#
# '''
# name_new, lum_new, lum_err_new, lag_new, lag_min_err_new, lag_max_err_new, z_new = np.loadtxt("newCIV_theyused.txt",
#     dtype={'names':('name', 'lum', 'lumerr', 'lag', 'lagerrmin', 'lagerrmax', 'z'),'formats':('|S100', np.float, np.float,
#         np.float, np.float, np.float, np.float)},unpack=True)
#
# name_new2, lum_new2, lum_err_new2, lag_new2, lag_min_err_new2, lag_max_err_new2, z_new2 = np.loadtxt("newCIV_notused.txt",
#     dtype={'names':('name', 'lum', 'lumerr', 'lag', 'lagerrmin', 'lagerrmax', 'z'),'formats':('|S100', np.float, np.float,
#         np.float, np.float, np.float, np.float)},unpack=True)
#
# name_new3, lum_new3, lum_err_new3, lag_new3, lag_min_err_new3, lag_max_err_new3, z_new3 = np.loadtxt("newCIV_iused.txt",
#     dtype={'names':('name', 'lum', 'lumerr', 'lag', 'lagerrmin', 'lagerrmax', 'z'),'formats':('|S100', np.float, np.float,
#         np.float, np.float, np.float, np.float)},unpack=True)
#
# lum_new = lum_new*(10**46)
# lum_err_new = lum_err_new*(10**46)
# lag_new_RF = [x/(1+z) for x,z in zip(lag_new, z_new)]
# lag_max_err_new_RF = [x/(1+z) for x,z in zip(lag_max_err_new, z_new)]
# lag_min_err_new_RF = [x/(1+z) for x,z in zip(lag_min_err_new, z_new)]
#
# lum_new2 = lum_new2*(10**46)
# lum_err_new2 = lum_err_new2*(10**46)
# lag_new2_RF = [x/(1+z) for x,z in zip(lag_new2, z_new2)]
# lag_max_err_new2_RF = [x/(1+z) for x,z in zip(lag_max_err_new2, z_new2)]
# lag_min_err_new2_RF = [x/(1+z) for x,z in zip(lag_min_err_new2, z_new2)]
#
# lum_new3 = lum_new3*(10**46)
# lum_err_new3 = lum_err_new3*(10**46)
# lag_new3_RF = [x/(1+z) for x,z in zip(lag_new3, z_new3)]
# lag_max_err_new3_RF = [x/(1+z) for x,z in zip(lag_max_err_new3, z_new3)]
# lag_min_err_new3_RF = [x/(1+z) for x,z in zip(lag_min_err_new3, z_new3)]
# '''
#
#
# lum_prev_10 = [10**x for x in lum_prev]
# lum_err_prev_10 = [10**x - 10**(x-y) for x,y in zip(lum_prev,lum_err_prev)]
#
# lum_us_10 = [10**x for x in lum_us]
# lum_err_us_10 = [10**x - 10**(x-y) for x,y in zip(lum_us,lum_err_us)]
#
# total_lum = np.append(lum_prev, lum_us)
# total_lag = np.append(lag_prev, lag_us)
# total_lag_log = np.log10(total_lag)
# total_lag_err_min = np.append(lag_min_err_prev, lag_min_err_us)
# total_lag_err_max = np.append(lag_max_err_prev, lag_max_err_us)
#
# total_lag_err_log = np.log10(total_lag + total_lag_err_max) - np.log10(total_lag)
# total_lag_err_log_min = np.log10(total_lag) - np.log10(total_lag - total_lag_err_min)
#
# # Our Fit!!!
# # intercept -4.105 +/- .09
# # slope 0.49 pm 0.026
#
# # now -4.13 pm 0.09 0.47 pm 0.03
# c = 3*10**8 #m/s
# ldtm = 2.59*10**13 #m/ld
#
# # 0.47 pm 0.03, 0.807 pm 0.086
# lum_OzDES = np.linspace(38, 48, 5)
# lum_OzDES_10 = [10**x for x in lum_OzDES]
# #lag_OzDES = [-4.13 + 0.47*np.log10(10**x/10**44) for x in lum_OzDES]
# #lag_OzDES_10 = [(ldtm/c)*10**x for x in lag_OzDES]
# lag_OzDES = [0.82 + 0.49*np.log10(10**x/10**44) for x in lum_OzDES]
# lag_OzDES_10 = [10**x for x in lag_OzDES]
#
# # Kaspi 2007
# #FITEXY
# lum_Kaspi = np.linspace(10**38, 10**48, 5)
# lag_Kaspi = np.zeros(5)
# for i in range(5):
#     lag_Kaspi[i] = 10*(0.17)*(lum_Kaspi[i]/10**43)**0.52
#
# #BCES
# lag_Kaspi_v2 = np.zeros(5)
# for i in range(5):
#     lag_Kaspi_v2[i] = 10*(0.24)*(lum_Kaspi[i]/10**43)**0.55
#
#
# # Peterson 2005
# lum_Peterson = np.linspace(10**38, 10**48, 5)
# lag_Peterson = np.zeros(5)
# for i in range(5):
#     lag_Peterson[i] = 10**(1.06 + 0.61*(np.log10(lum_Peterson[i])-44))
#
# #Lira 2018
# lum_Lira = np.linspace(10**38, 10**48, 5)
# lag_Lira = np.zeros(5)
# for i in range(5):
#     lag_Lira[i] = 10*(0.22)*(lum_Kaspi[i]/10**43)**0.46
#
# fig, ax = makeFigSingle("", "$\lambda L_\lambda$(1350$\mathrm{\AA}$) [ergs s$^{-1}$]", "Rest Frame CIV Lags [days]", [0, 0], [10**-2, 10**4])
# fig.set_size_inches(12, 10, forward=True)
# ax.set_xscale("log", nonposx='clip')
# ax.set_yscale("log", nonposy='clip')
# ax.tick_params(axis='x', pad=10)
# ax.locator_params(axis='x', numticks=7)
# ax.set_ylabel("Rest Frame CIV Lags [days]", **axis_font, labelpad=-12)
# ax.set_xlabel("$\lambda L_\lambda$(1350$\mathrm{\AA}$) [ergs s$^{-1}$]", **axis_font, labelpad=-4)
# ax.yaxis.set_major_formatter(FormatStrFormatter('%.5g'))
#
# ax2 = ax.twinx()
# ax2.set_ylabel("CIV Radius [10$^{16}$ cm]", **axis_font, labelpad=-12)
#
# convert = 86400*(3*10**10)/10**16 # convert to seconds, multiply by c, get result in 10^18 cm
# ax2.set_ylim([convert*10**-2, convert*10**3])
# ax2.set_xscale("log", nonposx='clip')
# ax2.set_yscale("log", nonposy='clip')
# ax2.locator_params(axis='x', numticks=7)
# ax2.yaxis.set_major_formatter(FormatStrFormatter('%.5g'))
# for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
#     label.set_fontsize(25)
#
#
# '''
# ax.errorbar(lum_prev_10, lag_prev, yerr=[lag_min_err_prev, lag_max_err_prev], xerr=lum_err_prev_10, fmt='o',
#             color = 'black', markersize = 7, label = 'Existing Measurements')
# ax.errorbar(lum_us_10, lag_us, yerr=[lag_min_err_us, lag_max_err_us], xerr=lum_err_us_10, fmt='s', color = 'blue',
#             markersize = 8, label = 'OzDES Measurements')
# ax.plot(lum_OzDES_10, lag_OzDES_10, color = 'black', linewidth = 3, label = '0.49 '+ r'$\pm$' +' 0.03 OzDES Fit')
# ax.plot(lum_Peterson, lag_Peterson, color = 'black', linewidth = 2, linestyle = ':', label = '0.61 '+ r'$\pm$' +' 0.05 Peterson et al 2005')
# ax.plot(lum_Kaspi, lag_Kaspi, color = 'black', linestyle = '--', label = '0.52 '+ r'$\pm$' +' 0.04 Kaspi et al 2007')
# ax.legend(loc='lower right', frameon=False, numpoints=1, prop={'size':20})
# plt.show()
# '''
#
# #ax.errorbar(lum_new, lag_new_RF, yerr=[lag_min_err_new_RF, lag_max_err_new_RF], xerr=lum_err_new, fmt='o',
# #            color = 'green', markersize = 7, label = "They Use, I Ignore")
# #ax.errorbar(lum_new2, lag_new2_RF, yerr=[lag_min_err_new2_RF, lag_max_err_new2_RF], xerr=lum_err_new2, fmt='o',
# #            color = 'red', markersize = 7, label = "We All Ignore")
# #ax.errorbar(lum_new3, lag_new3_RF, yerr=[lag_min_err_new3_RF, lag_max_err_new3_RF], xerr=lum_err_new3, fmt='o',
# #            color = 'purple', markersize = 7, label = "We All Use")
#
# ax.plot(lum_Peterson, lag_Peterson, color = 'mediumblue',  linewidth = 2, linestyle = ':', label = '0.61 '+ r'$\pm$' +' 0.05 Peterson et al 2005')
# #ax.plot(lum_Kaspi, lag_Kaspi, color = 'mediumblue', linestyle = (0, (3, 5, 1, 5, 1, 5)), linewidth = 3, label = '0.52 '+ r'$\pm$' +' 0.04 Kaspi et al 2007, FITEXY')
# ax.plot(lum_Kaspi, lag_Kaspi_v2, color = 'forestgreen', linestyle = '-.', linewidth = 3, label = '0.55 '+ r'$\pm$' +' 0.04 Kaspi et al 2007, BCSE')
# ax.plot(lum_Lira, lag_Lira, color = 'firebrick', linestyle = '--', linewidth = 2, label = '0.46 '+ r'$\pm$' +' 0.08 Lira et al 2018')
# ax.plot(lum_OzDES_10, lag_OzDES_10, color = 'black', linewidth = 4, label = '0.49 '+ r'$\pm$' +' 0.02 OzDES Fit')
# ax.errorbar(lum_prev_10, lag_prev, yerr=[lag_min_err_prev, lag_max_err_prev], xerr=lum_err_prev_10, fmt='o',
#             mfc = 'none', mec = 'black', color = 'black', markersize = 9, markeredgewidth=1.5, elinewidth=1, label = 'Existing Measurements')
# ax.errorbar(lum_us_10, lag_us, yerr=[lag_min_err_us, lag_max_err_us], xerr=lum_err_us_10, fmt='s', color = 'mediumblue',
#             markersize = 11, elinewidth=2, label = 'OzDES Measurements')
# ax.legend(loc='upper left', frameon=False, numpoints=1, prop={'size':18})
# plt.savefig('../thisIsIt/RL.pdf', format='pdf', dpi=1000)
#
# plt.show()
#
#
# # ---------------------------------- #
# # ------- Latex Table Output ------- #
# # ---------------------------------- #
#
# names = ['DES J0228-04', 'DES J0033-42']
# source = ['2937961955', '2971212466']
#
# # photo = open("../thisIsIt/OzDES_gBand.dat", 'w')
# # spec = open("../thisIsIt/OzDES_CIV.dat", 'w')
#
# for s in range(2):
#     date_line, flux_line, err_line = np.loadtxt("../thisIsIt/data/" + source[s] + "_CIV.txt", unpack=True)
#     date_cont, flux_cont, err_cont = np.loadtxt("../thisIsIt/data/" + source[s] + "_gBand.txt", unpack=True)
#     print(source[s])
#     print(np.std(flux_line)**2)
#     print(np.std(flux_cont)**2)
#
#     # for i in range(len(date_line)):
#     #     spec.write(names[s] + " " + "{0:.2f}".format(round(date_line[i],2)) + " " + "{0:.2f}".format(round(flux_line[i],2))+ " " + "{0:.2f}".format(round(err_line[i],2)) + "\n")
#     # for i in range(len(date_cont)):
#     #     photo.write(names[s] + " " + "{0:.2f}".format(round(date_cont[i],2)) + " " + "{0:.2f}".format(round(flux_cont[i],2))+ " " + "{0:.2f}".format(round(err_cont[i],2)) + "\n")
#     #
#
# # photo.close()
# # spec.close

# names = ['DESJ0228-04', 'DESJ0033-42']
# source = ['2937961955', '2971212466']
#
# #data = open("../thisIsIt/OzDES_allflux.dat", 'w')
#
# for s in range(1):
#     date_line, flux_line, err_line = np.loadtxt("../thisIsIt/data/" + source[s] + "_CIV.txt", unpack=True)
#     date_g, flux_g, err_g = np.loadtxt("../thisIsIt/data/" + source[s] + "_gBand.txt", unpack=True)
#     date_r, flux_r, err_r = np.loadtxt("../thisIsIt/data/" + source[s] + "_rBand.txt", unpack=True)
#     date_i, flux_i, err_i = np.loadtxt("../thisIsIt/data/" + source[s] + "_iBand.txt", unpack=True)
#
#     total = max([len(date_line), len(date_g), len(date_r), len(date_i)])
#     name = names[s]
#
#     #for i in range(total):
#     for i in range(10):
#         if i < len(date_line):
#             dl = "{:8.2f}".format(round(date_line[i],2))
#             fl = "{:5.2f}".format(round(flux_line[i],2)).zfill(2)
#             el = "{:4.2f}".format(round(err_line[i],2))
#         if i > len(date_line):
#             dl = "        "
#             fl = "     "
#             el = "    "
#         if i < len(date_g):
#             dg = "{:8.2f}".format(round(date_g[i],2))
#             fg = "{:5.2f}".format(round(flux_g[i],2))
#             eg = "{:4.2f}".format(round(err_g[i],2))
#         if i > len(date_g):
#             dg = "        "
#             fg = "     "
#             eg = "    "
#         if i < len(date_r):
#             dr = "{:8.2f}".format(round(date_r[i],2))
#             fr = "{:5.2f}".format(round(flux_r[i],2))
#             er = "{:4.2f}".format(round(err_r[i],2))
#         if i > len(date_r):
#             dr = "        "
#             fr = "     "
#             er = "    "
#         if i < len(date_i):
#             di = "{:8.2f}".format(round(date_i[i],2))
#             fi = "{:5.2f}".format(round(flux_i[i],2))
#             ei = "{:4.2f}".format(round(err_i[i],2))
#         if i > len(date_i):
#             di = "        "
#             fi = "     "
#             ei = "    "
#
#         print("%s & %s & %s $\pm$ %s & %s & %s $\pm$ %s & %s & %s $\pm$ %s & %s & %s $\pm$ %s" %(name, dl, fl, el, dg, fg, eg, dr, fr, er, di, fi, ei))
#
#         #data.write("%s %s %s %s %s %s %s %s %s %s %s %s %s \n" %(name, dl, fl, el, dg, fg, eg, dr, fr, er, di, fi, ei))
#
# #data.close()
#
#
#
#
