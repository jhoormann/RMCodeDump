# ---------------------------------------------------------- #
# --------------------- classifySpec.py -------------------- #
# --------- https://github.com/jhoormann/RMCodeDump -------- #
# ---------------------------------------------------------- #
# This code aims to help make it easier to decide which      #
# exposures, if any, are problematic and should be excluded  #
# from the analysis.                                         #
# ---------------------------------------------------------- #

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.integrate import fixed_quad
from matplotlib.ticker import MaxNLocator
import OzDES_Calculation as ozcalc
import pandas as pd
import sys

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

def makeFigSingle(title, xlabel, ylabel, xlim=[0, 0], ylim=[0, 0]):
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
def plot_fonts(size, color='black', weight='normal', align='bottom'):
    font = {'size': size, 'color': color, 'weight': weight, 'verticalalignment': align}
    return font


def plot_ticks(ax, size):
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(size)
    return
def plot_share_x(number, title, xlabel, xdat, ylabel, ydat, qc, xlim=(0, 0), ylim=(0, 0), lines=(0,0,0), asize=22,
                 tsize=22, xdim=10, ydim=10, xtick=None, ytick=2):
    fig, ax_array = plt.subplots(number, sharex=True)
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
            ax.set_ylim(ylim)
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
        if qc[i] == 'good':
            col = 'black'
        else:
            col = 'red'
        ax.plot(xdat, ydat[:,i], color=col)
        ax.axvline(x=lines[0], color='blue')
        ax.axvline(x=lines[1], color='blue')
        ax.axvline(x=lines[2], color='blue')
    return


def mark_as_bad(fluxes, variances, wavelength, numEpochs):
    for epoch in range(numEpochs):
        flux = fluxes[:, epoch]
        variance = variances[:, epoch]

        bad = np.zeros(len(flux))

        for i in range(len(flux)):
            if i % 100 == 0:
                avg = np.nanmean(variance[i:i + 99])
            if np.isnan(variance[i]) == False and variance[i] > 3.5 * avg:
                bad[i] = 2
                flux[i] = np.nan
                if i > 2 and i < 4996:
                    flux[i - 1] = np.nan
                    flux[i - 2] = np.nan
                    flux[i - 3] = np.nan
                    flux[i + 1] = np.nan
                    flux[i + 2] = np.nan
                    flux[i + 3] = np.nan

                    bad[i - 1] = 1
                    bad[i - 2] = 1
                    bad[i - 3] = 1
                    bad[i + 1] = 1
                    bad[i + 2] = 1
                    bad[i + 3] = 1

    return


def filter_bad_pixels(fluxes, variances, number):

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


# Either a data table with the column names below or just ID number, what you are expected to provide depends on the
# flag you choose
sources = pd.read_csv("specDetsClass.csv")

# I made a table with a bunch of data for each source (makeClassificationStats.py) these are the columns it expects.
# This code focuses on filling out the class, issue columns
# class (good or bad)
# issue (okay (good) or reason why bad)
# colNames = ['ID', 'z', 'ext', 'date', 'mg', 'mgerr', 'Fvarg', 'mr', 'mrerr', 'Fvarr', 'mi', 'mierr', 'Fvari', 'Fc',
# 'SNRc', 'Fm', 'SNRm', 'Fh', 'SNRh', 'F1350', 'SNR1350', 'F3000', 'SNR3000', 'F5100', 'SNR5100', 'Fred', 'SNRred',
# 'Fblue', 'SNRblue', 'badpix', 'class', 'issue']

dataPath = "../OzDES_Data/"

# If you include output from makeClassificationStats.py there will be multiple entries for each AGN ID, however
# we don't want to repeat ourselves so we just find the unique values and analyse those.
names = np.unique(sources['ID'].values)
nLines = len(names)

# There are three flags to choose from 'table', 'plot', and 'combine'.  'plot' will likely be the most useful
flag = 'plot'

# Helps to determine if there are exposure that are so bad they should be excluded from the coadd, noise dominated,
# issues with splicing, etc.  All the exposures for a given run will pop up and you can interactively specify via the
# command line which ones you want to exclude and why.  The results are appended to a text file.
if flag == 'plot':
    output = open("probNames.txt", "a")
    # Initialize while loop with index of names array you want to start at
    i = 0

    # labels for the y axis corresponding to each exposure of a given run, I think this is enough (yes there has to be
    # a better way to do this, no I am not worrying about it now).
    ylab = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    while i < nLines:
        print(i)
        agn = names[i]
        print(agn)

        # Read in the spectral data
        spectra = Spectrumv14(dataPath + "spectra180413/SVA1_COADD-" + str(agn) + ".fits")
        ozcalc.mark_as_bad(spectra.flux, spectra.variance)

        # You probably want to read in the file that includes the classifications made with the 'table' flag.  If you
        # do that spectra that already classified as bad will be plotted using a different color.
        quality = []
        for e in range(spectra.numEpochs):
            iVal = sources.index[(sources['ID'] == agn) & (sources['ext'] == 3 * e + 3)].values[0]
            quality.append(sources['class'].iloc[iVal])
            z = sources['z'].iloc[iVal]

        lines = [1549*(1+z), 2798*(1+z), 4861*(1+z)]


        e = 0
        while e < spectra.numEpochs:
            # Determine which extensions belong to the same observing run
            step = 0
            numStep = step
            while step < spectra.numEpochs - e:
                if spectra.run[e+step] == spectra.run[e]:
                    step += 1
                    numStep = step
                else:
                    step = spectra.numEpochs
            # Make it so all spectra are plotted with the same y-axis limits.
            maxV = np.nanmax(spectra.flux[:, e:e+numStep].flatten()) + 0.5
            minV = np.nanmin(spectra.flux[:, e:e+numStep].flatten()) - 0.5
            if maxV > 15:
                maxV = 15
            if minV < -5:
                minV = -5

            # Plot all epochs on the same graph.  If there is already a known issue mark plot the spectra as red,
            # otherwise plot it in black

            if numStep > 1:
                plot_share_x(numStep, str(agn) + " run " + str(spectra.run[e]), "Wavelength", spectra.wavelength,
                         ylab[0:numStep], spectra.flux[:, e:e+numStep], quality[e:e+numStep],
                             [spectra.wavelength[0], spectra.wavelength[4999]], [minV, maxV], lines)
            else:
                fig, ax = makeFigSingle(str(agn) + " run " + str(spectra.run[e]), "Wavelength","0",
                                        [spectra.wavelength[0], spectra.wavelength[4999]])
                if quality[e] == 'good':
                    col = 'black'
                else:
                    col = 'red'
                ax.plot(spectra.wavelength, spectra.flux[:,e], color=col)
                ax.axvline(x=lines[0], color='blue')
                ax.axvline(x=lines[1], color='blue')
                ax.axvline(x=lines[2], color='blue')

            # While the graph is open you will be prompted to specify if there are any issues with the run.  If you say
            # yes then you will go through each extension in the run one by one and you can say if there is a problem
            # with the specific extension.  You can specify four options 'w' = weather/noise, 's' = splice,
            # 'm' = missing (lots of bad pix, note we call mark_as_bad so this might mean there was a lot of noise
            # that were also interpolated over), or 'o' = other
            # If when asked if the run is bad you choose 'q' it will save the info up to this point to the output file
            # and clos edown
            plt.pause(0.1)
            b = 0
            run_flag = input("Is the run bad (y/n)? ")
            if run_flag == 'y':
                while b < numStep:
                    ext_flag = input("Is extension " + str(b) + " bad (n/y(w/s/m/o))? ")
                    if ext_flag in ['w', 's', 'm', 'o']:
                        output.write(str(agn) + " " + str(spectra.ext[e+b]) + " " + ext_flag + "\n")
                    if ext_flag == 'q':
                        b = numStep + 1
                    b = b + 1
            if run_flag == 'q':
                e = spectra.numEpochs
                print("I am quitting at index " + str(i))
                output.close()
                sys.exit()
            plt.close()
            e += numStep

        i += 1

    output.close()

# Classify a exposure as bad based on things like, bad quality flag, not enough photometry to calibrate, bad weather,
# and bad pixels, many of these include issue were picked up when trying to perform the spectrophotometric calibration
if flag == 'table':
    for i in range(nLines):
        if i%10 ==0:
            print("Analysing AGN # " + str(i))
        AGN = names[i]
        print(AGN)

        # Load in data
        spectra = ozcalc.SpectrumCoadd(dataPath + "processedSpectraY5/" + str(names[i]) + "_scaled.fits")
        spectraO = ozcalc.Spectrumv14(dataPath + "spectra180413/SVA1_COADD-" + str(AGN) + ".fits")

        photo = pd.read_table(dataPath + "photometryY5/" + str(AGN) + "_lc.dat", delim_whitespace=True)

        # bad if the quality flag was determined to be problematic (ie anything other than okay or backup),
        # issue  = 'qc'
        if spectra.badqc != '':
            badqc = np.array(spectra.badqc.split(","))
            badqc = badqc.astype(int)

            for q in badqc:
                iVal = sources.index[(sources['ID'] == AGN) & (sources['ext'] == q)].values[0]
                sources['class'].iloc[iVal] = 'bad'
                sources['issue'].iloc[iVal] = 'qc'

        # bad if there is insufficient photometry to calibrate
        # issue = 'nophoto'
        if spectra.nophoto != '':
            nophoto = np.array(spectra.nophoto.split(","))
            nophoto = nophoto.astype(int)

            for p in nophoto:
                iVal = sources.index[(sources['ID'] == AGN) & (sources['ext'] == p)].values[0]
                sources['class'].iloc[iVal] = 'bad'
                sources['issue'].iloc[iVal] = 'nophoto'

        # bad if there was a weather issue, typically so noisy magnitudes calculated in one of the bands during
        # calibration were nans
        # issue = 'noise'
        if spectra.weather != '':
            weather = np.array(spectra.weather.split(","))
            weather = weather.astype(int)

            for w in weather:
                iVal = sources.index[(sources['ID'] == AGN) & (sources['ext'] == w)].values[0]
                sources['class'].iloc[iVal] = 'bad'
                sources['issue'].iloc[iVal] = 'noise'

        # bad if the emission line flux is negative
        # issue = 'noise'
        # also bad if more than 10% of the pixels were bad
        # issue = 'missing'
        for e in range(spectraO.numEpochs):
            iVal = sources.index[(sources['ID'] == AGN) & (sources['ext'] == 3*e+3)].values[0]

            if sources['Fh'].iloc[iVal] < 0:
                sources['class'].iloc[iVal] = 'bad'
                sources['issue'].iloc[iVal] = 'noise'
            elif sources['Fm'].iloc[iVal] < 0:
                sources['class'].iloc[iVal] = 'bad'
                sources['issue'].iloc[iVal] = 'noise'
            elif sources['Fc'].iloc[iVal] < 0:
                sources['class'].iloc[iVal] = 'bad'
                sources['issue'].iloc[iVal] = 'noise'
            elif sources['class'].iloc[iVal] != 'bad':
                if sources['badpix'].iloc[iVal] / 5000 > 0.10:
                    sources['class'].iloc[iVal] = 'bad'
                    sources['issue'].iloc[iVal] = 'missing'
                else:
                    sources['class'].iloc[iVal] = 'good'
                    sources['issue'].iloc[iVal] = 'okay'
        sources.to_csv("specDetsClass.csv", index=False)


# This will read in the output table created from 'plot' and add those class/issues to the larger table created with
# 'table' and makeClassificationStats.py
if flag == 'combine':
    classList = pd.read_table("probNames.txt", delim_whitespace=True)
    for l in range(len(classList)):
        AGN = classList['ID'].iloc[l]
        ext = classList['ext'].iloc[l]
        issue = classList['issue'].iloc[l]
        if issue == 'w':
            prob = 'noise'
        elif issue == 's':
            prob = 'splice'
        elif issue == 'm':
            prob = 'missing'
        else:
            prob = 'other'

        iVal = sources.index[(sources['ID'] == AGN) & (sources['ext'] == ext)].values[0]

        if sources['class'].iloc[iVal] == 'good':
            sources['class'].iloc[iVal] = 'bad'
            sources['issue'].iloc[iVal] = prob

    sources.to_csv("specDetsModel.csv", index=False)
