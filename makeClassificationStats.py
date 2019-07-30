# ---------------------------------------------------------- #
# ---------------- makeClassificationStats.py -------------- #
# --------- https://github.com/jhoormann/RMCodeDump -------- #
# ---------------------------------------------------------- #
# Here I will create a table with various stats describing   #
# an AGN, variation, magnitudes, SNR ratio for various bands #
# and emission lines.  Spectral statistics are calculated    #
# for each exposure, before calibration.                     #
# ---------------------------------------------------------- #
import numpy as np
import pandas as pd
import OzDES_Calculation as ozcalc
import matplotlib.cm as cm
import matplotlib.pyplot as plt

title_font = {'size':'12', 'color':'black', 'weight':'normal', 'verticalalignment':'bottom'}
axis_font = {'size':'12'}

# Main direction where to find data
dataPath = "../OzDES_Data/"

# File listing the AGN ID's
sources = pd.read_table(dataPath + "RM_IDs.txt", delim_whitespace=True)
nsources = len(sources)
randAGN = nsources*np.random.rand(100)
randAGN = randAGN.astype(int)
randAGN = np.unique(randAGN)
used = []

colNames = ['ID', 'z', 'ext', 'date', 'mg', 'mgerr', 'Fvarg', 'mr', 'mrerr', 'Fvarr', 'mi', 'mierr', 'Fvari', 'Fc',
            'SNRc', 'Fm', 'SNRm', 'Fh', 'SNRh', 'F1350', 'SNR1350', 'F3000', 'SNR3000', 'F5100', 'SNR5100', 'Fred',
            'SNRred', 'Fblue', 'SNRblue', 'badpix', 'class', 'issue']

# Class and Issue labels can be determined using the visualization code, classifySpec.py


df = pd.DataFrame(columns=colNames)

for i in range(len(randAGN)):
    AGN = sources['ID'].iloc[randAGN[i]]

    if AGN not in used:
        # Load in data, change paths here
        spectra = ozcalc.Spectrumv14(dataPath + "spectra180413/SVA1_COADD-" + str(AGN) + ".fits")
        specCalib = ozcalc.SpectrumCoadd(dataPath + "processedSpectraY5/" + str(AGN) + "_scaled.fits")
        photo = pd.read_table(dataPath + "photometryY5/" + str(AGN) + "_lc.dat", delim_whitespace=True)

        z = specCalib.redshift

        ozcalc.filter_bad_pixels(spectra.flux, spectra.variance)

        # Calculate mean mag, uncertainty and variation for each band

        if 'g' in photo['BAND'].values:
            mg = photo[photo['BAND'] == 'g']['MAG'].mean()
            mgerr = photo[photo['BAND'] == 'g']['MAGERR'].mean()
            Fvarg = ozcalc.variability(photo[photo['BAND'] == 'g']['MAG'].values,
                                       photo[photo['BAND'] == 'g']['MAGERR'].values)[0]
        else:
            mg = 0
            mgerr = 0
            Fvarg = 0

        if 'r' in photo['BAND'].values:
            mr = photo[photo['BAND'] == 'r']['MAG'].mean()
            mrerr = photo[photo['BAND'] == 'r']['MAGERR'].mean()
            Fvarr = ozcalc.variability(photo[photo['BAND'] == 'r']['MAG'].values,
                                       photo[photo['BAND'] == 'r']['MAGERR'].values)[0]
        else:
            mr = 0
            mrerr = 0
            Fvarr = 0

        if 'i' in photo['BAND'].values:
            mi = photo[photo['BAND'] == 'i']['MAG'].mean()
            mierr = photo[photo['BAND'] == 'i']['MAGERR'].mean()
            Fvari = ozcalc.variability(photo[photo['BAND'] == 'i']['MAG'].values,
                                       photo[photo['BAND'] == 'i']['MAGERR'].values)[0]
        else:
            mi = 0
            mierr = 0
            Fvari = 0

        # Find emission lines, only considering CIV, MgII, and Hbeta

        # Note line_ref[0] = CIV, 2 = MgII, 3 = Hbeta

        if 0 < z < 0.6972:
            line_ref = [2]
        elif 0.6972 < z < 0.6988:
            line_ref = [2, 1]
        elif 0.6988 < z < 1.5784:
            line_ref = [1]
        elif 1.5784 < z < 1.8786:
            line_ref = [1, 0]
        elif 1.8786 < z < 3.8352:
            line_ref = [0]
        else:
            line_ref = [-9]

        if 0 in line_ref:
            contMinCIV = [1450 * (1 + z), 1460 * (1 + z)]
            contMaxCIV = [1780 * (1 + z), 1790 * (1 + z)]
            lineMinCIV = ozcalc.findBin(1470 * (1 + z), spectra.wavelength)
            lineMaxCIV = ozcalc.findBin(1595 * (1 + z), spectra.wavelength)
        else:
            Fc = 0
            SNRc = 0

        if 1 in line_ref:
            contMinMgII = [2190 * (1 + z), 2210 * (1 + z)]
            contMaxMgII = [3007 * (1 + z), 3027 * (1 + z)]
            lineMinMgII = ozcalc.findBin(2700 * (1 + z), spectra.wavelength)
            lineMaxMgII = ozcalc.findBin(2920 * (1 + z), spectra.wavelength)
        else:
            Fm = 0
            SNRm = 0

        if 2 in line_ref:
            contMinHb = [4760 * (1 + z), 4790 * (1 + z)]
            contMaxHb = [5100 * (1 + z), 5130 * (1 + z)]
            lineMinHb = ozcalc.findBin(4810 * (1 + z), spectra.wavelength)
            lineMaxHb = ozcalc.findBin(4940 * (1 + z), spectra.wavelength)
        else:
            Fh = 0
            SNRh = 0

        fluxesC = np.copy(spectra.flux)
        variancesC = np.copy(spectra.variance)

        fluxesM = np.copy(spectra.flux)
        variancesM = np.copy(spectra.variance)

        fluxesH = np.copy(spectra.flux)
        variancesH = np.copy(spectra.variance)

        # Find if luminosity values associated with each lines are present

        # lum_ref = 0 = 1350, 1 = 3000, 2 = 5100
        if 0 < z < 0.24:
            lum_ref = [2]
        elif 0.24 < z < 0.73:
            lum_ref = [1, 2]
        elif 0.73 < z < 1.74:
            lum_ref = [1]
        elif 1.74 < z < 1.93:
            lum_ref = [1, 0]
        elif z > 1.93:
            lum_ref = [0]
        else:
            lum_ref = [-9]

        if 0 in lum_ref:
            lumMin1350 = ozcalc.findBin(1345 * (1 + z), spectra.wavelength)
            lumMax1350 = ozcalc.findBin(1355 * (1 + z), spectra.wavelength)
        else:
            F1350 = 0
            SNR1350 = 0

        if 1 in lum_ref:
            lumMin3000 = ozcalc.findBin(2995 * (1 + z), spectra.wavelength)
            lumMax3000 = ozcalc.findBin(3005 * (1 + z), spectra.wavelength)
        else:
            F3000 = 0
            SNR3000 = 0

        if 2 in lum_ref:
            lumMin5100 = ozcalc.findBin(5095 * (1 + z), spectra.wavelength)
            lumMax5100 = ozcalc.findBin(5105 * (1 + z), spectra.wavelength)
        else:
            F5100 = 0
            SNR5100 = 0


        blueMin = ozcalc.findBin(5590, spectra.wavelength)
        blueMax = ozcalc.findBin(5690, spectra.wavelength)

        redMin = ozcalc.findBin(5710, spectra.wavelength)
        redMax = ozcalc.findBin(5810, spectra.wavelength)

        # Calculate flux, variance, and SNR for each line and luminosity

        for e in range(spectra.numEpochs):
            ext = e*3 + 3

            if 0 in line_ref:
                ozcalc.cont_fit_reject(spectra.wavelength, fluxesC, variancesC, contMinCIV, contMaxCIV)
                Fc = np.sum(fluxesC[lineMinCIV:lineMaxCIV, e])
                Vc = np.sum(variancesC[lineMinCIV:lineMaxCIV, e])
                SNRc = Fc/pow(Vc, 0.5)

            if 1 in line_ref:
                ozcalc.cont_fit_reject(spectra.wavelength, fluxesM, variancesM, contMinMgII, contMaxMgII)
                Fm = np.sum(fluxesM[lineMinMgII:lineMaxMgII, e])
                Vm = np.sum(variancesM[lineMinMgII:lineMaxMgII, e])
                SNRm = Fm / pow(Vm, 0.5)

            if 2 in line_ref:
                ozcalc.cont_fit_reject(spectra.wavelength, fluxesH, variancesH, contMinHb, contMaxHb)
                Fh = np.sum(fluxesH[lineMinHb:lineMaxHb, e])
                Vh = np.sum(variancesH[lineMinHb:lineMaxHb, e])
                SNRh = Fh / pow(Vh, 0.5)

            if 0 in lum_ref:
                F1350 = np.sum(spectra.flux[lumMin1350:lumMax1350, e])
                V1350 = np.sum(spectra.variance[lumMin1350:lumMax1350, e])
                SNR1350 = F1350 / pow(V1350, 0.5)

            if 1 in lum_ref:
                F3000 = np.sum(spectra.flux[lumMin3000:lumMax3000, e])
                V3000 = np.sum(spectra.variance[lumMin3000:lumMax3000, e])
                SNR3000 = F3000 / pow(V3000, 0.5)

            if 2 in lum_ref:
                F5100 = np.sum(spectra.flux[lumMin5100:lumMax5100, e])
                V5100 = np.sum(spectra.variance[lumMin5100:lumMax5100, e])
                SNR5100 = F5100 / pow(V5100, 0.5)

            # Calculate flux, variance, and SNR for regions on each side of red/blue splice

            Fblue = np.sum(spectra.flux[blueMin:blueMax, e])
            Vblue = np.sum(spectra.variance[blueMin:blueMax, e])
            SNRblue = Fblue / pow(Vblue, 0.5)

            Fred = np.sum(spectra.flux[redMin:redMax, e])
            Vred = np.sum(spectra.variance[redMin:redMax, e])
            SNRred = Fred / pow(Vred, 0.5)

            # Determine number of bad pixels

            badpix = np.sum(spectra.badpix[:, e])

            df2 = pd.DataFrame([[AGN, z, ext, spectra.dates[e], mg, mgerr, Fvarg, mr, mrerr, Fvarr, mi, mierr, Fvari,
                                 Fc, SNRc,Fm, SNRm, Fh, SNRh, F1350, SNR1350, F3000, SNR3000, F5100, SNR5100, Fred,
                                 SNRred, Fblue, SNRblue, badpix, 0, 0]], columns=colNames)
            df = df.append(df2)


df.to_csv("specDets.csv", index=False)


