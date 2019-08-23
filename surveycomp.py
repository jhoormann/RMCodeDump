# ---------------------------------------------------------- #
# ---------------------- surveycomp.py --------------------- #
# --------- https://github.com/jhoormann/RMCodeDump -------- #
# ---------------------------------------------------------- #
# Figure with SDSS RM comparison:                            #
# got data from Anthea                                       #
# This is the code I got from Paul Martini to make the lag   #
# versus redshift plots.  The data tables called here are in #
# the OzDES_RMData_O82019/otherSurveyResults folder linked   #
# on the OzDES wiki.                                         #
# ---------------------------------------------------------- #

import numpy as np 
from astropy.io import ascii
import matplotlib.pyplot as mpl
dat = ascii.read('OzDES_Predict_King.txt')
minmask = np.ones(len(dat), dtype=bool)
for i in range(len(dat)):
  if dat['lag_hb'][i] <= 30. and dat['lag_hb'][i] >= 0.:
    minmask[i] = False
  if dat['lag_hb'][i] >= 0. and dat['lagErrLower_hb'][i]/dat['lag_hb'][i] < 0.05:
    minmask[i] = False
  if dat['lag_mg'][i] >= 0. and dat['lagErrLower_mg'][i]/dat['lag_mg'][i] < 0.05:
    minmask[i] = False
  if dat['lag_civ'][i] >= 0. and dat['lagErrLower_civ'][i]/dat['lag_civ'][i] < 0.05:
    minmask[i] = False

hberr = np.array([dat['lagErrLower_hb'][minmask],dat['lagErrUpper_hb'][minmask]])
mgerr = np.array([dat['lagErrLower_mg'][minmask],dat['lagErrUpper_mg'][minmask]])
civerr = np.array([dat['lagErrLower_civ'][minmask],dat['lagErrUpper_civ'][minmask]])

sdss = ascii.read('shen15b_lags.dat') 
sdsserr = np.array([sdss['laglow'], sdss['lagupp']]) 

# Name    z        laghb  laglow  lagupp
gr = ascii.read('grier13.dat') 
grerr = np.array([gr['laglow'], gr['lagupp']]) 

# Name    z        laghb  laglow  lagupp
gr17 = ascii.read('grier17.dat')
gr17err = np.array([gr17['laglow']*(1+gr17['z']), gr17['lagupp']*(1+gr17['z'])])

# Name    z        laghb  laglow  lagupp
civ = ascii.read('existingCIV.txt')
civ2err = np.array([civ['laglow']*(1+civ['z']), civ['lagupp']*(1+civ['z'])])


fig = mpl.figure()
fig.set_size_inches(10, 8, forward=True)
mpl.yscale('log', basey=10.)
mpl.xticks([0,1,2,3,4],[0,1,2,3,4], fontsize=14)
mpl.yticks([1,10,100,1000],[1,10,100,1000], fontsize=14)
mpl.plot([0.,4.], [180., 180.], color='firebrick', ls='--', lw=1)
mpl.errorbar(gr17['z'], gr17['laghb']*(1+gr17['z']), xerr=None, yerr=gr17err, fmt='^', ms=5, mfc='grey', mec='grey',
             alpha=0.7, ecolor='grey', label="")
mpl.errorbar(sdss['z'], sdss['lag'], xerr=None, yerr=sdsserr, fmt='^', ms=5, mfc='grey', mec='grey', alpha=0.7,
             ecolor='grey', label="")
mpl.errorbar(gr['z'], gr['laghb'], xerr=None, yerr=grerr, fmt='^', ms=5, mfc='grey', mec='grey', alpha=0.7,
             ecolor='grey', label="")
mpl.errorbar(civ['z'], civ['laghb']*(1+civ['z']), xerr=None, yerr=civ2err, fmt='^', ms=5, mfc='grey', mec='grey',
             alpha=0.7, ecolor='grey', label="Previous Surveys")


mpl.xlabel('Redshift', fontsize=18)
mpl.ylabel('Observed Lag [days]', fontsize=18)
mpl.xlim([0,4])
mpl.ylim([1,2000])
mpl.legend(loc='lower right', ncol=1, numpoints=1, frameon = False, fontsize=12)

mpl.show()

