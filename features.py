# ---------------------------------------------------------- #
# ----------------------- features.py ---------------------- #
# --------- https://github.com/jhoormann/RMCodeDump -------- #
# ---------------------------------------------------------- #
# This is a code which will try and determine if a light     #
# curve contains any distinctive features which makes it     #
# easier to recover lags.                                    #
# ---------------------------------------------------------- #

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import OzDES_Calculation as ozcalc
import OzDES_Plotting as ozplot

photoPath = "../OzDES_Data/photometryY5/"
photoEnd = "_lc.dat"

sources = pd.read_table('../OzDES_Data/RM_IDs.txt', delim_whitespace=True)
newCols = pd.DataFrame(columns=['mean', 'std', 'diff', 'per', 'meanSV', 'stdSV', 'dateSV', 'meanY1', 'stdY1', 'dateY1',
                                'meanY2', 'stdY2', 'dateY2', 'meanY3', 'stdY3', 'dateY3', 'meanY4', 'stdY4', 'dateY4',
                                'meanY5', 'stdY5', 'dateY5', 'dY1', 'dY2', 'dY3', 'dY4', 'dY5', 'dtot', 'sY1', 'sY2',
                                'sY3', 'sY4', 'sY5', 'cY2', 'cY3', 'cY4', 'cY5', 'ctot'])
sources = pd.concat([sources, newCols], sort=True)

# Dates separating the observing seasons
dateSVY1 = 56400
dateY1Y2 = 56800
dateY2Y3 = 57150
dateY3Y4 = 57510
dateY4Y5 = 57900

for s in range(len(sources)):
    AGN = str(int(sources['ID'].iloc[s]))

    photo = pd.read_table(photoPath + AGN + photoEnd, delim_whitespace=True)
    # I am just worrying about the g band right now
    photo_g = photo[photo['BAND'] == 'g']

    # Calculate mean, standard deviation, percent differences, and difference between max and min values for whole
    # light curve
    sources['mean'].iloc[s] = np.mean(photo_g['MAG'])
    sources['std'].iloc[s] = np.std(photo_g['MAG'])
    sources['diff'].iloc[s] = ozcalc.diffVar(photo_g['MAG'])
    sources['per'].iloc[s] = ozcalc.perVar(photo_g['MAG'])

    # Calculate mean magnitude, standard deviation, and mean date for each observing season
    sources['meanSV'].iloc[s] = np.mean(photo_g[photo_g['MJD'] < dateSVY1]['MAG'])
    sources['stdSV'].iloc[s] = np.std(photo_g[photo_g['MJD'] < dateSVY1]['MAG'])
    sources['dateSV'].iloc[s] = np.mean(photo_g[photo_g['MJD'] < dateSVY1]['MJD'])

    sources['meanY1'].iloc[s] = np.mean(photo_g[(photo_g['MJD'] > dateSVY1) & (photo_g['MJD'] < dateY1Y2)]['MAG'])
    sources['stdY1'].iloc[s] = np.std(photo_g[(photo_g['MJD'] > dateSVY1) & (photo_g['MJD'] < dateY1Y2)]['MAG'])
    sources['dateY1'].iloc[s] = np.mean(photo_g[(photo_g['MJD'] > dateSVY1) & (photo_g['MJD'] < dateY1Y2)]['MJD'])

    sources['meanY2'].iloc[s] = np.mean(photo_g[(photo_g['MJD'] > dateY1Y2) & (photo_g['MJD'] < dateY2Y3)]['MAG'])
    sources['stdY2'].iloc[s] = np.std(photo_g[(photo_g['MJD'] > dateY1Y2) & (photo_g['MJD'] < dateY2Y3)]['MAG'])
    sources['dateY2'].iloc[s] = np.mean(photo_g[(photo_g['MJD'] > dateY1Y2) & (photo_g['MJD'] < dateY2Y3)]['MJD'])

    sources['meanY3'].iloc[s] = np.mean(photo_g[(photo_g['MJD'] > dateY2Y3) & (photo_g['MJD'] < dateY3Y4)]['MAG'])
    sources['stdY3'].iloc[s] = np.std(photo_g[(photo_g['MJD'] > dateY2Y3) & (photo_g['MJD'] < dateY3Y4)]['MAG'])
    sources['dateY3'].iloc[s] = np.mean(photo_g[(photo_g['MJD'] > dateY2Y3) & (photo_g['MJD'] < dateY3Y4)]['MJD'])

    sources['meanY4'].iloc[s] = np.mean(photo_g[(photo_g['MJD'] > dateY3Y4) & (photo_g['MJD'] < dateY4Y5)]['MAG'])
    sources['stdY4'].iloc[s] = np.std(photo_g[(photo_g['MJD'] > dateY3Y4) & (photo_g['MJD'] < dateY4Y5)]['MAG'])
    sources['dateY4'].iloc[s] = np.mean(photo_g[(photo_g['MJD'] > dateY3Y4) & (photo_g['MJD'] < dateY4Y5)]['MJD'])

    sources['meanY5'].iloc[s] = np.mean(photo_g[(photo_g['MJD'] > dateY4Y5)]['MAG'])
    sources['stdY5'].iloc[s] = np.std(photo_g[(photo_g['MJD'] > dateY4Y5)]['MAG'])
    sources['dateY5'].iloc[s] = np.mean(photo_g[(photo_g['MJD'] > dateY4Y5)]['MJD'])

    # This isn't done terribly efficiently so I am not checking if there is data missing in an observing season.
    # If data is missing it will break because of the nans so I will just set the mean, date, and std to 0 which may
    # make it look like there is features where there isn't really any.  Be careful but for sims you should have all
    # this data.  Occasionally you don't have everything with the real data.

    sources[['dateSV', 'dateY1', 'dateY2', 'dateY3', 'dateY4', 'dateY5', 'meanSV', 'meanY1', 'meanY2', 'meanY3',
             'meanY4', 'meanY5', 'stdSV', 'stdY1', 'stdY2', 'stdY3', 'stdY4', 'stdY5']] = \
        sources[['dateSV', 'dateY1', 'dateY2', 'dateY3', 'dateY4', 'dateY5', 'meanSV', 'meanY1', 'meanY2', 'meanY3',
                 'meanY4', 'meanY5', 'stdSV', 'stdY1', 'stdY2', 'stdY3', 'stdY4', 'stdY5']].fillna(0)

    # Next check if the mean from adjacent observing seasons is significantly different (outside of std) than its
    # neighbours.  This will let you know if there is a significant increase between years
    if sources['meanSV'].iloc[s] > sources['meanY1'].iloc[s] + sources['stdY1'].iloc[s] or \
            sources['meanSV'].iloc[s] < sources['meanY1'].iloc[s] - sources['stdY1'].iloc[s]:
        sources['dY1'].iloc[s] = 1
    else:
        sources['dY1'].iloc[s] = 0

    if sources['meanY1'].iloc[s] > sources['meanY2'].iloc[s] + sources['stdY2'].iloc[s] or \
            sources['meanY1'].iloc[s] < sources['meanY2'].iloc[s] - sources['stdY2'].iloc[s]:
        sources['dY2'].iloc[s] = 1
    else:
        sources['dY2'].iloc[s] = 0

    if sources['meanY2'].iloc[s] > sources['meanY3'].iloc[s] + sources['stdY3'].iloc[s] or \
            sources['meanY2'].iloc[s] < sources['meanY3'].iloc[s] - sources['stdY3'].iloc[s]:
        sources['dY3'].iloc[s] = 1
    else:
        sources['dY3'].iloc[s] = 0

    if sources['meanY3'].iloc[s] > sources['meanY4'].iloc[s] + sources['stdY4'].iloc[s] or \
            sources['meanY3'].iloc[s] < sources['meanY4'].iloc[s] - sources['stdY4'].iloc[s]:
        sources['dY4'].iloc[s] = 1
    else:
        sources['dY4'].iloc[s] = 0

    if sources['meanY4'].iloc[s] > sources['meanY5'].iloc[s] + sources['stdY5'].iloc[s] or \
            sources['meanY4'].iloc[s] < sources['meanY5'].iloc[s] - sources['stdY5'].iloc[s]:
        sources['dY5'].iloc[s] = 1
    else:
        sources['dY5'].iloc[s] = 0

    # Total number of big jumps
    sources['dtot'].iloc[s] = sources['dY1'].iloc[s] + sources['dY2'].iloc[s] + sources['dY3'].iloc[s] + \
                              sources['dY4'].iloc[s] + sources['dY5'].iloc[s]

    # Fit a line between the mean of season and save the slope
    sources['sY1'].iloc[s] = np.polyfit([sources['dateSV'].iloc[s], sources['dateY1'].iloc[s]],
                                        [sources['meanSV'].iloc[s], sources['meanY1'].iloc[s]], 1)[0]
    sources['sY2'].iloc[s] = np.polyfit([sources['dateY1'].iloc[s], sources['dateY2'].iloc[s]],
                                        [sources['meanY1'].iloc[s], sources['meanY2'].iloc[s]], 1)[0]
    sources['sY3'].iloc[s] = np.polyfit([sources['dateY2'].iloc[s], sources['dateY3'].iloc[s]],
                                        [sources['meanY2'].iloc[s], sources['meanY3'].iloc[s]], 1)[0]
    sources['sY4'].iloc[s] = np.polyfit([sources['dateY3'].iloc[s], sources['dateY4'].iloc[s]],
                                        [sources['meanY3'].iloc[s], sources['meanY4'].iloc[s]], 1)[0]
    sources['sY5'].iloc[s] = np.polyfit([sources['dateY4'].iloc[s], sources['dateY5'].iloc[s]],
                                        [sources['meanY4'].iloc[s], sources['meanY5'].iloc[s]], 1)[0]

    # Determine if the slope changes sign between seasons if one has a significant feature dY# = 1
    if sources['dY1'].iloc[s] == 1 or sources['dY2'].iloc[s] == 1:
        if np.sign(sources['sY1'].iloc[s]) != np.sign(sources['sY2'].iloc[s]):
            sources['cY2'].iloc[s] = 1
        else:
            sources['cY2'].iloc[s] = 0
    else:
        sources['cY2'].iloc[s] = 0

    if sources['dY2'].iloc[s] == 1 or sources['dY3'].iloc[s] == 1:
        if np.sign(sources['sY2'].iloc[s]) != np.sign(sources['sY3'].iloc[s]):
            sources['cY3'].iloc[s] = 1
        else:
            sources['cY3'].iloc[s] = 0
    else:
        sources['cY3'].iloc[s] = 0

    if sources['dY3'].iloc[s] == 1 or sources['dY4'].iloc[s] == 1:
        if np.sign(sources['sY3'].iloc[s]) != np.sign(sources['sY4'].iloc[s]):
            sources['cY4'].iloc[s] = 1
        else:
            sources['cY4'].iloc[s] = 0
    else:
        sources['cY4'].iloc[s] = 0

    if sources['dY4'].iloc[s] == 1 or sources['dY5'].iloc[s] == 1:
        if np.sign(sources['sY4'].iloc[s]) != np.sign(sources['sY5'].iloc[s]):
            sources['cY5'].iloc[s] = 1
        else:
            sources['cY5'].iloc[s] = 0
    else:
        sources['cY5'].iloc[s] = 0

    # Total number of turning points
    sources['ctot'].iloc[s] = sources['cY2'].iloc[s] + sources['cY3'].iloc[s] + sources['cY4'].iloc[s] + \
                              sources['cY5'].iloc[s]

    # Now lets make some plots, here is a light curve which has the mean values plotted
    # Grey - all LC data points
    # Red - mean for whole LC
    # Blue - mean for each season

    maxY = max(photo_g['MAG'] + photo_g['MAGERR'])
    minY = min(photo_g['MAG'] - photo_g['MAGERR'])
    height = maxY-minY

    maxX = max(photo_g['MJD'] + 150)
    minX = min(photo_g['MJD'] - 150)

    fig, ax = ozplot.makeFigSingle(AGN, "Date", "Magnitude", [minX, maxX], [minY - 0.05*height, maxY + 0.2*height])
    ax.xaxis.major.locator.set_params(nbins=5)
    ax.errorbar(photo_g['MJD'], photo_g['MAG'], yerr=photo_g['MAGERR'], fmt='s', color='grey', alpha=0.7,
                markersize=4, elinewidth=2)
    ax.errorbar(sources['dateSV'].iloc[s], sources['meanSV'].iloc[s], sources['stdSV'].iloc[s], fmt='s',
                color='navy', markersize=6, elinewidth=2)
    ax.errorbar(sources['dateY1'].iloc[s], sources['meanY1'].iloc[s], sources['stdY1'].iloc[s], fmt='s',
                color='navy', markersize=6, elinewidth=2)
    ax.errorbar(sources['dateY2'].iloc[s], sources['meanY2'].iloc[s], sources['stdY2'].iloc[s], fmt='s',
                color='navy', markersize=6, elinewidth=2)
    ax.errorbar(sources['dateY3'].iloc[s], sources['meanY3'].iloc[s], sources['stdY3'].iloc[s], fmt='s',
                color='navy', markersize=6, elinewidth=2)
    ax.errorbar(sources['dateY4'].iloc[s], sources['meanY4'].iloc[s], sources['stdY4'].iloc[s], fmt='s',
                color='navy', markersize=6, elinewidth=2)
    ax.errorbar(sources['dateY5'].iloc[s], sources['meanY5'].iloc[s], sources['stdY5'].iloc[s], fmt='s',
                color='navy', markersize=6, elinewidth=2)
    ax.axhline(y=sources['mean'].iloc[s], color='firebrick')

    t1 = ax.text(minX+150, maxY+0.15*height, str(sources['dtot'].iloc[s]) + ' jumps', fontdict={'color': 'black', 'size': 16})
    t1.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='white'))

    t2 = ax.text(minX+150, maxY+0.1*height, str(sources['ctot'].iloc[s]) + ' turning points',
                 fontdict={'color': 'black', 'size': 16})
    t2.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='white'))

    plt.savefig(AGN + "_lightcurve.png")
    plt.close(fig)

sources.to_csv("featureStats.txt", index=False, sep='\t')

