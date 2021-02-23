#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Quick code I can figure out before saving into an appropriate script
@author: jesse
"""
import matplotlib
matplotlib.use("Agg")

from matplotlib import colors, ticker, patches
from matplotlib import patheffects, gridspec
import matplotlib.dates as mdates

import numpy as np
import matplotlib.pyplot as plt
import iris # minimal fio done here 
import iris.quickplot as qplt
import cartopy.crs as ccrs
from datetime import datetime, timedelta

from utilities import utils, plotting, fio, constants

## RUN

####### Figure: fire contours #######
##################################### 
ffront_filename = '../data/exploratory_KI_run1/fire/firefront.20200102T1500Z.nc'
figpath= '../figures/sample/firefront_KI_run1_exploratory.png'

ffront_cubes=fio.read_nc_iris(ffront_filename)
ffront = ffront_cubes[0][::10] # only look at 1 in 10 timesteps
lats,lons = ffront.coord('latitude').points, ffront.coord('longitude').points

#print(ffront) # can check what data looks like
nt,nx,ny = ffront.shape # fire output in [T,lons,lats]

# make a plot with platecarree projection (so we can add coastlines)
ax = plt.axes(projection=ccrs.PlateCarree())

# add firefront contours
for ti in range(nt):
    ffti = ffront[ti].data.T # contour uses X,Y,DATA[Y,X]
    # check there is something to plot
    if np.any(ffti < 0):
        # show zero line (our firefront)
        plt.contour(lons, lats, ffti, np.array([0]),
                    transform=ccrs.PlateCarree(),
                    colors='red')

# add coastline
ax.coastlines()

# save figure
fio.save_fig_to_path(figpath,plt)

