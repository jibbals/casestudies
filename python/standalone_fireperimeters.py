#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    ##To run on NCI:
    ## REQUIRES HH5 project access
    ## need to load python environment like so
    module use /g/data3/hh5/public/modules
    module load conda/analysis3
@author: jesse
"""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
from datetime import datetime, timedelta

## RUN

####### Figure: fire contours #######
##################################### 
# where is data output
ffront_filename = '../data/KI_run1_exploratory/fire/firefront.20200102T1500Z.nc'
# yours may be at /g/data/en0/<username>/ACCESS-fire/<run name>/<yyyymmdd of run>/timestamp of simulation/<name of nest>/fire/firefront....

# can set extent to look at (for zooming in or out of something)
extent = None
#extent=[135, 138, -38, -34]

# where will plot go
figpath= 'firefront.png'

# read file into dataset
ds=xr.open_dataset(ffront_filename)
print(ds)
ffront=ds['firefront']
# pull out lats and lons for easy plotting
lats=ffront.lat.data
lons=ffront.lon.data
times=ffront.time.data

print(ffront) # can check what data looks like
nt,nx,ny = ffront.shape # fire output shape is [T,lon,lat]

# make a plot with platecarree projection (so we can add coastlines)
ax = plt.axes(projection=ccrs.PlateCarree())

# set the extent we will look at:
if extent is not None:
    ax.set_extent(extent, crs=ccrs.PlateCarree())

# add firefront contours, once per N time steps
for ti in range(0,nt,30):

    # contour uses X,Y,DATA[Y,X] so we need the transpose
    ffti = ffront[ti].data.T
    
    # check there is something to plot
    if np.any(ffti < 0):
        # show zero line (our firefront)
        plt.contour(lons, lats, ffti, np.array([0]),
                    transform=ccrs.PlateCarree(),
                    colors='red')
plt.title("Fire fronts between" + str(times[0]) + " and " + str(times[30]))

# add coastline
ax.coastlines()

# save figure
plt.savefig(figpath)
print("Saved figure: ",figpath)
