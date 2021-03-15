#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Quick code I can figure out before saving into an appropriate script
@author: jesse
"""
import matplotlib
#matplotlib.use("Agg")

from matplotlib import colors, ticker, patches
from matplotlib import patheffects, gridspec
import matplotlib.dates as mdates

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from datetime import datetime, timedelta
from glob import glob
import os

#from utilities import utils, plotting, fio, constants

## RUN

####### Figure: fire contours #######
##################################### 
filepattern= '../data/KI_run1_exploratory/atmos/*mdl_th1.nc'
filepaths = glob(filepattern)
filepaths.sort()
print(filepaths)
# can set extent to look at (for zooming in or out of something)
extent = None
#extent=[135, 138, -38, -34]

# where will plots go
def makefolder(folder):
    if not os.path.exists(folder):
        print("INFO: Creating folder:",folder)
        os.makedirs(folder)

figdir="airpressures"
makefolder(figdir)

# make air pressure for one time slice per hour:
for fpath in filepaths:
    ds=xr.open_dataset(fpath)
    #print(ds)
    P=ds['pressure'] # [T,lev,lat,lon] in Pascals
    
    # pull out lats and lons for easy plotting
    lats=P.latitude.data
    lons=P.longitude.data
    times=P.time.data
    
    # datetime to OK string
    tstr = str(times[0])[:18]
    
    #print(P) # can check what data looks like
    # make a plot with platecarree projection (so we can add coastlines)
    fig = plt.figure(figsize=(15,11))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    
    # set the extent we will look at:
    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    # just pulling one time step per file, and only looking at surface level
    Surf_Press_hPa = P[0,0].values/100. # hectapascals
    Pmin=965
    Pmax=1015
    Levels=np.arange(Pmin,Pmax+.1, 2.5)
    plt.contourf(lons, lats, Surf_Press_hPa,
                 Levels,
                 cmap='Purples',
                 extend="both",
                 vmin=Pmin,
                 vmax=Pmax,
                 )
    plt.colorbar(label="hPa")
    plt.title("Surface Air pressure "+tstr+" (UTC)")
    ## Should probably add firefront contour for this sort of plot
    ## that would go here
    
    # add coastline
    ax.coastlines()

    # save figure
    figpath=figdir + "/" + tstr
    print("INFO: saving figure:",figpath)
    plt.savefig(figpath)
    plt.close(fig)
