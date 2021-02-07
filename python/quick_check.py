#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 08:45:14 2019
    Weather summary looking at winds and clouds
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
import cartopy.crs as ccrs
from datetime import datetime, timedelta

from utilities import utils, plotting, fio, constants

## GLOBAL
#Script name
_sn_ = 'quick_check'

## RUN
# set up stuff, pull fpath from fio if possible
mr='KI_run1'
hours = fio.run_info[mr]['filedates']
hour = hours[12]
hourstr = hour.strftime("%Y%m%d %H:%M (UTC)")
figpath = "../figures/quick_check/"+mr+"/"

## First: fire contours

# read fire model output
ffront_cubelist = fio.read_fire(mr)
ffront = ffront_cubelist[0]
lats,lons = ffront.coord('latitude').points, ffront.coord('longitude').points

# slice time: take one in 60 (hourly)
ffront = ffront[::60] 
#print(ffront) # can check what data looks like
nt,ny,nx = ffront.shape

# make a plot with platecarree projection (so we can add coastlines)
ax = plt.axes(projection=ccrs.PlateCarree())

# add firefront contours
for ti in range(nt):
    ffti = ffront[ti].data
    # check there is something to plot
    if np.any(ffti < 0):
        plt.contour(lons, lats, ffti, np.array([0]),
                    transform=ccrs.PlateCarree(),
                    colors='red')

# add coastline
ax.coastlines()

# save figure
figname=figpath+"fire spread"
fio.save_fig_to_path(figname,plt)
#fio.make_folder(figname) # make sure folder exists
#plt.savefig(figname)

