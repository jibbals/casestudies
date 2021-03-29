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

####### Figure: wind direction #######
##################################### 


from utilities import fio, utils

import matplotlib.pyplot as plt
from matplotlib import colors

mr='KI_run1_exploratory'
alldatetimes=fio.hours_available(mr)
hour=alldatetimes[9]
hstr=hour.strftime("%Y-%m-%dT%H:%M")
#DS = fio.read_model_run_hour(mr,hour=9)
## Read topography
#topog = fio.model_run_topography(mr)
DS_fire = fio.read_model_run_fire(mr)
#print(DS_fire)
DS_fire = DS_fire.isel(time=9*20) # exploratory is every 3 minutes
print(DS_fire)
DA_u10 = DS_fire['UWIND_2']
DA_v10 = DS_fire['VWIND_2']
lats = DS_fire.lat.values
lons = DS_fire.lon.values
# transpose and get wind dir
WDir=utils.wind_dir_from_uv(DA_u10.values.T,DA_v10.values.T)


## colorbar manual test
#direction colors from 0-15,15-45,...,345-360
# 13 colors over 14 areas (so 15 bounds)
dircolorlist=['cyan','darkturquoise',
              'darkblue','mediumblue','blue',
              'limegreen','lime','green',
              'salmon','red','darkred',
              'aquamarine','cyan']
dircolorbounds=[0,15,45,75,105,135,165,195,225,255,285,315,345,360]
dirticks=[0,45,90,135,180,225,270,315]
straight_colorbar=False
ring_colorbar=True

cmap = colors.ListedColormap(dircolorlist)
norm=colors.BoundaryNorm(dircolorbounds, cmap.N)

# need to set levels to maintain consistency
fig = plt.figure(figsize=[11,11])
img = plt.pcolormesh(lons,lats,WDir, 
             #levels=dircolorbounds,
             cmap=cmap,
             norm=norm,
             vmin=0,vmax=360,
             #extend='max',
             )

#if straight_colorbar:
#if not ring_colorbar:
cb=plt.colorbar(img, 
             cmap=cmap, 
             norm=norm, 
             boundaries=dircolorbounds, 
             ticks=dircolorbounds,
             )
if ring_colorbar:
    ring_ax=fig.add_axes([.7,.15,.08,.08],projection="polar")
    # define colormap normalization for 0 to 2*pi
    ring_norm=colors.Normalize(0,2*np.pi)
    n=200 # secants for mesh
    #t = np.linspace(np.pi/2.0,-1.5*np.pi,n) # angles to show
    t = np.linspace(0,2*np.pi,n) # angles to show
    r = np.linspace(0.2,.9,2) # radius
    rg, tg = np.meshgrid(r,t)
    ring_img = ring_ax.pcolormesh(t,r,
                                  # colors range from 0 to 2pi, 
                                  # I convert to degrees and match with my own colorbar
                                  (-1*tg.T*180/np.pi+90)%360,
                                  #norm=ring_norm,
                                  cmap=cmap,
                                  norm=norm,
                                  )
    ring_ax.set_yticklabels([])
    # convert 0 to 360 math direction ticks to met direction ticks
    math_ticks=np.arange(0,360,30)
    met_ticks=np.array([90,60,30,0,330,300,270,240,210,180,150,120])
    ring_ax.set_xticks([]
            #math_ticks
            )
    ring_ax.set_xticklabels([]
            #met_ticks
            )