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
import iris.quickplot as qplt
import cartopy.crs as ccrs
from datetime import datetime, timedelta

from utilities import utils, plotting, fio, constants

## GLOBAL
#Script name
_sn_ = 'quick_check'

## RUN
# set up stuff, pull fpath from fio if possible
mr='badja_run1'
#backdrop="../data/QGIS/KI.tiff"
backdrop = "badja.tiff" # I will use my helper function

# based on run name
hours = fio.run_info[mr]['filedates']
hour = hours[12]
hourstr = hour.strftime("%Y%m%d %H:%M (UTC)")
figpath = "../figures/quick_check/"+mr+"/"
#optionally can add extent to zoom in on
# extent=[west,east,south,north]

# Lets get file pointers to all the files we will read here:
# these are only read when .data is called or some other operations are performed on the cubes

# read fire model output (fire front is default)
ffront_cubelist = fio.read_fire(mr)
ffront = ffront_cubelist[0]
# read fire model output fire flux
fflux_cubelist = fio.read_fire(mr, filename='sensible_heat')
fflux = fflux_cubelist[0]
# read model output (th1 and ro1 files only)
th1_files = fio.model_run_filepaths(mr, hours=hour, suffix="th1") # has vertical_wnd, pressure, air_temp
ro1_files = fio.model_run_filepaths(mr, hours=hour, suffix="ro1") # has x_wind,y_wind
th1_cubes = iris.load(th1_files[0])
ro1_cubes = iris.load(ro1_files[0])

# 

####### Figure: horizontal winds #######
########################################

x_wind0 = ro1_cubes.extract("x_wind")[0] # extract returns list of cubes
y_wind0 = ro1_cubes.extract("y_wind")[0]
## subset to first time, first level:
x_wind1 = x_wind0[0,0]
y_wind1 = y_wind0[0,0]
## these fields are differently staggered: need to fix before combining to plot
x_wind,y_wind = utils.destagger_wind_cubes([x_wind1,y_wind1])

lats,lons = x_wind.coord('latitude').points, x_wind.coord('longitude').points

# magnitude from x and y

speed = iris.analysis.maths.apply_ufunc(np.hypot,x_wind,y_wind)
#speed= (x_wind**2 + y_wind**2)**0.5

# set up plot projection to use lats and lons
plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
# get the coord reference system
transform = x_wind.coord('latitude').coord_system.as_cartopy_projection()
# Plot the wind speed as a contour plot
qplt.contourf(speed, 20)

# Add arrows to show the wind vectors
# only show 1 in N arrow
N=10
plt.quiver(lons[::N], lats[::N], 
           x_wind[::N,::N].data, y_wind[::N,::N].data, 
           pivot='middle', transform=transform)

# add coastline
ax.coastlines()

# title and save
plt.title("Wind speed")
figname=figpath+"windspeed"
fio.save_fig_to_path(figname,plt)

####### Figure: VERTICAL winds #######
########################################

# get the data
z_wind0 = th1_cubes.extract("upward_air_velocity")[0]

# set up bidirectional logarithmic colorscale for vert motion
norm = colors.SymLogNorm(0.25,base=2.0) # linear up to +- .25
contours = np.array([-16,-8,-4,-2,-1,0,1,2,4,8,16]) # setting colour levels for plot
#formatter = ticker.ScalarFormatter() # scalar ticks on cbar

plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
qplt.contourf(z_wind0[0,10], contours, norm=norm, 
            #format=formatter,
            )

plt.title("vert winds at model level 10")
figname=figpath+"vertwinds"
fio.save_fig_to_path(figname,plt)


####### Figure: fire contours #######
##################################### 

lats,lons = ffront.coord('latitude').points, ffront.coord('longitude').points

# slice time: take one in N minutes
ffront = ffront[::120] 
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

########  FIGURE: NICER FIRE flux lines ######
############################################

# lats and lons will match ffront
#slice time:
fflux = fflux[::120]
nt,ny,nx = fflux.shape

# helper function to set up tiff image backdrop
fig,ax = plotting.map_tiff_qgis(fname=backdrop,)

# set up colorbar scaling (norm and levels)
index_min,index_max=2,5
fflux_min,fflux_max=10**index_min,10**index_max
fflux_norm = colors.LogNorm(vmin=fflux_min,vmax=fflux_max)
fflux_levels = np.logspace(index_min,index_max,20)
for ti in range(nt):
    ffluxti = fflux[ti].data
    if np.max(ffluxti) > fflux_min:
        # add contourfs showing heat flux
        cs=plt.contourf(lons,lats,ffluxti+1, fflux_levels,
                cmap='hot',norm=fflux_norm, 
                vmin=fflux_min, vmax=fflux_max,
                alpha=0.6,
                extend='max',
                )
        # NOTES: extend is required in this case or else values under vmin hide many of the contourfs
        #      : I added 1 to the ffluxti so that we don't need to worry about the 0 value in log scaling
        #      : The KI.tiff is not as large as the domain, so there is white space around it in the figure

# add colorbar
cax=fig.add_axes([.91,.2,.02,.6],frameon=False) # [X,Y,Width,Height]
plt.colorbar(cs, cax=cax,label='sensible heat flux',
        #ticks = fflux_levels[::5],
        ticks=[1e2,1e3,1e4,1e5],
        orientation='vertical',
        )

figname=figpath+"fire heat"
fio.save_fig_to_path(figname,plt)

