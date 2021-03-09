#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 10:22:08 2021
    wrf-python vertical cross section usage
@author: jesse
"""

import matplotlib
#matplotlib.use("Agg")

#from matplotlib import colors, ticker, patches

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib import colors, ticker
import Ngl # curly vector plotting
import cartopy.crs as ccrs

from datetime import datetime, timedelta

import xarray as xr # for FIO
#from utilities import utils, plotting, fio, constants
from utilities import utils,plotting

from glob import glob

# from wrf import (vertcross, CoordPair, to_np,
#                  get_cartopy, latlon_coords, cartopy_xlim, cartopy_ylim, 
#                  ll_points, destagger)

#### GLOBAL #####
__cloud_thresh__ = .01 #constants.cloud_threshold
proj = ccrs.PlateCarree()
# set up bidirectional logarithmic colorscale for vert motion
vert_wind_norm = colors.SymLogNorm(0.25,base=2.0) # linear up to +- .25
vert_wind_contours = np.array([-16,-8,-4,-2,-1,0,1,2,4,8,16]) # setting colour levels for plot
vert_wind_cmap = "PiYG_r"
h_wind_contours = np.arange(30)
h_wind_cmap = "Blues"
add_windstream=False
add_quiver=True

##### RUN #####
## netcdf datasets
mdl_th1_paths=glob("../data/exploratory_KI_run1/atmos/*mdl_th1.nc")
mdl_ro1_paths=glob("../data/exploratory_KI_run1/atmos/*mdl_ro1.nc")
#slv_paths=glob("../data/exploratory_KI_run1/atmos/*slv.nc")

## Pull out just one file for now
dt_index=8
ds_th1 = xr.load_dataset(mdl_th1_paths[dt_index])
ds_ro1 = xr.load_dataset(mdl_ro1_paths[dt_index])
#ds_slv = xr.load_dataset(slv_paths[dt_index])

## grab topography and level heights DataArrays
da_topog = ds_th1['surface_altitude']
da_z = ds_th1['level_height']

lats=da_topog.latitude.values
lons=da_topog.longitude.values
## grab vertical motion
da_w = ds_th1['vertical_wnd']
## grab pressure (Pa)
da_p = ds_th1['pressure']
## grab x and y winds
#print(ds_ro1)
da_x = ds_ro1['wnd_ucmp']
da_y = ds_ro1['wnd_vcmp']
#print(da_y)
lats1 = da_y.latitude_0.values
lons1 = da_x.longitude_0.values

### Plot vert cross of vertical wind motion...
## start and end points
lat0,lon0=-35.93,136.6
lat1,lon1=-35.76,136.9

## vertical levels to show vertcross upon
# using all for now, exploratory only has 19 levels

## loop over time here probably:
wi=da_w[0].values # pull out time slice
pi=da_p[0].values
xi=da_x[0].values
yi=da_y[0].values

## Show island and transect line
ax1=plt.subplot(3,1,1,projection=proj)
plt.contourf(lons,lats,da_topog.values,
             60, # number of color levels
             cmap="terrain")
plt.plot([lon0,lon1], 
         [lat0,lat1], 
         'ro--', 
         transform=proj,
         linewidth=2, 
         markersize=12)
plt.title("topography (m)")
plt.colorbar(pad=0) 
# aligned by default on east edge of figure: move to centre
ax1.set_anchor('C')

ax2=plt.subplot(3,1,2)
## Cross section for w

### transect of vertical motion
cross1 = utils.transect(wi,lats,lons,
                  start=[lat0,lon0],
                  end=[lat1,lon1],
                  z_th=pi)
cross_w = cross1['transect']
xaxis=cross1['x']
yaxis=cross1['y']
label=cross1['label']

## Make the contour plot for wind speed
#contours_w = plt.contourf(xaxis,yaxis,cross_w, cmap=get_cmap("jet"))
contours_w = plt.contourf(xaxis,yaxis,cross_w, 
                          vert_wind_contours/4.0,
                          #shading="auto",
                          cmap=vert_wind_cmap,
                          norm=vert_wind_norm,
                          extend="both",
                          )

ax2.invert_yaxis()
ax2.set_yscale('log')
plt.xticks([],[])
plt.title('vertical motion (m/s)')
plt.ylabel("Pressure(Pa)")
plt.colorbar(contours_w, format=ticker.ScalarFormatter(), pad=0)

## Contout plot for wind speed
ax3=plt.subplot(3,1,3)

## create transects of x and y winds, combine and plot using combined vs vertical wind
## First we need to destagger x and y winds
ui,vi = utils.destagger_winds(xi,yi,lats=lats,lons=lons,lats1=lats1,lons1=lons1)

si = utils.wind_speed(ui,vi)
cross2=utils.transect(si,lats,lons,
                  start=[lat0,lon0],
                  end=[lat1,lon1],
                  z_th=pi)
cross_s = cross2['transect'] # horizontal wind magnitude
xaxis=cross2['x']
yaxis=cross2['y']
label=cross2['label']

## Make the contour plot for wind speed
#contours_w = plt.contourf(xaxis,yaxis,cross_w, cmap=get_cmap("jet"))
contours_s = plt.contourf(xaxis,yaxis,cross_s, 
                          h_wind_contours,
                          cmap=h_wind_cmap,
                          extend="max",
                          )

if add_quiver:
    ## Add wind stream
    cross_winds_str=utils.transect_winds(ui,vi,lats,lons,
                                     start=[lat0,lon0],
                                     end=[lat1,lon1],
                                     z=pi)
    cross_H=cross_winds_str['wind'] # horizontal winds along transect
    #print([np.shape(arr) for arr in [cross_H,cross_w,xaxis,yaxis]])
    plt.quiver(xaxis[::2,::2],yaxis[::2,::2],
                cross_H[::2,::2],cross_w[::2,::2], 
                color='k',
                pivot='mid',
                #linewidth=streamLW,
                #minlength=0.5,
                #density=(0.7, 0.7),
                )
    
if add_windstream:
    # requires increasing monotonic dimensions, so we flip pressure levels (y dim)
    plotting.streamplot_regridded(xaxis[::-1],yaxis[::-1],cross_H[::-1],cross_w[::-1],
                                  minlength=.5,
                                  density=(.7,.3))

ax3.invert_yaxis()
ax3.set_yscale('log')
plt.xticks([xaxis[0,0],xaxis[0,-1]],[label[0],label[-1]],rotation=15)
plt.title('horizontal wind speed (m/s)')
plt.ylabel("Pressure(Pa)")
plt.colorbar(contours_s,pad=0)


