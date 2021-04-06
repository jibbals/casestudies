# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 16:51:15 2021
    Grey area where fire already burnt, 
    heat flux and 10m winds overlaid
@author: jgreensl
"""

#import matplotlib
#matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
from datetime import datetime, timedelta
from utilities import fio, utils, plotting

def plot_fire_spread(DA_sh, DA_ff, DA_u, DA_v):
    """
    """
    if hasattr(DA_u,"longitude"):
        lats=DA_u.latitude
        lons=DA_u.longitude
    elif hasattr(DA_u,"lon"):
        lats=DA_u.lat.values
        lons=DA_u.lon.values
    else:
        print(DA_u)
        print("ERROR: COULDN'T FIND L(ong/at)ITUDE")
        
    if len(lats) == DA_u.shape[1]:
        u = DA_u.values.T
        v = DA_v.values.T
        ff= DA_ff.values.T
        sh= DA_sh.values.T
    else:
        u = DA_u.values
        v = DA_v.values
        ff= DA_ff.values
        sh= DA_sh.values
    #ws = utils.wind_speed(u,v)
    print(np.min(ff),np.max(ff))
    
    # mask burnt area
    masked = np.ma.masked_where(ff < -.1, sh)
    
    #plt.imshow(lons,lats,masked, interpolation='none')
    # heat flux
    if np.max(sh)>100:
        plotting.map_sensibleheat(sh,lats,lons,colorbar=False)
    
    # quiver
    xskip=int(np.max([len(lons)//30-1,1]))
    yskip=int(np.max([len(lats)//30-1,1]))
    plt.quiver(lons[::xskip], lats[::yskip], u[::yskip,::xskip],v[::yskip,::xskip],
               pivot='mid',
               scale_units="inches",
               scale=50,)
    
    

def fire_spread(mr, zoom=None, subdir=None, coastline=-5):
    """
    ARGS:
        mr: model run name
        zoom: [W,E,S,N] in degrees
        subdir: name to save zoom into
        coastline: set to positive number to add coastline contour in metres
    """
    # Read firefront, heatflux (W/m2), U and V winds    
    DS_fire = fio.read_model_run_fire(mr)
    DA_topog = fio.model_run_topography(mr)
    if zoom is not None:
        DS_fire = fio.extract_extent(DS_fire,zoom)
        topog = fio.extract_extent(topog,zoom)
        if subdir is None:
            subdir=str(zoom)
    #print(DS_fire)
    #print(topog)
    lats=DS_fire.lat.data
    lons=DS_fire.lon.data
    times=DS_fire.time.data
    
    houroffset=utils.local_time_offset_from_lats_lons(lats,lons)
    # loop over timesteps
    for ti,time_utc in enumerate(times):
        
        ## slice time
        DS_fire_slice = DS_fire.sel(time=time_utc)
        DA_u10 = DS_fire_slice['UWIND_2']
        DA_v10 = DS_fire_slice['VWIND_2']
        DA_ff  = DS_fire_slice['firefront']
        DA_sh  = DS_fire_slice['SHEAT_2']
        
        ## get local time
        time_lt = utils.local_time_from_time_lats_lons(time_utc,lats,lons)
        time_str=time_lt.strftime("%Y%m%d %H%M")+"(UTC+%.2f)"%houroffset
                        
        ## FIRST FIGURE: 10m WIND DIR:
        fig=plt.figure(figsize=[11,11])
        
        plot_fire_spread(DA_sh, DA_ff, DA_u10, DA_v10)
        
        plt.title(time_str)
        
        if coastline>0:
            plt.contour(lons,lats,DA_topog.values, np.array([coastline]),
                        colors='k')
        # save figure
        fio.save_fig(mr,"fire_spread", time_utc, plt, subdir=subdir)


fire_spread("KI_run1_exploratory",coastline=5)