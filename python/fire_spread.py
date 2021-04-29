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
from matplotlib import colors
from datetime import datetime, timedelta
from utilities import fio, utils, plotting

def isochrones(mr, extent=None, subdir=None):
    """
    """
    DS_fire = fio.read_model_run_fire(mr)
    DA_topog = fio.model_run_topography(mr)
    
    if extent is not None:
        DS_fire = fio.extract_extent(DS_fire,extent)
        print("DEBUG: Extracting topog:")
        print(DA_topog)
        print(extent)
        DA_topog = fio.extract_extent(DA_topog,extent)
        if subdir is None:
            subdir=str(extent)
    
    DA_FF = DS_fire['firefront']
    lats=DS_fire.lat.data
    lons=DS_fire.lon.data
    if extent is None:
        extent=[lons[0],lons[-1],lats[0],lats[-1]]
    
    times=DS_fire.time.data
    
    ## Plot starts here, we put isochrones onto topography
    plt.contourf(lons,lats,DA_topog,cmap='terrain')
    # loop over timesteps after fire starts
    hasfire=np.min(DA_FF.values,axis=(1,2)) < 0
    for ti,time_utc in enumerate(times[hasfire][::60]):
        
        ## slice time
        FF = DA_FF.sel(time=time_utc).data.T
        
        plotting.map_fire(FF,lats,lons)
        
    plotting.map_add_locations_extent(extent,hide_text=True)
    
    # title from local time at fire ignition
    time_lt = utils.local_time_from_time_lats_lons(times,lats,lons)
    title=np.array(time_lt)[hasfire][0].strftime("Hourly fire front from %H%M")
    plt.title(title)
        
    plt.gca().set_aspect("equal")

    # save figure
    fio.save_fig(mr,"isochrones", mr, plt, subdir=subdir)
    

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
    #print(np.min(ff),np.max(ff))
    # min is -0.0316...
    # max is .0084
    
    # burnt area
    if np.min(ff)<-0.02:
        #    contourfargs['norm']=col.LogNorm()
        burn_levels=[-.07,-.001,0]
        colors.Colormap
        ## Do the filled contour plot
        plt.contourf(lons, lats, ff,
                     burn_levels, # color levels
                     #vmax=-.01,
                     cmap='Greys_r',
                     #alpha=0.8,
                     )
        
    #plt.imshow(lons,lats,masked, interpolation='none')
    # heat flux
    if np.max(sh)>100:
        plotting.map_sensibleheat(sh,lats,lons,colorbar=False)
    
    # quiver
    xskip,yskip=plotting.xyskip_for_quiver(lats,lons)
    #print(xskip,yskip) # 24,24 in unzoomed run of corryong output
    plt.quiver(lons[::xskip], lats[::yskip], 
               u[::yskip,::xskip], v[::yskip,::xskip],
               pivot='mid',
               scale_units="inches",
               scale=50,)
    
    

def fire_spread(mr, extent=None, subdir=None, coastline=5):
    """
    ARGS:
        mr: model run name
        extent: [W,E,S,N] in degrees
        subdir: name to save zoomed into
        coastline: set to positive number to add coastline contour in metres
    """
    # Read firefront, heatflux (W/m2), U and V winds    
    DS_fire = fio.read_model_run_fire(mr)
    DA_topog = fio.model_run_topography(mr)
    if extent is not None:
        DS_fire = fio.extract_extent(DS_fire,extent)
        print("DEBUG: Extracting topog:")
        print(DA_topog)
        print(extent)
        DA_topog = fio.extract_extent(DA_topog,extent)
        if subdir is None:
            subdir=str(extent)
    
    lats=DS_fire.lat.data
    lons=DS_fire.lon.data
    if extent is None:
        extent=[lons[0],lons[-1],lats[0],lats[-1]]
    times=DS_fire.time.data
    
    houroffset=utils.local_time_offset_from_lats_lons(lats,lons)
    # loop over timesteps
    # interesting times don't begin before 3 hours in any run
    ind_interest = np.union1d([0,30,60,90,120,150,180],np.arange(181,24*60-1,5))
    times_of_interest=[times[i] for i in ind_interest]
    for ti,time_utc in enumerate(times_of_interest):
        
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
        plt.figure(
            #figsize=[14,11],
            )
        
        plot_fire_spread(DA_sh, DA_ff, DA_u10, DA_v10)
        plotting.map_add_locations_extent(extent,hide_text=False)
        plt.title(time_str)
        
        if coastline>0 and np.min(DA_topog.values)<coastline:
            plt.contour(lons,lats,DA_topog.values, np.array([coastline]),
                        colors='k')
        plt.gca().set_aspect("equal")

        # save figure
        fio.save_fig(mr,"fire_spread", time_utc, plt, subdir=subdir)


if __name__ == '__main__':
    # keep track of used zooms
    KI_zoom = [136.5,137.5,-36.1,-35.6]
    KI_zoom_name = "zoom1"
    KI_zoom2 = [136.5887,136.9122,-36.047,-35.7371]
    KI_zoom2_name = "zoom2"
    badja_zoom=[149.4,150.0, -36.4, -35.99]
    badja_zoom_name="zoom1"
    
    mr='KI_eve_run1'
    zoom=KI_zoom
    zoom_name=KI_zoom_name
    isochrones(mr, extent=zoom, subdir=zoom_name)
    #fire_spread(mr,zoom,zoom_name)
