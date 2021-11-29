# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 21:11:21 2021
    Topdown view of wind direction, colourbar is ring of colours, 

    Run on badja Wandella subset:
        CPU Time Used: 00:18:12 
        Memory Used: 26.6GB 
        Walltime Used: 00:18:30 
@author: jgreensl
"""

import matplotlib
#matplotlib.use('Agg')

# plotting stuff
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, LinearLocator#, LogFormatter
#import matplotlib.patheffects as PathEffects
import numpy as np
import warnings

#from datetime import datetime,timedelta
from pandas import Timedelta, DatetimeIndex
from datetime import datetime
#from scipy import interpolate
#import cartopy.crs as ccrs

# local modules
from utilities import plotting, utils, constants, fio

###
## GLOBALS
###

##
## METHODS
###
def plot_wind_dir(DA_u, DA_v,
                      addring=False,
                      ring_XYwh=[.7,.88,.1,.1]):
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
    else:
        u = DA_u.values
        v = DA_v.values
    
    WDir=utils.wind_dir_from_uv(u,v)

    color_args=plotting.wind_dir_color_ring()
    cmap = color_args['cmap']
    norm = color_args['norm']
    
    plt.pcolormesh(
        lons,
        lats,
        WDir, 
        #levels=dircolorbounds,
        cmap=cmap,
        norm=norm,
        shading="nearest", # since not passing lon,lat corners
        #vmin=0,vmax=360,
        #extend='max',
        )
    ax = plt.gca()
    rax=None
    if addring:
        _, rax = plotting.add_wind_dir_color_ring(
                plt.gcf(),
                color_args,
                XYwh=ring_XYwh,
                deglable=False,
                )
    return ax,rax

def wind_dir_10m(
        mr,
        extent=None,
        subdir=None,
        coastline=2,
        ):
    """
    
    ARGUMENTS:
        mr: model run name
        extent: subset extent
        subdir: savefolder in case of specific extent
    """
    
    ## topography: maybe we want coastline
    topog=fio.model_run_topography(mr)
    
    # read fire model output
    DS_fire=fio.read_model_run_fire(mr)
    if extent is not None:
        DS_fire = fio.extract_extent(DS_fire,extent)
        topog = fio.extract_extent(topog,extent)
    lats=DS_fire.lat.values
    lons=DS_fire.lon.values
    coastflag = np.min(topog.values) < coastline
    
    times = DS_fire.time[::10].values # every 10 minute
    
    houroffset=utils.local_time_offset_from_lats_lons(lats,lons)
    
    # loop over timesteps
    for ti,time_utc in enumerate(times):
        
        ## get local time
        time_lt = utils.local_time_from_time_lats_lons(time_utc,lats,lons)
        time_str=time_lt.strftime("%Y%m%d %H%M")+"(UTC+%.2f)"%houroffset
                    
        ## FIRST FIGURE: 10m WIND DIR:
        plt.figure()
        
        DS_fire_slice = DS_fire.sel(time=time_utc)#, method="nearest")
        #print(DS_fire_slice) 
        DA_u10 = DS_fire_slice['UWIND_2']
        DA_v10 = DS_fire_slice['VWIND_2']
        DA_ff  = DS_fire_slice['firefront']
        ax,ringax = plot_wind_dir(DA_u10,DA_v10,
                                      addring=True,
                                      ring_XYwh=[.7,.81,.1,.1],
                                      )
        plt.sca(ax)
        plt.title(time_str + "10m wind direction")
        # add fire line            
        plotting.map_fire(DA_ff.values,lats,lons)
        # add topog
        if coastflag:
            plt.contour(lons,lats,topog.values,np.array([coastline]),colors='k')
            
        ax.set_aspect("equal")
        
        # save figure
        fio.save_fig(mr,"topdown_wdir_10m", time_utc, plt, subdir=subdir)



# def wind_dir(
#         mr,
#         extent=None,
#         subdir=None,
#         hours=range(24),
#         coastline=2,
#         ):
#     """
    
#     ARGUMENTS:
#         mr: model run name
#         hours: optional run for subset of model hours
#         extent: subset extent
#         subdir: savefolder in case of specific extent
#     """
#     # Defaults
#     if hours is None:
#         hours=range(24)
    
#     ## topography: maybe we want coastline
#     topog=fio.model_run_topography(mr)
#     coastflag = np.min(topog.values) < coastline
    
#     # read fire model output
#     DS_fire=fio.read_model_run_fire(mr)
#     lats=DS_fire.lat.values
#     lons=DS_fire.lon.values

#     for hour in hours:
#         DS=fio.read_model_run_hour(mr,hour=hour)
#         if extent is not None:
#             DS = fio.extract_extent(DS,extent)
        
#         houroffset=utils.local_time_offset_from_lats_lons(lats,lons)
         
#         times=DS.time.data # np.datetime64 array
        
#         # loop over timesteps
#         for ti,time_utc in enumerate(times):
            
#             ## get local time
#             time_lt = utils.local_time_from_time_lats_lons(time_utc,lats,lons)
#             time_str=time_lt.strftime("%Y%m%d %H%M")+"(UTC+%.2f)"%houroffset
                        
#             ## FIRST FIGURE: 10m WIND DIR:
#             fig=plt.figure()
            
#             DS_fire_slice = DS_fire.sel(time=time_utc)#, method="nearest")
#             #print(DS_fire_slice) 
#             DA_u10 = DS_fire_slice['UWIND_2']
#             DA_v10 = DS_fire_slice['VWIND_2']
#             DA_ff  = DS_fire_slice['firefront']
#             ax,ringax = plot_wind_dir(DA_u10,DA_v10,
#                                           addring=True,
#                                           ring_XYwh=[.7,.81,.1,.1],
#                                           )
#             plt.sca(ax)
#             plt.title(time_str + "10m wind direction")
#             # add fire line            
#             plotting.map_fire(DA_ff.values,lats,lons)
#             # add topog
#             if coastflag:
#                 plt.contour(lons,lats,topog.values,np.array([coastline]),colors='k')
                
#             ax.set_aspect("equal")
            
#             # save figure
#             fio.save_fig(mr,"topdown_wdir_10m", time_utc, plt, subdir=subdir)
            
    
def suitecall(mr, extent=None, subdir=None):
    wind_dir_10m(mr, extent=extent, subdir=subdir)

if __name__ == '__main__':

    # keep track of used zooms
    KI_zoom = [136.5,137.5,-36.1,-35.6]
    KI_zoom_name = "zoom1"
    KI_zoom2 = [136.5887,136.9122,-36.047,-35.7371]
    KI_zoom2_name = "zoom2"
    badja_zoom=[149.4,150.0, -36.4, -35.99]
    badja_zoom_name="zoom1"


    mr="badja_run3"
    zoom=badja_zoom
    zoom_name=badja_zoom_name
    wind_dir_10m(mr,zoom,zoom_name)
