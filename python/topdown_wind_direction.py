# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 21:11:21 2021

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
from pandas import Timedelta
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
def topdown_wind_plot(DA_u, DA_v,
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
        vmin=0,vmax=360,
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

def topdown_winds(
        mr,
        hours=range(24),
        extent=None,
        levels=[0,1,3,5,10],
        subdir=None,
        ):
    """
    
    ARGUMENTS:
        mr: model run name
        hours: optional run for subset of model hours
        extent: subset extent
        levels: model levels to plot
        subdir: savefolder in case of specific extent
    """
    # Defaults
    if hours is None:
        hours=range(24)
    
    ## topography: maybe we want coastline
    topog=fio.model_run_topography(mr)
    coastline = 5.0
    coastflag = np.min(topog.values) < coastline
    
    # read fire model output
    DS_fire=fio.read_model_run_fire(mr)
    lats=DS_fire.lat.values
    lons=DS_fire.lon.values
    
    for hour in hours:
        DS=fio.read_model_run_hour(mr,hour=hour)
        if extent is not None:
            DS = fio.extract_extent(DS,extent)
        
        houroffset=utils.local_time_offset_from_lats_lons(lats,lons)
         
        times=DS.time.data # np.datetime64 array
        
        # loop over timesteps
        for ti,time_utc in enumerate(times):
            
            ## get local time
            time_lt = utils.local_time_from_time_lats_lons(time_utc,lats,lons)
            time_str=time_lt.strftime("%Y%m%d %H%M")+"(UTC+%.2f)"%houroffset
                        
            ## FIRST FIGURE: 10m WIND DIR:
            fig=plt.figure(figsize=[11,11])
            
            DS_fire_slice = DS_fire.sel(time=time_utc)
            #print(DS_fire_slice) 
            DA_u10 = DS_fire_slice['UWIND_2']
            DA_v10 = DS_fire_slice['VWIND_2']
            DA_ff  = DS_fire_slice['firefront']
            ax,ringax = topdown_wind_plot(DA_u10,DA_v10,addring=True)
            plt.sca(ax)
            plt.title(time_str + "10m wind direction")
            # add fire line            
            plotting.map_fire(DA_ff.values,lats,lons)
            # add topog
            if coastflag:
                plt.contour(lons,lats,topog.values,np.array([coastline]),colors='k')
                
            # save figure
            fio.save_fig(mr,"topdown_wdir_10m", time_utc, plt, subdir=subdir)
            
            fig = plt.figure(figsize=[11,15])
            DS_timeslice=DS.loc[dict(time=time_utc)]
            #print(DS_timeslice)
            DA_x = DS_timeslice['wnd_ucmp']
            DA_y = DS_timeslice['wnd_vcmp']
            # destagger x and y winds
            DA_u,DA_v = utils.destagger_winds_DA(DA_x,DA_y)
            for il,level in enumerate(levels):
                topdown_wind_plot(DA_u,DA_v,addring=(il==0))
                plt.title("")
                plt.xlabel("height")
                
            print(DA_u)
            # title and saved
            plt.suptitle(time_str+ "model level winds")
            fio.save_fig(mr,"topdown_wind_dirs",time_utc,plt,)
            assert False, "Stop here for now"
            
    

if __name__ == '__main__':
    mr="KI_run1_exploratory"
    topdown_winds(mr)