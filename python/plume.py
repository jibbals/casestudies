# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 14:20:15 2021
    Plume discovery based on model levels of winds
    # run with 50% extent subset:
       CPU Time Used: 09:06:44
       Memory Used: 19.84GB
       Walltime Used: 09:08:31
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

def plot_plume(DA_u, DA_v, DA_w, 
                     DA_sh=None, 
                     DA_ff=None,
                     thresh_windspeed=2,
                     hwind_limits=[0,26]):
    """
    Plot showing Hwind contourf, Vert motion contour, firefront or sens heat optionally
    INPUTS:
        u, v, w, [sh, ff] xr.DataArrays with [latitude,longitude] dims
        
    """
    # u and v are from atmos files
    lats=DA_u.latitude
    lons=DA_u.longitude
    
    u = DA_u.values
    v = DA_v.values
    w = DA_w.values
    
    #print("DEBUG: DA_u, DA_v, DA_w:")
    #print(DA_u)
    #print(DA_v)
    #print(DA_w)
    
    # horizontal winds
    
    # heat flux
    if DA_sh is not None:
        sh=DA_sh.values.T
        plotting.map_sensibleheat(sh,
                                  DA_sh.lat.values,
                                  DA_sh.lon.values,
                                  colorbar=False)
    
    # fire front
    if DA_ff is not None:
        ff=DA_ff.values.T
        plotting.map_fire(ff,
                          DA_ff.lat.values,
                          DA_ff.lon.values,
                          alpha=0.6,)
    
    # hwinds in Blues
    hwind_vmin,hwind_vmax=hwind_limits
    hwind_n_contours=hwind_vmax-hwind_vmin+1
    hcontours=np.linspace(hwind_vmin,hwind_vmax,hwind_n_contours)
    hwind_norm=colors.Normalize(vmin=hwind_vmin,vmax=hwind_vmax, clip=True)
    hwind = np.hypot(u,v)
    hwind_cs = plt.contourf(
            lons, lats, hwind, 
            levels=hcontours,
            cmap="Blues", 
            norm=hwind_norm,
            )
    
    
    # zwinds in contour pygs
    zwind_norm=colors.SymLogNorm(0.25,base=2.) # linear to +- 0.25, then log scale
    zwind_min,zwind_max = -1,5 # 2**-1 up to 2**-5
    zwind_contours=np.union1d( #np.union1d(
                    2.0**np.arange(zwind_min,zwind_max+1),
                    -1*(2.0**np.arange(zwind_min,zwind_max+1))
                    ) #, np.array([0]))
    zwind_cs = plt.contour(
            lons, lats, w, 
            levels=zwind_contours, 
            cmap="PiYG_r", 
            norm=zwind_norm,
            )
    
    # quiver/barbs
    plotting.quiverwinds(
            lats,lons,u,v, 
            thresh_windspeed=thresh_windspeed,
            n_arrows=15,
            alpha=0.6,
            )
    
    return hwind_cs, zwind_cs
    

def plume(mr, extent=None, subdir=None, levels=[2,10,20,30,40, 50,60,70,90], coastline=2):
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
        DA_topog = fio.extract_extent(DA_topog,extent)
        if subdir is None:
            subdir=str(extent)
    
    lats=DS_fire.lat.data
    lons=DS_fire.lon.data
    if extent is None:
        extent=[lons[0],lons[-1],lats[0],lats[-1]]
        
    #times=DS_fire.time.data
    #localtimes = utils.local_time_from_time_lats_lons(times,lats,lons)
    #houroffset=utils.local_time_offset_from_lats_lons(lats,lons)
    
    # loop over timesteps
    hours=fio.hours_available(mr)
    for hi,hour_utc in enumerate(hours):
        
        ## slice time
        DS_atmos = fio.read_model_run_hour(mr,hour=hi)
        times_utc = DS_atmos.time.values
        
        DA_x = DS_atmos['wnd_ucmp']
        DA_y = DS_atmos['wnd_vcmp']
        # destagger x and y winds
        DA_u,DA_v = utils.destagger_winds_DA(DA_x,DA_y)
        DA_w = DS_atmos['vertical_wnd']
        if extent is not None:
            DA_u = fio.extract_extent(DA_u, extent)
            DA_v = fio.extract_extent(DA_v, extent)
            DA_w = fio.extract_extent(DA_w, extent)
        
        for ti, time_utc in enumerate(times_utc):
            DS_fire_slice = DS_fire.sel(time=time_utc)
            DA_ff  = DS_fire_slice['firefront']
            DA_sh  = DS_fire_slice['SHEAT_2']
            
            ## get local time
            time_lt = utils.local_time_from_time_lats_lons(time_utc,lats,lons)
            houroffset = utils.local_time_offset_from_lats_lons(lats,lons)
            time_str=time_lt.strftime("%Y%m%d %H%M")+"(UTC+%.2f)"%houroffset
                            
            ## FIRST FIGURE: 10m WIND DIR:
            fig = plt.figure(
                figsize=[12,12],
                )
            #fig_grid = fig.add_gridspec(3, 3, wspace=0, hspace=0)
            for li,level in enumerate(levels):
                plt.subplot(3,3,1+li, aspect="equal")
                if li == 0:
                    in_sh = DA_sh
                    in_ff = None
                else:
                    in_sh = None
                    in_ff = DA_ff
                plot_plume(DA_u[ti,li],DA_v[ti,li],DA_w[ti,li], DA_sh=in_sh, DA_ff=in_ff)
                # add model level height average to each subplot
                plt.text(0.01,0.01, # bottom left using axes coords
                        "%.2f m"%DS_atmos.level_height[level].values,
                        fontsize=9,
                        transform=plt.gca().transAxes,# bottom left using axes coords
                        )
                plotting.map_add_locations_extent(extent,hide_text=True)
                if coastline>0 and np.min(DA_topog.values)<coastline:
                    plt.contour(lons,lats,DA_topog.values, np.array([coastline]),
                                colors='k')
                #plt.gca().set_aspect("equal")
                plt.gca().set(xticks=[],yticks=[],aspect="equal")
    
            plt.suptitle(time_str)

            
            plt.tight_layout()
            plt.subplots_adjust(
                    wspace = 0.0,  # the amount of width reserved for space between subplots,
                    hspace = 0.0,
                    )
    
            # save figure
            fio.save_fig(mr,"plume", time_utc, plt, subdir=subdir)


if __name__ == '__main__':
    # keep track of used zooms
    KI_zoom = [136.5,137.5,-36.1,-35.6]
    KI_zoom_name = "zoom1"
    KI_zoom2 = [136.5887,136.9122,-36.047,-35.7371]
    KI_zoom2_name = "zoom2"
    badja_zoom=[149.4,150.0, -36.4, -35.99]
    badja_zoom_name="zoom1"
    
    mr='KI_run2'
    zoom=KI_zoom2
    zoom_name=KI_zoom2_name

    plume(mr,zoom,zoom_name)