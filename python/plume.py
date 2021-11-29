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
from matplotlib import colors, ticker, cm
from datetime import datetime, timedelta
from utilities import fio, utils, plotting

def plot_plume(DA_u, DA_v, DA_w, 
                     DA_sh=None, 
                     DA_ff=None,
                     thresh_windspeed=2,
                     hwind_limits=[0,26],
                     add_quiver_key=False):
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
                                  colorbar=False,
                                  zorder=3,
                                  )
    
    # fire front
    if DA_ff is not None:
        ff=DA_ff.values.T
        plotting.map_fire(ff,
                          DA_ff.lat.values,
                          DA_ff.lon.values,
                          alpha=0.6,
                          zorder=3,
                          )
    
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
            zorder=1, # under fire stuff, other stuff 
            )
    
    
    # zwinds in contour pygs
    zwind_norm=colors.SymLogNorm(0.25,base=2.) # linear to +- 0.25, then log scale
    zwind_cmap="PiYG_r"
    zwind_min,zwind_max = -1,5 # 2**0 up to 2**5
    zwind_contours=np.union1d( #np.union1d(
                    2.0**np.arange(zwind_min,zwind_max+1),
                    -1*(2.0**np.arange(zwind_min,zwind_max+1))
                    ) #, np.array([0]))
    zwind_cs = plt.contour(
            lons, lats, w, 
            levels=zwind_contours, 
            cmap=zwind_cmap, 
            norm=zwind_norm,
            zorder=4, # above all but quivers
            alpha=0.85,
            )
    # make nicer scalarmappable object for colorbars:
    zwind_scalarmappable=cm.ScalarMappable(norm=zwind_norm, cmap=zwind_cmap)
    # quiver/barbs
    plotting.quiverwinds(
            lats,lons,u,v, 
            thresh_windspeed=thresh_windspeed,
            n_arrows=12,
            alpha=0.5,
            zorder=5, # above all
            add_quiver_key=add_quiver_key,
            )
    
    return hwind_cs, zwind_scalarmappable
    

def plume(mr, extent=None, subdir=None, levels=[3,10,20,30,40, 50,60,70,90], coastline=2):
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
            time_str=time_lt.strftime("%dT%H:%M")+"(UTC+%.2f)"%houroffset
                            
            ## FIRST FIGURE: 10m WIND DIR:
            fig = plt.figure(
                figsize=[11,11],
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
                cs_h,cs_w = plot_plume(DA_u[ti,li],DA_v[ti,li],DA_w[ti,li], 
                        DA_sh=in_sh, 
                        DA_ff=in_ff, 
                        add_quiver_key=(li==0),
                        )
                # add model level height average to each subplot
                plt.text(0.01,0.01, # bottom left using axes coords
                        "%.2f m"%DS_atmos.level_height[level].values,
                        fontsize=9,
                        transform=plt.gca().transAxes,# bottom left using axes coords
                        zorder=10,
                        )
                plotting.map_add_locations_extent(extent,hide_text=True)
                if coastline>0 and np.min(DA_topog.values)<coastline:
                    plt.contour(lons,lats,DA_topog.values, np.array([coastline]),
                                colors='k')
                #plt.gca().set_aspect("equal")
                plt.gca().set(xticks=[],yticks=[],aspect="equal")
    
            plt.suptitle(mr+" "+time_str)

            plt.subplots_adjust(
                    wspace = 0.0,  # the amount of width reserved for space between subplots,
                    hspace = 0.0,
                    )
            #plt.tight_layout()
            
            # add space in specific area, then add vert wind colourbar
            cbar_ax = fig.add_axes([0.05, 0.975, 0.25, 0.015]) # X Y Width Height
            fig.colorbar(cs_h, 
                    cax=cbar_ax, 
                    format=ticker.ScalarFormatter(), 
                    pad=0,
                    orientation='horizontal',
                    )
            # Add horizontal wind colourbar (if uniform)
            cbar_ax2 = fig.add_axes([0.7, 0.975, 0.25, 0.015]) #XYWH
            cb_w = fig.colorbar(cs_w, 
                    cax=cbar_ax2, 
                    format=ticker.ScalarFormatter(),
                    pad=0,
                    orientation='horizontal',
                    )
            cb_w_ticks = [-32,-8,-2,0,2,8,32]
            cb_w.set_ticks(cb_w_ticks)
            cb_w.set_ticklabels([str(val) for val in cb_w_ticks])
    
            # save figure
            fio.save_fig(mr,"plume", time_utc, plt, subdir=subdir)

def suitecall(mr, extent=None,subdir=None):
    plume(mr, extent=extent, subdir=subdir)

if __name__ == '__main__':
    # keep track of used zooms
    KI_zoom_name = "zoom1"
    KI_zoom = constants.extents['KI'][KI_zoom_name]
    badja_zoom_name="zoom1"
    badja_zoom=constants.extents['badja'][badja_zoom_name]
    
    mr='badja_run3'
    zoom=badja_zoom
    zoom_name=badja_zoom_name

    plume(mr,zoom,zoom_name)
