# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 2021
    Top down view of rotation

@author: jgreensl
"""

import matplotlib
#matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

#from matplotlib import colors
#from datetime import datetime, timedelta
from utilities import fio, utils, plotting, constants

_SN_ = "rotation"
    
def plot_rotation(
        mr, 
        extent=None, 
        subdir=None, 
        hours_range=None,
        use_agl=False,
        altitude=2000,
        ):
    """
    model run rotation top down plots
    ARGS:
        model run name, ...
        altitude: model level to examine
    """
    DS_fire = fio.read_model_run_fire(mr,extent=extent)
    lats=DS_fire.lat.data
    lons=DS_fire.lon.data
    #times=DS_fire.time.data
    houroffset=utils.local_time_offset_from_lats_lons(lats, lons)
    
    hours = np.array(fio.hours_available(mr))
    his = np.arange(len(hours))
    if hours_range is not None:
        his = his[hours_range]
        hours = hours[hours_range]

    for hi, hour_utc in zip(his,hours):
        # read hour
        DS_atmos = fio.read_model_run_hour(mr, extent=extent,hour=hi)
        #print(DS_atmos.info())
        # add u,v,z_th
        utils.extra_DataArrays(DS_atmos,add_z=True, add_winds=True,)
        times = DS_atmos.time.data #? 
        
        ltimes = utils.local_time_from_time_lats_lons(times, lats, lons)
        u,v,w = DS_atmos['u'], DS_atmos['v'], DS_atmos['vertical_wnd']
        level_height=DS_atmos['level_height'].data
        zi = np.argmin(np.abs(level_height - altitude))
        zstr = "%.1f m"%level_height[zi]
        
        # rather than calculate for all levels, just look at level of interest +-1
        #print("DEBUG: ",type(u[:,zi-1:zi+2].compute().data), u[:,zi-1:zi+2].compute().data.shape, )
        #print("DEBUG: ",type(level_height[zi-1:zi+2]), level_height[zi-1:zi+2].shape, )
        # need to load the lazy arrays into memory and put into numpy arrays here for gradient function to work (compute().data)
        rotation = utils.rotation(
            u[:,zi-1:zi+2].compute().data,
            v[:,zi-1:zi+2].compute().data,
            w[:,zi-1:zi+2].compute().data,
            level_height[zi-1:zi+2].compute(),
            lats,lons)[:,1]

        for ti, time in enumerate(ltimes):
            # convert dask lazy arrays into DataArrays (can be used like numpy arrays)
            #wi = w[ti,zi].compute()
            ui = u[ti,zi].compute()
            vi = v[ti,zi].compute()
            roti = rotation[ti]

            fire = DS_fire.sel(time=times[ti])
            sh = fire['SHEAT_2'].T # fire stuff is transposed :(
            
            # figure title and name will be based on model run, z, and local time
            title = mr + time.strftime(" %d %H:%M ") + "(UTC+%.2f)"%houroffset+'\n'+" rotation at %s"%(zstr)
            
            colormap = "seismic"
            contours = np.arange(-50,50.1,2.5)
            plt.contourf(lons,lats,roti,contours,cmap=colormap)
            plt.colorbar()
            plt.title(title)
            
            # add locations for reference
            if extent is not None:
                plotting.map_add_locations_extent(extent,hide_text=True)
            
            # add heat flux and quiver for 10m winds (fire products)
            plotting.map_sensibleheat(sh.data, lats, lons,
                    colorbar=False,)
            plotting.quiverwinds(lats, lons, ui, vi,
                    add_quiver_key=True,
                    alpha=0.7)
            
            fio.save_fig(mr,_SN_,time,plt,subdir=subdir)
            
            title2 = mr+time.strftime(" %d %H:%M ") + "(UTC+%.2f)"%houroffset+'\n'+r'w, diff winds at ' + zstr
            plt.contourf(lons,lats,roti,contours,cmap=colormap)
            plt.colorbar()
            plt.title(title2)
            
            # add locations for reference
            if extent is not None:
                plotting.map_add_locations_extent(extent,hide_text=True)
            
            # add heat flux and quiver for 10m winds (fire products)
            plotting.map_sensibleheat(sh.data, lats, lons,
                    colorbar=False,)
            
            plotting.quiverwinds(lats, lons, ui, vi,
                    differential=True,
                    alpha=0.7,
                    )
            fio.save_fig(mr,_SN_,time,plt,subdir=subdir+"_diff")

# This runs via the shell script "run_several_methods.sh"
def suitecall(mr, extent=None, subdir=None):
    if subdir is None:
        subdir=""
    # Check a bunch of overnight vert velocity levels
    zrange = [0.1, 0.5, 1, 2, 5]
    for z in zrange:
        plot_rotation(
            mr, 
            extent=extent, 
            subdir=subdir+"_%.2f"%(z),
            altitude=z*1000,
            hours_range=np.arange(2,16),
            )


            
if __name__=="__main__":
    # Check a bunch of overnight vert velocity levels
    #zrange = [0.1, 0.5, 1, 2, 3, 4, 5]
    zrange=[1,3,5]
    hrange=np.arange(5,13)
    #hrange=np.arange(5,6)
    for z in zrange:
        plot_rotation(
            "badja_am1",
            extent=constants.extents['badja_am']['zoom2'],
            subdir="zoom2_%.2f"%z,
            hours_range=hrange,
            altitude=z*1000.0,
            )
            
