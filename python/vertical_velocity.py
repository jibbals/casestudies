# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 11:26:52 2021

@author: jgreensl
"""

import matplotlib
#matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

#from matplotlib import colors
#from datetime import datetime, timedelta
from utilities import fio, utils, plotting, constants

_SN_ = "vertmotion"


    
def plot_vertical_velocity(
        mr, 
        extent=None, 
        subdir=None, 
        hours_range=None,
        use_agl=True,
        altitude=2000,
        altitude_2=None,
        ):
    """
    model run vertical velocity plots
    ARGS:
        model run name, ...
        altitude: z_th_agl to examine
        altitude_2: if this is set take average over altitude range
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
        z = DS_atmos['z_th']
        if use_agl:
            z = DS_atmos['z_th_agl']
        level_height=DS_atmos['level_height']
        zi = np.argmin(np.abs(level_height - altitude))
        zstr = "%.1f m"%level_height[zi]
        zi_2 = None
        if altitude_2 is not None:
            zi_2 = np.argmin(np.abs(level_height - altitude_2))
            assert zi_2 > zi, "Need zi < zi_2: %.3f %.3f "%(zi,zi_2) + str(level_height)
            zstr = "%.1f - %.1f m"%(level_height[zi],level_height[zi_2])
        
        for ti, time in enumerate(ltimes):
            vertvel = w[ti,zi].data
            ui = u[ti,zi].data
            vi = u[ti,zi].data
            if zi_2 is not None:
                vertvel = np.mean(w[ti,zi:zi_2+1].data,axis=0)
                ui = np.mean(u[ti,zi:zi_2+1].data,axis=0)
                vi = np.mean(v[ti,zi:zi_2+1].data,axis=0)
            
            fire = DS_fire.sel(time=times[ti])
            sh = fire['SHEAT_2'].T # fire stuff is transposed :(
            #u10 = fire['UWIND_2'].T
            #v10 = fire['VWIND_2'].T
            
            # figure title and name will be based on z0, z1, and local time
            title = mr+" w (m/s) %s"%(zstr) + time.strftime(" %d %H:%M ") + "(UTC+%.2f)"%houroffset
            figname = zstr + time.strftime(" %d %H:%M ")
            
            colormap = "seismic"
            contours = np.arange(-25,25,2.5)
            plt.contourf(lons,lats,w[ti],contours,cmap=colormap)
            plt.colorbar()
            plt.title(title)
            
            # add locations for reference
            if extent is not None:
                plotting.map_add_locations_extent(extent,hide_text=True)
            
            # add heat flux and quiver for 10m winds (fire products)
            plotting.map_sensibleheat(sh.data, lats, lons,
                    colorbar=False,)
            plotting.quiverwinds(lats, lons, ui, vi,
                    add_quiver_key=False,
                    alpha=0.7)
            
            fio.save_fig(mr,_SN_,title,plt,subdir=subdir)

# This runs via the shell script "run_several_methods.sh"
def suitecall(mr, extent=None, subdir=None):
    # Check a bunch of overnight vert velocity levels
    zrange = [0.1, 0.5, 1, 2, 5]
    for z in zrange:
        plot_vertical_velocity(
            mr, 
            extent=extent, 
            subdir=subdir+"_%.1f"%(z),
            altitude=z*1000,
            hours_range=np.arange(2,16),
            )


            
if __name__=="__main__":
    # Check a bunch of overnight vert velocity levels
    zrange = [0.1, 0.5, 1, 2, 3, 4, 5]
    for z in zrange:
        plot_vertical_velocity("badja_am1",
                extent=constants.extents['badja_am']['zoom2'],
                subdir="zoom2_%.2f"%z,
                hours_range=np.arange(5,13),
                altitude=z*1000.0,
                )
            
