# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 16:51:15 2021
    Grey area where fire already burnt, 
    heat flux and 10m winds overlaid
    Run on badja zoom1:   
        CPU Time Used: 04:15:40  
        Memory Used: 7.41GB    
        Walltime Used: 04:15:54  

@author: jgreensl
"""

import matplotlib
#matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

#from matplotlib import colors
#from datetime import datetime, timedelta
from utilities import fio, utils, plotting, constants

_SN_ = "Helicity"

def Helicity_vertical(u, v, w, lats, lons):
    """
    Horizontal component of the environmental helicity (potential for helical flow)
    H_v = w(v_x - u_y)
    ARGS:
        u[...,y,x] : east-west wind (m/s)
        v[...,y,x] : north-south wind (m/s)
        w[...,y,x] : vertical motion (m/s)
    """
    u_lat,u_lon,v_lat,v_lon = utils.del_u_del_v(u,v,lats,lons)
    return w*(v_lon-u_lat)
    
def Updraft_Helicity(u,v,w, 
        z,
        lats, 
        lons, 
        z0=2000,
        z1=5000,
        remove_downdraft=True,
        ):
    """
    UH = integral{z0,z1} of w zeta(z) dz
       = sum(w*zeta*dt) where z is between z0 and z1
    dt = diff(edges(z)) where edges of z are interpolated linearly
    ARGS:
        u,v,w [...alts,lats,lons] : winds and vert motion (m/s)
        z[...alts,lats,lons] : altitude (m)
        lats,lons [1d] : (degrees)
        z0,z1 : integration start and finish (m)
        remove_downdraft : (True) set negative w to zero in UH calc
    """
    # we don't want to replace original vert motion
    w = w.copy()
    # u is west to east (longitudinal wind)
    # v is south to north (latitudinal wind)
    u_lat,u_lon,v_lat,v_lon = utils.del_u_del_v(u,v,lats,lons)
    
    # vorticity zeta [..., alts,lats,lons]
    zeta = v_lon - u_lat
    
    # edges of z [..., alts+1, lats, lons]
    z_edges = utils.edges(z,axis=-3)
    # dz [..., alts, lats, lons]
    dz = np.diff(z_edges,axis=-3)
    
    if remove_downdraft:
        w[w<0]=0

    # let's mask out everywhere outside the altitudinal range
    outside = np.logical_or(z < z0, z > z1)
    w[outside] = 0

    # integral from Kain et al. (2008)
    # [..., lat, lon]
    UH = np.sum(w*zeta*dz, axis=-3) # sum over altitude
    
    return UH
    
def plot_UH(mr, extent=None, subdir=None, 
            hours_range=None,
            use_agl=True,
            z0=2000,
            z1=5000,
            **UHArgs,
            ):
    """
    model run UH plots
    ARGS:
        model run name, ...
        Can pass in arguments to updraft_helicity
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
        UH = Updraft_Helicity(u.data, v.data, w.data, z.data, lats, lons,
                              z0=z0, z1=z1,
                              **UHArgs)
        
        for ti, time in enumerate(ltimes):
            
            fire = DS_fire.sel(time=times[ti])
            sh = fire['SHEAT_2'].T # fire stuff is transposed :(
            u10 = fire['UWIND_2'].T
            v10 = fire['VWIND_2'].T

            # figure title and name will be based on z0, z1, and local time
            title = "UH %.0f - %.0f"%(z0,z1) + time.strftime(" %d %H:%M ") + "(UTC+%.2f)"%houroffset
            
            # Plotting (Dragana color scale)
            #uh_con = (-325,-300,-275,-250,-225,-200,-175,-150,-125,-100,-75,-50) # UH should be negative in SH for cyclonic
            #uh_colormap = "YlGnBu"
            # allow negative vorticity too
            uh_colormap = "seismic"
            uh_con = np.arange(-300,301,25)
            plt.contourf(lons,lats,UH[ti],uh_con,cmap=uh_colormap)
            plt.colorbar()
            plt.title(title)
            
            # add locations for reference
            if extent is not None:
                plotting.map_add_locations_extent(extent,hide_text=True)
            
            # add heat flux and quiver for 10m winds (fire products)
            plotting.map_sensibleheat(sh.data, lats, lons,
                    colorbar=False,)
            plotting.quiverwinds(lats, lons, u10.data, v10.data,
                    add_quiver_key=False,
                    alpha=0.7)
            
            fio.save_fig(mr,_SN_,title,plt,subdir=subdir)

# This runs via the shell script "run_several_methods.sh"
def suitecall(mr, extent=None, subdir=None):
    if subdir is None:
        subdir=""
    for (z0,z1) in [(0,2000),(1000,3000),(2000,5000)]:
        plot_UH(mr, extent=extent, 
                subdir=subdir+"%d_%d"%(z0//1000,z1//1000),
                z0=z0,
                z1=z1,
                hours_range=np.arange(2,16),
                )


            
if __name__=="__main__":
    # Check a bunch of z0,z1 pairs
    zrange = [(0,1),(0,2),(0,3),(0,4),(0,5),
            (1,2),(1,3),(1,4),(1,5),
            (2,3),(2,4),(2,5),
            (3,5),]
    for (z0,z1) in zrange:
        plot_UH("badja_am1",
                extent=constants.extents['badja_am']['zoom2'],
                subdir="zoom2_%d_%d"%(z0,z1),
                hours_range=np.arange(5,13),
                z0=z0*1000,
                z1=z1*1000,
                )
            
