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


def del_u_del_v(u,v,lats,lons):
    """
    calculate metres per degree, 
    then take gradients along u and v in the horizontal dimensions.
    I think this maintains [...,y,x] shape
    ARGS:
        u,v [...,y,x]: winds horizontal
        lats,lons: in degrees
    """

    lat_deg_per_metre = 1/111.32e3 # 111.32km per degree
    lat_mean = np.mean(lats)
    lon_deg_per_metre = lat_deg_per_metre * np.cos(np.deg2rad(lat_mean))
    
    mlats = lats / lat_deg_per_metre # convert lats into metres
    mlons = lons / lon_deg_per_metre # convert lons into metres
    
    # array[...,lat,lon]
    print("DEBUG: u,v,w:", u.shape, v.shape)
    u_lat, u_lon = np.gradient(u, mlats, mlons, axis=(-2,-1))
    v_lat, v_lon = np.gradient(v, mlats, mlons, axis=(-2,-1))
    print("DEBUG: u_lat,u_lon:", u_lat.shape, u_lon.shape)
    print("DEBUG: v_lat,v_lon:", v_lat.shape, v_lon.shape)
    return u_lat, u_lon, v_lat, v_lon

def Helicity_vertical(u, v, w, lats, lons):
    """
    Horizontal component of the environmental helicity (potential for helical flow)
    H_v = w(v_x - u_y)
    ARGS:
        u[...,y,x] : east-west wind (m/s)
        v[...,y,x] : north-south wind (m/s)
        w[...,y,x] : vertical motion (m/s)
    """
    u_lat,u_lon,v_lat,v_lon = del_u_del_v(u,v,lats,lons)
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
        u,v,w [...alts,lats,lons]: winds and vert motion (m/s)
        z[...alts,lats,lons] : altitude (m)
        lats,lons [1d]: (degrees)
        z0,z1 : integration start and finish (m)
        remove_downdraft : (True) set negative w to zero in UH calc
    """
    # we don't want to replace original vert motion
    w = w.copy()
    # u is left to right (longitudinal wind)
    # v is south to north (latitudinal wind)
    u_lat,u_lon,v_lat,v_lon = del_u_del_v(u,v,lats,lons)
    # vorticity zeta [..., alts,lats,lons]
    zeta = v_lon - u_lat
    # edges of z
    z_edges = utils.edges(z,axis=-3)
    # dz [..., alts, lats, lons]
    dz = np.diff(z,axis=-3,prepend=0) #pseudo first altitude of 0m so we keep shape
    
    if remove_downdraft:
        w[w<0]=0

    # let's mask out everywhere outside the altitudinal range
    outside = (z < z0) * (z > z1)
    w[outside] = 0
    
    # integral from Kain et al. (2008)
    # [..., lat, lon]
    UH = np.cumsum(w*zeta*dz, axis=-3) # sum over altitude
    
    return UH
    
def plot_UH(mr, extent=None, subdir=None, 
            z0=2000,
            z1=5000,
            ):
    """
    """
    DS_fire = fio.read_model_run_fire(mr,extent=extent)
    lats=DS_fire.lat.data
    lons=DS_fire.lon.data
    #times=DS_fire.time.data
    houroffset=utils.local_time_offset_from_lats_lons(lats, lons)
    
    hours = fio.hours_available(mr)
    for hi, hour_utc in hours:
        # read hour
        DS_atmos = fio.read_model_run_hour(mr, extent=extent,hour=hi)
        # add u,v,z_th
        utils.extra_DataArrays(DS_atmos,add_z=True, add_winds=True,)
        times = DS_atmos.time.data #? 
        print(DS_atmos)
        print("DEBUG: ",hi, hour_utc)
        print("DEBUG: ",times)
        ltimes = utils.local_time_from_time_lats_lons(times, lats, lons)
        u,v,w = DS_atmos['u','v','vertical_wnd']
        z = DS_atmos['z_th']
        UH = Updraft_Helicity(u.data, v.data, w.data, z.data, lats, lons,
                              z0=z0, z1=z1)
        
        for ti, time in enumerate(ltimes):
            
            fire = DS_fire.sel(time=times[ti])
            sh = fire['SHEAT_2']
            title = "UH %.0f - %.0f"%(z0,z1) + time.strftime(" %d %H:%M ") + "(UTC+%.2f)"%houroffset
            
            # Plotting
            uh_con = (-325,-300,-275,-250,-225,-200,-175,-150,-125,-100,-75,-50) # UH should be negative in SH for cyclonic
            uh_colormap = "YlGnBu"

            plt.contourf(lons,lats,UH[ti].data,uh_con,cmap=uh_colormap)
            plt.colorbar()
            plt.title(title)
            
            if extent is not None:
                plotting.map_add_locations_extent(extent)
            
            plotting.map_sensibleheat(sh.data, lats, lons)
            
            plotting.quiverwinds(lats, lons, u.data, v.data,
                                 alpha=0.6)
            
            fio.save_fig(mr,_SN_,title,plt,subdir=subdir)
            
if __name__=="__main__":
    plot_UH("badja_am1",extent=constants.extents['badja_am1']['zoom2'],subdir="zoom2")
            