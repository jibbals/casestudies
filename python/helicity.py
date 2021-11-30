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
    print("DEBUG: u,v,w:", u.shape,v.shape,w.shape)
    u_lat, u_lon = np.gradient(u,mlats,mlons,axis=(-2,-1))
    v_lat, v_lon = np.gradient(v,mlats,mlons,axis=(-2,-1))
    print("DEBUG: u_lat,u_lon:", u_lat.shape,u_lon.shape)
    print("DEBUG: v_lat,v_lon:", v_lat.shape,v_lon.shape)
    return u_lat,u_lon,v_lat,v_lon

def Helicity_vertical(u,v,w, lats,lons):
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
        lats, 
        lons, 
        z,
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
        lats,lons [1d]: (degrees)
        z[...alts,lats,lons] : altitude (m)
        z0,z1 : integration start and finish (m)
        remove_downdraft : (True) set negative w to zero in UH calc
    """
    # u is left to right (longitudinal wind)
    # v is south to north (latitudinal wind)
    np.cumsumau_lat,u_lon,v_lat,v_lon = del_u_del_v(u,v,lats,lons)
    # vorticity zeta [..., alts,lats,lons]
    zeta = v_lon - u_lat
    # edges of z
    z_edges = utils.edges(z,axis=-3)
    # dz [..., alts, lats, lons]
    dz = np.diff(z,axis=-3,prepend=0) #pseudo first altitude of 0m so we keep shape
    
    if remove_downdraft:
        w=np.copy(w)
        w[w<0]=0

    # let's mask out everywhere outside the altitudinal range
    outside= (z < z0) * (z > z1)
    
    # integral from Kain et al. (2008)
    np

    
