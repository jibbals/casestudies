# -*- coding: utf-8 -*-
# Script for calculating and plotting updraft helicity for ACCESS_fire case studies

import matplotlib.colors as col
import matplotlib.pyplot as plt
import numpy as np
np.seterr('ignore') 
from netCDF4 import Dataset
import netCDF4 as ncd
import scipy.io as sio # input/output
import time
from datetime import datetime as dt

# 2min output
stage = 300
time_int = '2019123006'
datadir = '/g/data/en0/dzr563/green_valley2_fix_2min_nc_data/'
figdir = '/g/data/en0/dzr563/Plots/green_valley_2min/UH_2min/'
print(datadir)
print(figdir)

# Map coordinates
map  = sio.loadmat('/home/563/dzr563/Python_Scripts/Australia_medium') # load matlab file
map_lon = map['Aust_medium'][:,0]
map_lat = map['Aust_medium'][:,1]

# Read netcdf file
ncufile = Dataset(datadir+'u_' + time_int + '_agl_{:.0f}m_2min.nc'.format(stage),'r')
ncvfile = Dataset(datadir+'v_' + time_int + '_agl_{:.0f}m_2min.nc'.format(stage),'r')
ncwfile = Dataset(datadir+'w_' + time_int + '_agl_{:.0f}m_2min.nc'.format(stage),'r')

# Dimensions
nt   = len(ncufile.dimensions['t'])
nlat = len(ncufile.dimensions['latitude'])
nlon = len(ncufile.dimensions['longitude'])

# Coordinates
t    = ncufile.variables['t'][:]
lat  = ncufile.variables['latitude' ][:]
lon  = ncufile.variables['longitude' ][:]
z  = ncufile.variables['height' ][:]

# Zoom
# xl = (lon.min(),lon.max()) # x axis length
# yl = (lat.min(),lat.max()) # y axis length
xl = (147.5,148.1)
yl = (-36.4,-35.7)

# Converting time to UTC
utc_time = dt.utcfromtimestamp(round(t[0]*3600))
print(utc_time)

levels = [0,1,2,3,4,5,6] # From surface (10m) to 3 km
for it in range(0, nt):
    print(t[it]*3600)

    vorts = None
    vert_velocities = None
    uh = None
    for i in range(0,len(levels)):
        lev = levels[i]
        u     = ncufile.variables['u'         ][it,lev,:,:]
        v     = ncvfile.variables['v'         ][it,lev,:,:]
        w     = ncwfile.variables['w'         ][it,lev,:,:]

        # For vorticity calculation
        lon2 = lon[np.newaxis,:] 
        lat2 = lat[:,np.newaxis]
        lon2_rad = np.radians(lon2)
        lat2_rad = np.radians(lat2)

        re = 6370e3 # m
        omega = 7.292115e-5 # 1/s
        R = 287 # JK-1kg-1
        cor_term =  2 * omega * np.sin(lat2_rad)

        # Allocation
        rel_vor =  np.zeros(u.shape,dtype=np.float32) 

        if i == 0:
            vorts = np.empty(len(levels),dtype=type(rel_vor))
            vert_velocities = np.empty(len(levels),dtype=type(w))
            uh = np.zeros(u.shape,dtype=np.float32)

        rel_vor[1:-1,1:-1] = ( (v[1:-1,2:]-v[1:-1,:-2])/( (lon2_rad[:,2:]-lon2_rad[:,:-2])*re*np.cos(lat2_rad[1:-1,:])) )-\
                        ( (u[2:,1:-1]-u[:-2,1:-1])/( (lat2_rad[2:,:]-lat2_rad[:-2,:])*re) ) +\
                        ((u[1:-1,1:-1]/re) * np.tan(lat2_rad[1:-1,:]))

        vorts[i] = rel_vor

        # Looking at updrafts,setting negative vert.velocity to zero.
        w[w<0] = 0
        vert_velocities[i] = w

    # Calculation based on Kain et al. (2008)
    for i in range(1,len(levels)):
        tmp = (((vert_velocities[i-1]*vorts[i-1])+(vert_velocities[i]*vorts[i]))/2) 
        uh = uh + tmp

    uh = uh * 500 # dz=500 m

    # Ploting
    uh_con = (-325,-300,-275,-250,-225,-200,-175,-150,-125,-100,-75,-50) # UH should be negative in SH for cyclonic
    uh_colormap = "YlGnBu"

    # Create figure and subplots
    fig = plt.figure(1,figsize=(12,7))
    plt.contourf(lon,lat,uh,uh_con,cmap=uh_colormap)
    plt.colorbar()
    # # # # Plot map of Australia
    plt.plot(map_lon,map_lat,'k-',linewidth=2.0)

    plt.xlabel('longitude') # x-axis label
    plt.ylabel('latitude')  # y-axis label
    plt.xlim(xl); plt.ylim(yl) # Axis limits

    plt.plot(147.63655,-35.94174,marker='o',color='k',markersize=6) # Lightowood, location of FGV and truck incident
    plt.plot(147.69578,-36.00827,marker='o',color='k',markersize=6) # Karumba
    plt.title(' Updraft helicity at ' + str(dt.utcfromtimestamp(round((t[it]*3600))+39600)) + ' LT' )
    plt.show()

ncufile.close()
ncvfile.close()
ncwfile.close()







