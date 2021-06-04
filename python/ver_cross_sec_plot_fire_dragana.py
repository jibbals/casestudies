# -*- coding: utf-8 -*-

# Script for plotting vertical cross-sections.

import matplotlib
import matplotlib.colors as col
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
#import scipy.signal as sig
import scipy.io as sio
import scipy.signal as sig
# from matplotlib.mlab import griddata
from scipy import interpolate
from datetime import datetime as dt

# Data info
datadir_sh = '/g/data/en0/dzr563/ACCESS-fire/green_valley/2021-04-14/20191230T0300Z/0p3/atmos/'
datadir = '/g/data/en0/dzr563/green_valley_nc_data/'
figdir = '/g/data/en0/dzr563/Plots/au-aa945_greenvalley/Vertical_CS/'

time_int = '2019123006'

ncpfile = Dataset(datadir+'p_' + time_int + '_CS1_plane_300m_NWSE.nc','r')
ncwfile = Dataset(datadir+'w_' + time_int + '_CS1_plane_300m_NWSE.nc','r')
nctfile = Dataset(datadir+'temp_' +time_int +  '_CS1_plane_300m_NWSE.nc','r')
ncufile = Dataset(datadir+'u_' +time_int + '_CS1_plane_300m_NWSE.nc','r')
ncvfile = Dataset(datadir+'v_' + time_int + '_CS1_plane_300m_NWSE.nc','r')

ncshfile = Dataset(datadir_sh+'umnsaa_' + time_int + '_slv.nc','r')

nt  = len(ncpfile.dimensions['time'])
nz   = len(ncpfile.dimensions['height'])
nx = 100 # Number of points used for horiz.interp. to slice
ztop = 9000 # 10000 # 14000 # Optional upper limit to plot

# Lat and lon for sens.heat flux
nlat  = len(ncshfile.dimensions['latitude'])
nlon  = len(ncshfile.dimensions['longitude'])

lat  = ncshfile.variables['latitude' ][:]
lon  = ncshfile.variables['longitude' ][:]

# Green Valley
# CS2
# end1 = (147.55,-35.85)
# end2=(147.7,-36.15)
# CS1
end1 = (147.5,-35.85)
end2=(147.8,-36.05)

lon1,lat1 = end1
lon2,lat2 = end2
# Angle
dx = lon2 - lon1
dy = lat2 - lat1

# Cross-section angle, E-W cross-section angle = 0
angle = (np.rad2deg(np.arctan(dy/dx)))

x_plane = np.linspace(0.0,1.0,nx)
slicelon = lon1 + (lon2-lon1)*x_plane
slicelat = lat1 + (lat2-lat1)*x_plane
lon_plane = slicelon
lat_plane = slicelat

t   = ncpfile.variables['time'        ][:]
z    = ncpfile.variables['height'   ][:]
zu    = ncufile.variables['height'   ][:]
print(z)
numheights=nz

utc_time = dt.utcfromtimestamp(round((t[0]*3600)))
print(utc_time)

for tx in range(0,nt):
    
    w  = ncwfile.variables['w_plane'    ][tx,:,:]
    p  = ncpfile.variables['p_plane'    ][tx,:,:]
    temp  = nctfile.variables['temp_plane'    ][tx,:,:]
    u_cs  = ncufile.variables['u_plane'    ][tx,:,:]
    v_cs  = ncvfile.variables['v_plane'    ][tx,:,:]
    z_plane  = ncpfile.variables['z_plane'][tx,:,:]
    topo_plane  = ncpfile.variables['topog_plane'][tx,:]

    # Potential temperature (in K)
    theta = temp * ((1000.0/(p*1e-2))**0.286)
    # Sensible heat
    sh_flx  = ncshfile.variables['sens_hflx'][tx,:,:]

    # Along-section wind (wind magnitude)
    wind_mag = u_cs*np.cos(np.deg2rad(angle)) + v_cs*np.sin(np.deg2rad(angle))

    # Sensible heat flux interp. (necessary for plot)
    sh_flx_intrp = interpolate.RectBivariateSpline(lon,lat,sh_flx.transpose())
    sh_flx_plane = sh_flx_intrp.ev(lon_plane,lat_plane)

    # # Proceed to plotting

    # Set up a tuple of strings for the labels. Very crude!
    xticks = (0.0,0.5,1.0)
    xlabs = ('{:.2f}S {:.2f}E'.format(-lat1,lon1),'{:.2f}S {:.2f}E'.format(-0.5*(lat1+lat2),0.5*(lon1+lon2)),'{:.2f}S {:.2f}E'.format(-lat2,lon2))    
    # print(xlabs)
    # X-axis
    xaxis = np.tile(x_plane,(numheights,1))
    
    # Contours
    s_con  = np.arange(0,24,2)
    scon2 = np.arange(20,80,10)
    s_map = 'YlOrRd'
    
    w_con = np.arange(-8,8,0.5)
    w_con1 = np.arange(-8,8,2)
    wmap = plt.cm.get_cmap('bwr') # 'RdYlBu_r')
    wnorm = col.Normalize(vmin=-0.5,vmax=0.5)
    wnorm(0.)

    # Theta contour resources
    theta_con = np.arange(300,340,1)
    theta_map = 'rainbow'

    # Theta, vertical velocity
    fig = plt.figure(1,figsize=(16,8))
    plt.clf()
    ax = plt.subplot(1,1,1)
    plt.contourf(xaxis,z_plane,w,w_con,cmap=wmap, extend='both')
    wbar = plt.colorbar()
    wbar.ax.tick_params(labelsize=8) # Access to cbar tick labels (change font, etc.)
    thc = plt.contour(xaxis,z_plane,theta,theta_con,colors='k', linewidths = 0.5)
    plt.clabel(thc,thc.levels[::2],inline=True)

    plt.fill_between(x_plane,topo_plane,interpolate=True,facecolor='black')

    # # Colour topog where heat flux greater than some value (from Jesse)
    if (sh_flx is not None):
        if (np.max(sh_flx) < 1):
            plt.fill_between(x_plane,topo_plane,interpolate=True,facecolor='black')
        else:
            cmap = plt.get_cmap('plasma')
            normalize = col.SymLogNorm(vmin=0,vmax=10000,linthresh=100,base=10.0)
            # colour  where sh greater than
            for i in range(len(sh_flx_plane)-1):
                if sh_flx_plane[i]<100:
                    color = 'black'
                else:
                     color=cmap(normalize(sh_flx_plane[i]))
                     plt.fill_between([x_plane[i], x_plane[i+1]],
                                     [topo_plane[i],topo_plane[i+1]],
                                     color=color,
                                     zorder=2) # put on top of most things
    plt.xticks(xticks,xlabs)
    zstart = 0 #2000 # 4000 # m
    # plt.xlim(0.2,0.7)
    if ztop != None:
        plt.ylim(zstart,ztop)
    plt.xlabel('Lat/Lon', fontsize='10')
    plt.ylabel('Height', fontsize='10')
    plt.title('Pot.temperature and ver.velocity at ' + str(dt.utcfromtimestamp(round((t[tx]*3600)))) + ' UTC' )  # Plot title
    # fig.savefig(figdir + 'Theta_Ver_Vel_CS1_SH_' + str(dt.utcfromtimestamp(round(t[tx]*3600))) + '_UTC.png')
    # plt.close(fig)
    plt.show()

    fig = plt.figure(2,figsize=(16,8))
    plt.clf()
    ax = plt.subplot(1,1,1)
    plt.contourf(xaxis,z_plane,theta,theta_con,cmap=theta_map,extend='max')
    tbar = plt.colorbar()
    tbar.ax.tick_params(labelsize=8) # Access to cbar tick labels (change font, etc.)
    wc = plt.contour(xaxis,z_plane,w,w_con1,colors='k', linewidths = 1.0)
    plt.clabel(wc,inline=True)
    plt.fill_between(x_plane,topo_plane,interpolate=True,facecolor='black')

    # # Colour topog where heat flux greater than some value
    if (sh_flx is not None):
        if (np.max(sh_flx) < 1):
            plt.fill_between(x_plane,topo_plane,interpolate=True,facecolor='black')
        else:
            cmap = plt.get_cmap('plasma')
            normalize = col.SymLogNorm(vmin=0,vmax=10000,linthresh=100,base=10.0)
            # colour  where sh greater than
            for i in range(len(sh_flx_plane)-1):
                if sh_flx_plane[i]<100:
                    color = 'black'
                else:
                     color=cmap(normalize(sh_flx_plane[i]))
                     plt.fill_between([x_plane[i], x_plane[i+1]],
                                     [topo_plane[i],topo_plane[i+1]],
                                     color=color,
                                     zorder=2) # put on top of most things

    plt.xticks(xticks,xlabs)
    zstart = 0 #2000 # 4000 # m
    # plt.xlim(0.2,0.7)
    if ztop != None:
        plt.ylim(zstart,ztop)
    plt.xlabel('Lat/Lon', fontsize='10')
    plt.ylabel('Height', fontsize='10')
    plt.title('Pot.temperature and ver.velocity at ' + str(dt.utcfromtimestamp(round((t[tx]*3600)))) + ' UTC' )  # Plot title
    # fig.savefig(figdir + 'Theta_Con_Ver_Vel_CS1_SH_' + str(dt.utcfromtimestamp(round(t[tx]*3600))) + '_UTC.png')
    # plt.close(fig)
    plt.show()

    fig = plt.figure(2,figsize=(16,8))
    plt.clf()
    ax = plt.subplot(1,1,1)
    plt.contourf(xaxis,z_plane,wind_mag,s_con,cmap=s_map,extend='max')
    sbar = plt.colorbar()
    sbar.ax.tick_params(labelsize=8) # Access to cbar tick labels (change font, etc.)

    qx = 5
    Q = plt.quiver(xaxis[::qx,::qx],z_plane[::qx,::qx],u_cs[::qx,::qx],w[::qx,::qx],scale=25,units='inches')
    plt.quiverkey(Q, 0.1, 1.05, 5, r'$5 \frac{m}{s}$', labelpos='W', fontproperties={'weight': 'bold'})

    plt.fill_between(x_plane,topo_plane,interpolate=True,facecolor='black')

    # # Colour topog where heat flux greater than some value
    if (sh_flx is not None):
        if (np.max(sh_flx) < 1):
            plt.fill_between(x_plane,topo_plane,interpolate=True,facecolor='black')
        else:
            cmap = plt.get_cmap('plasma')
            normalize = col.SymLogNorm(vmin=0,vmax=10000,linthresh=100,base=10.0)
            # colour  where sh greater than
            for i in range(len(sh_flx_plane)-1):
                if sh_flx_plane[i]<100:
                    color = 'black'
                else:
                     color=cmap(normalize(sh_flx_plane[i]))
                     plt.fill_between([x_plane[i], x_plane[i+1]],
                                     [topo_plane[i],topo_plane[i+1]],
                                     color=color,
                                     zorder=2) # put on top of most things
    plt.xticks(xticks,xlabs)
    zstart = 0 #2000 # 4000 # m
    # plt.xlim(0.2,0.7)
    if ztop != None:
        plt.ylim(zstart,ztop)
    plt.xlabel('Lat/Lon', fontsize='10')
    plt.ylabel('Height', fontsize='10')
    plt.title('Wind magnitude at ' + str(dt.utcfromtimestamp(round((t[tx]*3600)))) + ' UTC' )  # Plot title
    # fig.savefig(figdir + 'Wind_CS1_SH_' + str(dt.utcfromtimestamp(round(t[tx]*3600))) + '_UTC.png')
    # plt.close(fig)
    plt.show()


ncwfile.close()
ncufile.close()
ncvfile.close()
nctfile.close()
ncpfile.close()







