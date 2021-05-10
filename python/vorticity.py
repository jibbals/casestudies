# -*- coding: utf-8 -*-
"""
Created on Thu May  6 14:47:04 2021

@author: jgreensl
"""

# plotting stuff
import matplotlib.pyplot as plt
#from matplotlib.ticker import FormatStrFormatter, LinearLocator#, LogFormatter
#from matplotlib import colors
import matplotlib
#import matplotlib.patheffects as PathEffects
import numpy as np
from scipy import signal

# local modules
from utilities import plotting, utils, fio

##
## METHODS
###

def calc_vorticity(lat,lon,u_wind,v_wind):
    """
    From dragana's script
    """
    # Necessary for vorticity calculation
    lon2 = lon[np.newaxis,:]
    lat2 = lat[:,np.newaxis]
    lon2_rad = np.radians(lon2)
    lat2_rad = np.radians(lat2)
    re = 6370e3 # m
    # # # # # For Coriolis parameter (planetary vort.)
    omega = 7.292115e-5 # 1/s
    R = 287
    cor_term = 2 * omega * np.sin(lat2_rad)
    
    # looks like u and v need to be [lat,lon]
    if u_wind.shape[0] == len(lon):
        u_wind = u_wind.T
        v_wind = v_wind.T
        
    rel_vor = np.zeros(u_wind.shape,dtype=np.float32)

    rel_vor[1:-1,1:-1] = ( (v_wind[1:-1,2:]-v_wind[1:-1,:-2])/( (lon2_rad[:,2:]-lon2_rad[:,:-2])*re*np.cos(lat2_rad[1:-1,:])) )-\
                               ( (u_wind[2:,1:-1]-u_wind[:-2,1:-1])/( (lat2_rad[2:,:]-lat2_rad[:-2,:])*re) ) +\
                               ((u_wind[1:-1,1:-1]/re) * np.tan(lat2_rad[1:-1,:]))

    # Setup for smoothing
    m2 = 5 #convrad
    m = 2*m2+1
    y = np.zeros((m,m))
    [y1,y2] = np.meshgrid(np.arange(0,m)-m2,np.arange(0,m)-m2)
    yr = np.hypot(y1,y2)
    y[yr <= m2] = 1
    y /= y.sum()
    rel_vor = signal.fftconvolve(rel_vor,y,mode='full')[m2:-m2,m2:-m2]
    
    return rel_vor

def plot_vorticity(DA_u,DA_v,DA_ff=None,DA_topog=None):
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

    # Relative vorticity)
    rel_vor = calc_vorticity(lats,lons,u,v)
    
    ## color map something
    avo_con = 1e-4*2**np.arange(0,9)
    avo_con = np.union1d(avo_con,-avo_con)
    avo_con = np.union1d(np.array([0]),avo_con)
    avo_map = ('RdBu_r')
    avo_norm = matplotlib.colors.SymLogNorm(2e-4,)#base=10)
    
    # Modify colormap to have white for certain range
    n=90
    x = 0.5
    lower = plt.cm.RdBu_r(np.linspace(0, x, n))
    white = plt.cm.RdBu_r(np.ones(100)*0.5)
    upper = plt.cm.RdBu_r(np.linspace(1-x, 1, n))
    colors = np.vstack((lower, white, upper))
    tmap = matplotlib.colors.LinearSegmentedColormap.from_list('terrain_map_white', colors)

    # Create figure and subplots

    # Vorticity
    #fig = plt.figure(1,figsize=(12,7))
    #ax = plt.subplot(1,1,1,aspect='equal') # One subplot
    rv = plt.contourf(lons,lats,rel_vor,avo_con,
                      cmap=tmap,
                      norm=avo_norm,
                      extend='both',
                      )
    rv_bar=plt.colorbar(rv) # Add colorbar to the plot
    if DA_ff is not None:
        plotting.map_fire(DA_ff.values,lats,lons)
    
    
    plotting.quiverwinds(lats,lons,u,v,
                         n_arrows=15,alpha=.5)
    #qx = 5
    #Q = plt.quiver(lon[::qx],lat[::qx],u10_wind.T[::qx,::qx],v10_wind.T[::qx,::qx],scale=1/0.02,units='inches')
    #plt.quiverkey(Q, 0.1, 1.05, 5, r'$5 \frac{m}{s}$', labelpos='W', fontproperties={'weight': 'bold'})

    #plt.plot(147.63655,-35.94174,marker='o',color='k',markersize=6) # Lightowood, location of FGV and truck incident
    #plt.plot(147.69578,-36.00827,marker='o',color='k',markersize=6) # Karumba
    #plt.plot(147.902,-36.196,marker='o',color='m',markersize=6) # Corryong

    # topography
    if DA_topog is not None:
        topog_con = (2,200,400,800)
        if len(lats) == DA_topog.shape[1]:
            topog=DA_topog.values.T
        else:
            topog = DA_topog.values
        plt.contour(lons,lats,topog,topog_con)

def vorticity_10m(
        mr,
        extent=None,
        subdir=None,
        timeskip=10,
        ):
    """
    
    ARGUMENTS:
        mr: model run name
        hours: optional run for subset of model hours
        extent: subset extent
        subdir: savefolder in case of specific extent
    """
    
    
    # read fire model output
    DS_fire=fio.read_model_run_fire(mr, extent=extent)
    
    lats = DS_fire.lat.values
    lons = DS_fire.lon.values
    times = DS_fire.time.data[::timeskip] # np.datetime64 array
    
    DA_u10 = DS_fire['UWIND_2'][::timeskip]
    DA_v10 = DS_fire['VWIND_2'][::timeskip]
    DA_ff = DS_fire['firefront'][::timeskip]
    
        
    houroffset = utils.local_time_offset_from_lats_lons(lats,lons)
    localtimes = utils.local_time_from_time_lats_lons(times,lats,lons)
    # loop over timesteps
    for ti,time_utc in enumerate(times):
            
        ## get local time
        time_lt=localtimes[ti]
        time_str=time_lt.strftime("%dT%H:%M")+"(UTC+%.2f)"%houroffset
                        
        ## FIRST FIGURE: 10m WIND DIR:
        fig=plt.figure()
        
        plot_vorticity(DA_u10[ti], DA_v10[ti], DA_ff[ti], )#topog)
        
        plt.title(mr + "\nvorticity (10m) " + time_str)
        plt.gca().set_aspect("equal")
            
        # save figure
        fio.save_fig(mr,"vorticity_10m", time_utc, plt, subdir=subdir)
        
        # comparison figure:
        rel_vor2,_,_,_ = utils.vorticity(DA_u10[ti].values.T,
                                   DA_v10[ti].values.T,
                                   lats,lons)
        
        avo_con = 1e-4*2**np.arange(0,9)
        avo_con = np.union1d(avo_con,-avo_con)
        avo_con = np.union1d(np.array([0]),avo_con)
        avo_map = plt.cm.RdBu_r
        avo_norm = matplotlib.colors.SymLogNorm(2e-4,)#base=10)
        tmap=plotting.cmap_with_white_range(avo_map,n=30,x=0.5,)
        
        rv2=plt.contourf(lons,lats,rel_vor2, avo_con,
                         cmap=tmap,
                         norm=avo_norm,)
        
        plotting.map_fire(DA_ff[ti].values,lats,lons)
        plotting.quiverwinds(lats,lons,DA_u10[ti].values.T,DA_v10[ti].values.T,
                             n_arrows=15,
                             alpha=.5)
        
        plt.gca().set_aspect("equal")
        plt.colorbar(rv2)
        fio.save_fig(mr,"vorticity_10m_compare",time_utc,plt,subdir=subdir)
        
    

if __name__ == '__main__':

    # keep track of used zooms
    KI_zoom = [136.5,137.5,-36.1,-35.6]
    KI_zoom_name = "zoom1"
    KI_zoom2 = [136.5887,136.9122,-36.047,-35.7371]
    KI_zoom2_name = "zoom2"
    badja_zoom=[149.4,150.0, -36.4, -35.99]
    badja_zoom_name="zoom1"


    mr="badja_run2_exploratory"
    zoom=badja_zoom
    zoom_name=badja_zoom_name
    vorticity_10m(mr,zoom,zoom_name)