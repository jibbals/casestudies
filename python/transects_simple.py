#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 15:37:41 2021
    Vertical cross sections PLUS top down contextual map
@author: jesse
"""

import matplotlib
#matplotlib.use('Agg')

# plotting stuff
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter,LinearLocator, LogFormatter, ScalarFormatter
import numpy as np
import warnings
from datetime import datetime,timedelta
#from scipy import interpolate
#import cartopy.crs as ccrs

# local modules
from utilities import plotting, utils, constants
from utilities import fio_iris as fio

from transects import firefront_centres, interp_centres

###
## GLOBALS
###
_sn_ = 'transects_simple'

## Some plot stuff
# -32 -> 32 on log scale base 2
def get_symlog_scale(base, min_index,max_index):
    scale = np.union1d(
        np.union1d(
            base**np.arange(min_index,max_index),
            -1*(base**np.arange(min_index,max_index))
            ),
        np.array([0]) 
    )
    label=[]
    return scale

vertmotion_contours = get_symlog_scale(2.0,-2,6)

###
## METHODS
###

def cut_to_ztop(data_list,z,ztop):
    """
    ARGS:
        data_list: [data_arr1, ...]
        ztop: float of height to keep 
    return data_list with arrays all cut down in first dimension
    """

    # First we subset all the arrays so to be below the z limit
    zmin = np.min(z,axis=(1,2)) # lowest altitude on each model level
    ztopi = np.argmax(ztop<zmin)+1 # highest index where ztop is less than model level altitude
    
    retarr = []
    for darr in data_list:
        retarr.append(darr[:ztopi])
    
    return retarr

def mr_transect_hour(mr,hour,extent, start,end,
        ztop=10000,npoints=None,
        extra_cube_names=[],
        ):
    """
    ARGS:
        mr: model run name
        hour: int, which hour to look at
        extent: [W,E,S,N]
        start: [lat0,lon0]
        end: [lat1,lon1]
        ztop=10000: max altitude for transect
        npoints=None: set how many points along x axis you want (default is fine)
        extra_cube_names: list of variables to also take a transect of
            list with no extras is [u,v,upward_air_velocity,z_th,potential_temperature,]

    """

    umdtimes = fio.hours_available(mr)
    
    # hours input can be datetimes or integers
    umdtime = umdtimes[hour]
    
    # read cube list
    cubelist = fio.read_model_run(mr, 
                                  hours=[umdtime],
                                  extent=extent,
                                  )
    # add temperature, height, destaggered wind cubes
    utils.extra_cubes(cubelist,
                      add_theta=True,
                      add_z=True,
                      add_winds=True,)
    theta, = cubelist.extract('potential_temperature')
    lats=theta.coords("latitude").points
    lons=theta.coords("longitude").points
    dtimes = utils.dates_from_iris(theta)
    
    # read fire
    ff, sh = fio.read_fire(model_run=mr, 
                             dtimes=dtimes, 
                             extent=extent,
                             filenames=['firefront','sensible_heat'],
                             )
    
    # pull out bits we want
    u,v,w = cubelist.extract(['u','v','upward_air_velocity'])
    zcube, = cubelist.extract(['z_th'])
    
    # extra cubes too
    extra_cubes=[]
    if len(extra_cube_names) > 0:
        extra_cubes = cubelist.extract(extra_cube_names)
    
    transects_struct={}
    # loop over cubes and get the transect
    for cube in [u,v,w,zcube,sh,theta]+extra_cubes:
        # transect code
        cube_data=cube.data
        zcube_data=zcube.data
        if ztop is not None:
            cube_data,zcube_data=cut_to_ztop([cube.data,zcube.data],zcube.data,ztop)
        
        # do the transecting
        transect_data=utils.transect(cube_data,lats,lons,start,end,nx=npoints,z=zcube_data)
        
        # save for returning
        transects_struct[cube.name()]=transect_data['transect']

    # Also copy other useful stuff (will be same for all transects
    for tname in ['x','y','xdistance','xlats','xlons','xlabel']:
        transects_struct[tname]=transect_data[tname]

    return transects_struct

    

def transect_winds(u,v,w,z,
                  lats,lons,
                  start,end,
                  ztop=5000,
                  npoints=None,
                  n_arrows=20,
                  streamplot=False,
                  ):
    """
    Plot transect wind streamplot: uses utils.transect_winds to get winds along transect plane
    
    ARGUMENTS:
        u,v,w,z: arrays [lev,lats,lons] of East wind, N wind, Z wind, and level altitudes
        lats,lons: dims, in degrees
        start,end: [lat0, lon0], [lat1,lon1]
        topog: [lats,lons] surface altitude array
        ztop: top altitude to look at, defualt 5000m
    """
    
    if ztop is not None:
        u,v,w,z = cut_to_ztop([u,v,w,z],z,ztop)
    
    # interpolation points
    
    if npoints is None:
        npoints=utils.number_of_interp_points(lats,lons,start,end)
    
    # vertical wind speed along transect
    transect_w_struct = utils.transect(w,lats,lons,start,end,nx=npoints,z=z)
    transect_w = transect_w_struct['transect']
    
    # transect direction left to right wind speed
    transect_winds_struct = utils.transect_winds(u,v,lats,lons,start,end,nx=npoints,z=z)
    transect_s = transect_winds_struct['transect_wind']
    slicex = transect_winds_struct['x']
    slicez = transect_winds_struct['y']
    # scale for Y axis based on axes
    Yscale=ztop/transect_winds_struct['xdistance'][-1]
    
    retdict['xlabel'] = transect_winds_struct['xlabel']
    # left to right wind speed along transect
    retdict['s'] = transect_s
    # vertical wind motion along transect
    retdict['w'] = transect_w
    # x and y axis coordinates along transect
    retdict['x'] = slicex
    retdict['y'] = slicez
    # east to west, south to north wind speeds along transect
    retdict['u'] = transect_winds_struct['transect_u']
    retdict['v'] = transect_winds_struct['transect_v']
    retdict['yscale'] = Yscale
    # Streamplot
    if streamplot:
        plotting.streamplot_regridded(slicex,slicez,transect_s,transect_w,
                                      density=(.5,.5), 
                                      color='darkslategrey',
                                      zorder=1,
                                      minlength=0.2, # longer minimum stream length (axis coords: ?)
                                      arrowsize=1.5, # arrow size multiplier
                                      )
    else:
        plotting.quiverwinds(slicez,slicex,transect_s,transect_w,
                             n_arrows=n_arrows,
                             add_quiver_key=False,
                             alpha=0.5,
                             )
    plt.xlim(np.nanmin(slicex),np.nanmax(slicex))
    plt.ylim(np.nanmin(slicez),ztop)
    
    return retdict

if __name__ == '__main__':
    latlontimes=firefront_centres["KI_run1"]['latlontimes']
    # keep track of used zooms
    KI_zoom = [136.5,137.5,-36.1,-35.6]
    KI_zoom_name = "zoom1"
    KI_zoom2 = [136.5887,136.9122,-36.047,-35.7371]
    KI_zoom2_name = "zoom2"
    badja_zoom=[149.4,150.0, -36.4, -35.99]
    badja_zoom_name="zoom1"
    belowra_zoom=[149.61, 149.8092, -36.2535, -36.0658]
    belowra_zoom_name="Belowra"
    
    KI_jetrun_zoom=[136.6,136.9,-36.08,-35.79]
    KI_jetrun_name="KI_earlyjet"
    # settings for plots
    mr='badja_run2_exploratory'
    zoom=badja_zoom #belowra_zoom
    subdir=badja_zoom_name #belowra_zoom_name
    
