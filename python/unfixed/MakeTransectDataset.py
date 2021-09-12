#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 09:36:23 2021
    Create model transect dataset
@author: jesse
"""

# math stuff
import numpy as np
#from scipy import interpolate

# file reading stuff
import xarray as xr
from glob import glob
from utilities.fio import read_model_run_hour
from utilities.utils import destagger_winds_DA
#from datetime import datetime,timedelta

###
## GLOBALS
###





###
## METHODS
###



def distance_between_points(latlon0,latlon1):
    """
    return distance between lat0,lon0 and lat1,lon1
        IN METRES
    
    calculated using haversine formula, shortest path on a great-circle
     - see https://www.movable-type.co.uk/scripts/latlong.html
    """
    R = 6371e3 # metres (earth radius)
    lat0, lon0 = latlon0
    lat1, lon1 = latlon1
    latr0 = np.deg2rad(lat0)
    latr1 = np.deg2rad(lat1)
    dlatr = np.deg2rad(lat1-lat0)
    dlonr = np.deg2rad(lon1-lon0)
    a = np.sin(dlatr/2.0)**2 + np.cos(latr0)*np.cos(latr1)*(np.sin(dlonr/2.0)**2)
    c = 2*np.arctan2(np.sqrt(a),np.sqrt(1-a))
    return R*c

def latslons_axes_along_transect(lats,lons,start,end,nx):
    """
    INPUTS:
        lats, lons: 1darrays of degrees
        start: [lat0,lon0] # start of transect
        end: [lat1,lon1] # end of transect
        
    return lats,lons
        interpolated degrees from start to end along lats,lons grid
    """
    x_factor = np.linspace(0,1,nx)
    lat0,lon0 = start
    lat1,lon1 = end
    lons_axis = lon0 + (lon1-lon0)*x_factor
    lats_axis = lat0 + (lat1-lat0)*x_factor
    return lats_axis,lons_axis

def number_of_xpoints_in_transect(lats,lons,start,end,factor=1.0):
    """
    get good number of grid points for transect based on native resolution.
    if grid size is 300m (diagonally about 424m), and transect is 42 400m, 
    then there are ~100 grid points native resolution diagonally.
    multiply this diagonal native available number by <factor> to get number of xpoints
    NB: error from high latitudes not accounted for

    ARGUMENTS:
        lats,lons to figure out grid size
        start = [lat0,lon0]
        end = [lat1,lon1]
        factor: multiply nx by some fudge factor if you want, default is to halve nx
    RETURNS:
        integer number of xpoints
    """
    fulldist=distance_between_points(start, end)
    gridres = distance_between_points([lats[0],lons[0]], [lats[1],lons[1]])
    
    # base interp points on grid size
    return(int(np.floor(factor*fulldist/gridres)))





def transect(DA, start, end, nx,
             interpmethod='linear'):
    '''
    interpolate along cross section
    USES XARRAY INTERPOLATION 
    inputs: 
        DA: xarray DataArray with latitude, longitude dimension
        start = [lat0,lon0]
        end = [lat1,lon1]
        nx = how many points along horizontal.
    RETURNS: 
            {
            'transect'   # [[y,]x] interpolation of data along transect
            'xdistance'  # [x] metres from start
            'x'          # [[y,]x] # metres from start, repeated along z dimension
            'y'          # [y,x] # interpolation of z input along transect
            'xlats'      # [x] # lats along x axis
            'xlons'      # [x] # lons along x axis
            'xlabel'     # [x] # (%.2f,%.2f)%(lat,lon) #  along x axis
            }
    '''
    lat1,lon1 = start
    lat2,lon2 = end
    
    # Interpolation line is really a list of latlons
    lataxis,lonaxis = latslons_axes_along_transect(lats,lons,start,end,nx=nx)
    # Create label to help interpret output
    label=["(%.2f, %.2f)"%(lat,lon) for lat,lon in zip(lataxis,lonaxis)]
    xdistance = np.array([distance_between_points(start, latlon) for latlon in zip(lataxis,lonaxis)])
    
    # Lets put our data into an xarray data array 
    coords = []
    if len(data.shape) ==3:
        coords = [("z",np.arange(data.shape[0]))]    
    coords.extend([("lats",lats),("lons",lons)])
    da = xr.DataArray(data,
                      coords)
    # we also put lat and lon list into data array with new "X" dimension
    da_lats = xr.DataArray(lataxis,dims="X")
    da_lons = xr.DataArray(lonaxis,dims="X")
    
    # interpolat to our lat,lon list
    slicedata = np.squeeze(da.interp(lats=da_lats,lons=da_lons,method=interpmethod).values)
    X=xdistance
    Y=None
    
    if z is not None:
        NZ=data.shape[0] # levels is first dimension
        da_z = xr.DataArray(z,
                            coords)
        # Y in 2d: Y [y,x]
        Y = np.squeeze(da_z.interp(lats=da_lats,lons=da_lons,method=interpmethod).values)
        # X in 2d: X [y,x]
        X = np.repeat(xdistance[np.newaxis,:],NZ,axis=0)
        
    return {'transect':slicedata, # interpolation of data along transect
            'xdistance':xdistance, # [x] metres from start
            'x':X, # [[y,]x] # metres from start, repeated along z dimension
            'y':Y, # [y,x] # interpolation of z input along transect
            'xlats':lataxis, # [x] 
            'xlons':lonaxis, # [x]
            'xlabel':label, # [x]
            }    

def transect_u_v_to_s(u,v,
                      start,end,
                      ):
    """
    Wind speed vector along transect line
    First calculates transect angle from East (0 is East, 90 is North, etc)
    then applies wind_mag = u * np.cos(theta_rads) + v * np.sin(theta_rads)
    
    ARGUMENTS:
        u[...,x]: east-west wind speed
        v[...,x]: north_south wind speed
        start: [lat,lon] start point for transect
        end: [lat,lon] end point for transect
        
    RETURNS: 
        S [..., x]: transect oriented wind magnitude
        
    """
    lat0,lon0=start
    lat1,lon1=end
    # signed angle in radians for transect line
    theta_rads=np.arctan2(lat1-lat0,lon1-lon0)
    theta_degs=np.rad2deg(theta_rads)
    print("CHECK: angle between", start, end)
    print("     : is ",theta_degs, "degrees?")
        
    wind_mag = u * np.cos(theta_rads) + v * np.sin(theta_rads)
    print("CHECK: mean(u)=%.2f, mean(v)%.2f, mean(s)%.2f"%(np.mean(u),np.mean(v),np.mean(wind_mag)))
    return wind_mag

def write_transect_dataset(dict_data):
    """
    Example data dictionary to write:
    {
     'coords': 
         {
        'x': {
            'dims': ('x',), 
            'attrs': {}, 
            'data': [10, 20, 30, 40]
            },
        'y': {
            'dims': ('y',), 
            'attrs': {}, 
            'data': [datetime.datetime(2000, 1, 1, 0, 0),    datetime.datetime(2000, 1, 2, 0, 0),    datetime.datetime(2000, 1, 3, 0, 0),    datetime.datetime(2000, 1, 4, 0, 0),    datetime.datetime(2000, 1, 5, 0, 0)]},
        'z': {
            'dims': ('x',), 
            'attrs': {}, 
            'data': ['a', 'b', 'c', 'd']
            }
        },
    'attrs': {},
    'dims': {'x': 4, 'y': 5},
    'data_vars': {
        'foo': {
            'dims': ('x', 'y'),
            'attrs': {},
            'data': [[0.12696983303810094,     0.966717838482003,     0.26047600586578334,     0.8972365243645735,     0.37674971618967135],    
                     [0.33622174433445307,     0.45137647047539964,     0.8402550832613813,     0.12310214428849964,     0.5430262020470384],
                     [0.37301222522143085,     0.4479968246859435,     0.12944067971751294,     0.8598787065799693,    0.8203883631195572],
                     [0.35205353914802473,     0.2288873043216132,     0.7767837505077176,     0.5947835894851238,     0.1375535565632705]]
            }
        }
    }

    ds_dict = xr.Dataset.from_dict(d)
    ds_dict
    Out[27]: 
        <xarray.Dataset>
        Dimensions:  (x: 4, y: 5)
        Coordinates:
          * x        (x) int64 10 20 30 40
          * y        (y) datetime64[ns] 2000-01-01 2000-01-02 ... 2000-01-04 2000-01-05
            z        (x) <U1 'a' 'b' 'c' 'd'
        Data variables:
            foo      (x, y) float64 0.127 0.9667 0.2605 0.8972 ... 0.7768 0.5948 0.1376
    """

def make_transect_dataset(path,start,end,
                          outfilename, nhours=24, nx=None):
    """
    Read model run hourly, build up transect data and store in netcdf
    ARGS:
        path: string path to parent of atmos folder
        start: [lat0,lon0] for start of transect
        end: [lat1,lon1] for end of transect
        outfilename: string with name for data output
        nhours: how many model hours available? default 24
        nx: how many grid points (default will give resolution of about 800m)
    """
    
    for hour in range(nhours):
        DS = read_model_run_hour(path,hour=hour)
        lats=DS['latitude'].values
        lons=DS['longitude'].values
        if hour==0:
            height_theta=DS['height_theta'] # [lev,lat,lon]
        if nx is None:
            nx = number_of_xpoints_in_transect(lats, lons, start, end)
        
        # we need to destagger xwind and ywind:
        u,v = destagger_winds_DA(DS['wnd_ucmp'],DS['wnd_vcmp']) # [time,lev,lat,lon]
        
        

if __name__ == '__main__':
    read_model_run_hour("../data/KI_run1_exploratory",hour=0)
