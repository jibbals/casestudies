# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    ##To run on NCI:
    ## REQUIRES HH5 project access
    ## need to load python environment like so
    module use /g/data3/hh5/public/modules
    module load conda/analysis3
@author: jesse
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

## METHODS

def destagger_winds_DA(DA_u, DA_v):
    """
    ## Staggered grid:
    ## u[latitude,longitude_edges] # minus one edge for some reason
    ## v[latitude_edges,longitude]
    ## fix by linear interpolation
    ARGS:
        DA_u = DataArray 'xwind' from model output with staggered longitudes
        DA_v = DataArray 'ywind' from model output with staggered latitudes
    RETURNS:
        u,v DataArrays on destaggered coordinates ['latitude','longitude']
    """
    if hasattr(DA_u,"latitude"):
        # move x wind onto y wind longitudes
        u = DA_u.interp(longitude_0=DA_v['longitude'].values)
        # then move y wind onto xwind latitudes
        v = DA_v.interp(latitude_0=DA_u['latitude'].values)
        # rename dimensions
        u=u.rename({"longitude_0":"longitude"})
        v=v.rename({"latitude_0":"latitude"})
    elif hasattr(DA_u,"lat"):
        # move x wind onto y wind longitudes
        u = DA_u.interp(lon_0=DA_v['lon'].values)
        # then move y wind onto xwind latitudes
        v = DA_v.interp(lat_0=DA_u['lat'].values)
        # rename dimensions
        u=u.rename({"lon_0":"lon"})
        v=v.rename({"lat_0":"lat"})
    else:
        print(DA_u)
        print("ERROR: COULDN'T FIND L(ong/at)ITUDE")
    
    return u,v

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
        
    return lons,lats
        interpolated degrees from start to end along lats,lons grid
    """
    x_factor = np.linspace(0,1,nx)
    lat0,lon0 = start
    lat1,lon1 = end
    lons_axis = lon0 + (lon1-lon0)*x_factor
    lats_axis = lat0 + (lat1-lat0)*x_factor
    return lats_axis,lons_axis

def transect(data, lats, lons, start, end, nx=10, z=None,
             interpmethod='linear'):
    '''
    interpolate along cross section
    USES XARRAY INTERPOLATION 
    inputs: 
        wind: data [[z], lats, lons]
        lats, lons: horizontal dims 1d arrays
        start = [lat0,lon0]
        end = [lat1,lon1]
        nx = how many points along horizontal. defaults to grid size
        z_th = optional altitude array [z, lats, lons]
    RETURNS: 
        struct: {
            'transect': vertical cross section of data 
            'x': 0,1,...,len(X axis)-1
            'y': y axis [Y,X] in terms of z
            'lats': [X] lats along horizontal axis
            'lons': [X] lons along horizontal axis
        } 
        xaxis: x points in metres
        yaxis: y points in metres or None if no z provided
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


## RUN

####### Figure: fire contours #######
##################################### 
# where is data output
filename = '../data/KI_run1_exploratory/atmos/umnsaa_2020010301_mdl_ro1.nc'

# read file into dataset
ds=xr.open_dataset(filename)
print(ds)

# pull out lats and lons for easy plotting
# just first timestep
U0=ds['wnd_ucmp'].isel(time=0)
V0=ds['wnd_vcmp'].isel(time=0)
U,V = destagger_winds_DA(U0,V0)
#print(U)
lats=ds.latitude.values
lons=ds.longitude.values

start=[lats[5],lons[5]]
end=[lats[-5],lons[-5]]


#print(U0) # can check what data looks like

# For now not worried about accurate levels (uses model levels)
U_transect = transect(U.values, lats,lons, start, end,)
X_u = U_transect['transect']
X_x = U_transect['x']
X_label = U_transect['xlabel']

V_transect = transect(V.values, lats,lons, start, end,)
X_v = V_transect['transect']
windspeed = np.sqrt(X_u**2+X_v**2)
plt.contourf(windspeed, cmap="Blues")
plt.xticks([0,3,6,9],X_label[::3], rotation=10)


figpath="standalone_transect.png"
plt.savefig(figpath)
print("Saved figure: ",figpath)
