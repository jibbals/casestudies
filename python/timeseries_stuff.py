# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 13:50:14 2021
    Time series of information for Jeff Keppert to use in his fire spotting program
@author: jgreensl
"""


import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os

from datetime import datetime, timedelta
from utilities import fio, utils, plotting

def read_model_timeseries(mr, 
                          latlon,
                          force_recreate=False,
                          ):
    """
    ARGUMENTS:
        mr: model run name
        latlon: [lat,lon] at which to interpolate model output profile
        
    RETURNS:
        xarray DataSet with profiles 
    """
    lat,lon = latlon
    
    fname = "../data/timeseries/"+mr+str(lat)+","+str(lon)+".nc"
    if os.path.isfile(fname) and not force_recreate:
        print("INFO: Reading already created netcdf timeseries:",fname)
        return xr.open_dataset(fname)
    
    # will interpolate onto new dimension "latlon" with associated coordinate
    DA_lat = xr.DataArray([lat], dims="latlon", coords={"lat":lat,"lon":lon})
    DA_lon = xr.DataArray([lon], dims="latlon", coords={"lat":lat,"lon":lon})
    
    DS=None
    for hour in range(24):
        
        DS_atmos = fio.read_model_run_hour(mr,hour)
        
        DS_atmos_point=DS_atmos.interp(latitude=DA_lat,latitude_0=DA_lat,
                                       longitude=DA_lon,longitude_0=DA_lon)
        
        #print("DEBUG: interped DS:",DS_atmos_point)
        # now need to merge into one bigger dataset
        if DS is None:
            DS = DS_atmos_point.copy(deep=True)
        else:
            DS = xr.combine_by_coords([DS,DS_atmos_point],combine_attrs="override")
    
    # Read firefront, heatflux (W/m2), U and V winds    
    # TODO calculate firepower
    # DS_fire = fio.read_model_run_fire(mr)
    
    # add wind speed
    U = DS.wnd_ucmp
    V = DS.wnd_vcmp
    S = xr.DataArray(np.hypot(U.data,V.data), 
                     dims=U.dims, coords=U.coords)
    DS["wind_speed"]=S
    
    # add wind direction
    WD = xr.DataArray(utils.wind_dir_from_uv(U.data,V.data),
                      dims=U.dims, coords=U.coords)
    WD.attrs["formulation"] = "wind_dir_rads = np.arctan2(v,u);   wind_dir = (-1*wind_dir_rads*180/np.pi - 90) % 360"
    DS["wind_direction"]=WD
    
    # add RH
    Q=DS.spec_hum
    Temp=DS.air_temp
    Press=DS.pressure
    Press_mb = Press.data/100.0 # Pa to hPa (millibars)
    RH = xr.DataArray(utils.relative_humidity_from_specific(Q.data,Temp.data,Press_mb),
                      dims=Q.dims, coords=Q.coords)
    RH.attrs["formulation"] = "es =  6.112 * np.exp((17.67 * tempC)/(tempC + 243.5)); e  = qair * press / (0.378 * qair + 0.622);    rh = e / es;    rh[rh > 1] = 1;    rh[rh < 0] = 0"
    DS["relative_humidity"] = RH
    
    
    fname = "../data/timeseries/"+mr+str(lat)+","+str(lon)+".nc"
    fio.make_folder(fname)
    DS.to_netcdf(fname)
    print("INFO: SAVED NETCDF:",fname)
    return DS
    
    #houroffset=utils.local_time_offset_from_lats_lons([lat],[lon])
    

if __name__ == '__main__':
    mr="badja_run1"
    latlon=[-36.12,149.425]
    DS = read_model_timeseries(mr,latlon)
    
    print (DS)
    
