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
from matplotlib import colors

from datetime import datetime, timedelta
from utilities import fio, utils, plotting

def read_model_timeseries(mr, 
                          latlon,
                          force_recreate=False,
                          interp_method="nearest",
                          ):
    """
    Read model, interpolated to latlon. Method saves time series for later use.
    Can force recreation of time series if it's already been created, otherwise just 
    read the old timeseries
    
    ARGUMENTS:
        mr: model run name
        latlon: [lat,lon] at which to interpolate model output profile
        force_recreate: bool - create time series from atmos output even if already done
        interp_method: method passed to DataSet.interp (default="nearest")
    RETURNS:
        xarray DataSet with profile [time,level]
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
                                       longitude=DA_lon,longitude_0=DA_lon,
                                       method=interp_method)
        
        #print("DEBUG: interped DS:",DS_atmos_point)
        # now need to merge into one bigger dataset
        if DS is None:
            DS = DS_atmos_point.copy(deep=True)
        else:
            if xr.__version__ >= '0.17.0': # NCI analysis3 env has this version
                DS = xr.combine_by_coords([DS,DS_atmos_point],combine_attrs="override")
            else:
                DS = xr.combine_by_coords([DS,DS_atmos_point])
    
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
    utc=Temp.time.values
    localtime = utils.local_time_from_time_lats_lons(utc,[lat],[lon])
    DS_plus_lt=DS.assign_coords(localtime=("time",localtime))
    
    fio.make_folder(fname)
    DS_plus_lt.to_netcdf(fname)
    print("INFO: SAVED NETCDF:",fname)
    return DS_plus_lt

def read_fire_time_series(mr, force_recreate=False):
    """
    Read/save model run time series for fire metrics, using 5 number summary and mean:
        fire power, fire speed, heat flux, (10m wind speeds?), 
    """
    fname = "../data/timeseries/"+mr+"_fire.nc"
    if os.path.isfile(fname) and not force_recreate:
        print("INFO: Reading already created netcdf fire timeseries:",fname)
        return xr.open_dataset(fname)
    
    DS_fire = fio.read_model_run_fire(mr)
    lats = DS_fire.lat.values
    lons = DS_fire.lon.values
    utc = DS_fire.time.values
    # constructed local time and area
    time = utils.local_time_from_time_lats_lons(utc,lats,lons)
    area = utils.lat_lon_grid_area(lats,lons)
    
    ## firepower in W/m2
    DA_SH =  DS_fire['SHEAT_2'] # [t, lons, lats]
    # [t, lons, lats] broadcasts with [lons,lats] to repeat along time dim
    firepower = np.sum(DA_SH.values * area.T * 1e-9,axis=(1,2)) # W/m2 * m2 * GW/W
    DA_firepower = xr.DataArray(firepower, dims=["time"],coords=[DS_fire.time])
    
    # LHEAT_2 comes from water_vapour output (units???)
    # less than SH by 6 magnitudes
    #DA_LH =  DS_fire['LHEAT_2'] # [t, lons, lats] 
    
    # fire_speed in m/s?
    # min val is .008...
    # so no zeros need to be masked out (I guess nans are in there)
    DA_FS = DS_fire['fire_speed'] # [t, lons, lats]
    DA_FS.load() # need to load for quantile
    # quantiles have shape [q, time]
    DA_FS_quantiles=DA_FS.quantile([0,.25,.5,.6,.7,.8,.9,.95,.96,.97,.98,.99,1],dim=("lat","lon"))
    
    
    DS_fire_timeseries = xr.Dataset({"firespeed_quantiles":DA_FS_quantiles, 
                                     "firepower":DA_firepower,})
    DS_fire_timeseries_plus_lt=DS_fire_timeseries.assign_coords(localtime=("time",time))
    
    fio.make_folder(fname)
    DS_fire_timeseries_plus_lt.to_netcdf(fname)
    print("INFO: SAVED NETCDF:",fname)
    return DS_fire_timeseries_plus_lt
    
def plot_firespeed(DS):
    """
    """
    # read fire speed timeseries
    DA_FS_quantiles=DS['firespeed_quantiles']
    time = DS.localtime.values
    quantiles = DA_FS_quantiles['quantile'].values
    for qi,q in enumerate(quantiles[5:]):
        plt.plot_date(time, DA_FS_quantiles[qi], label=q, fmt='-',color=plt.cm.Reds(q))
    plt.legend(title='quantiles')
    plt.title("fire speed (m/s?)")
    plt.gcf().autofmt_xdate()

def plot_firepower(DS):
    """
    """
    time=DS.localtime.values
    firepower=DS.firepower.values
    plt.plot_date(time,firepower,color='r',fmt='-',label='firepower')
    
    plt.gcf().autofmt_xdate()
    plt.ylabel('Gigawatts')
    plt.xlabel('local time')

def plot_fireseries(mr,extent=None,subdir=None):
    """
    """
    ## Read/create time series
    DS = read_fire_time_series(mr)
    time=DS.localtime.values
    firepower=DS.firepower.values
    DA_FS=DS['firespeed_quantiles']
    FS_q95 = DA_FS.sel(quantile=0.95).values
    FS_max = DA_FS[-1].values

    ## Plot stuff
    plt.plot_date(time,firepower,color='r',fmt='-',label='firepower')
    plt.ylabel('Gigawatts',color='r')
    plt.twinx()
    plt.plot_date(time,FS_q95, color='k',fmt='-', label='fire speed (95th pctile)')
    plt.plot_date(time,FS_max, color='k',fmt='-', label='max fire speed')
    plt.ylabel("firespeed")
    # create legend 
    
    
    plt.gcf().autofmt_xdate()
    plt.xlabel('local time')
    plt.title(mr+" fire series")
    fio.save_fig(mr, 
            plot_name="fire_series", 
            plot_time="fire_series",
            subdir=subdir,
            plt=plt,
            )


if __name__ == '__main__':
    mr="badja_run2"
    latlon=[-36.12,149.425]
    lat,lon = latlon
    
    plot_fireseries(mr)

    #DS=read_model_timeseries(mr,latlon)
    #DS_fire = read_fire_time_series(mr)
    #print(DS)
    #print(DS_fire)
    

    

