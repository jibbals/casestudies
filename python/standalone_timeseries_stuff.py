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
from pandas import Timedelta


COMPAT='override' # xr open_mfdataset option
DATADIR="../data/" #, you may need to change DATADIR


def lat_lon_grid_edges(lats,lons):
    """
    take lats and lons, return grid edges
    """
    # using diff means we don't need regular grids
    dx = np.diff(lons) #lons[1]-lons[0]
    dy = np.diff(lats) #lats[1]-lats[0]
    
    lat_edges = np.zeros(len(lats)+1)
    # first edge is first lat - (dist to next lat) / 2.0
    lat_edges[0] = lats[0] - dy[0]/2.0
    # subsequent edges are orig lats - (dist to prior lat) / 2.0
    lat_edges[1:-1] = lats[1:]-dy/2.0
    # final edge is final lat + (dist to prior lat) / 2.0
    lat_edges[-1] = lats[-1]+dy[-1]/2.0
    
    lon_edges = np.zeros(len(lons)+1)
    lon_edges[0] = lons[0] - dx[0]/2.0
    lon_edges[1:-1] = lons[1:]-dx/2.0
    lon_edges[-1] = lons[-1]+dx[-1]/2.0
    return lat_edges,lon_edges

def lat_lon_grid_area(lats,lons):
    """
    Take lats and lons (grid centres), produce edges, return grid areas
    lats and lons need to be in degrees
    area returned is in square metres
    (taken from https://www.pmel.noaa.gov/maillists/tmap/ferret_users/fu_2004/msg00023.html)
    """
    lat_edges,lon_edges = lat_lon_grid_edges(lats,lons)
    lat_edges_rad = np.deg2rad(lat_edges)
    # 6.371 million metre radius for earth
    R = 6.3781e6
    Area_between_lats = 2 * np.pi * R**2.0 * np.abs(
            np.sin(lat_edges_rad[1:])-np.sin(lat_edges_rad[:-1])
            )
    Fraction_between_lons = np.abs(lon_edges[1:]-lon_edges[:-1])/360.0
    # repeat lon fraction array over number of lats
    ALon = np.tile(Fraction_between_lons,[len(lats),1]) # now [lats,lons]
    # repeat area between lats over lon space
    ALat = np.transpose(np.tile(Area_between_lats, [len(lons),1])) # now [lats,lons]
    
    # gridded area
    grid_area = ALat * ALon
    return grid_area

def local_time_from_time_lats_lons(time_utc,lats,lons):
    if isinstance(time_utc,datetime):
        time_utc=np.datetime64(time_utc)
        
    houroffset=local_time_offset_from_lats_lons(lats,lons)
    if hasattr(time_utc,'__iter__'):
        time_lt = [ np.datetime64((utc+Timedelta(houroffset,'h'))).astype(datetime) for utc in time_utc]
    else:
        time_lt=np.datetime64((time_utc+Timedelta(houroffset,'h'))).astype(datetime)
    return time_lt

def relative_humidity_from_specific(qair, temp, press = 1013.25):
    '''
    modified from https://earthscience.stackexchange.com/questions/2360/how-do-i-convert-specific-humidity-to-relative-humidity
    qair specific humidity, dimensionless (e.g. kg/kg) ratio of water mass / total air mass
    temp degrees K
    press pressure in mb
    
    return rh relative humidity, ratio of actual water mixing ratio to saturation mixing ratio
    Author David LeBauer
    '''
    tempC= temp-273.16
    es =  6.112 * np.exp((17.67 * tempC)/(tempC + 243.5))
    e  = qair * press / (0.378 * qair + 0.622)
    rh = e / es
    rh[rh > 1] = 1
    rh[rh < 0] = 0
    return(rh)

def wind_dir_from_uv(u,v):
    """
        input u and v
        returns wind_dir (data array):
            met wind source direction in degrees clockwise from due north
    """
    #wind_dir_rads = iris.analysis.maths.apply_ufunc(np.arctan2,v,u)
    wind_dir_rads = np.arctan2(v,u)
    # this is anticlockwise from directly east
    # meteorological wind dir: 0 is due north, + is clockwise
    # -WIND-IS-GOING-DIRECTION = (-1*wind_dir_rads.data*180/np.pi+90)%360
    # met standard points to where the wind is coming from
    wind_dir = (-1*wind_dir_rads*180/np.pi - 90) % 360
    return wind_dir

def make_folder(pname):
    """
    Create folder to hold file with path name argument
    EG: 
        make_folder("../data/timeseries/blah.nc")
        will create ../data/timeseries folder
    """
    folder = '/'.join(pname.split('/')[:-1]) + '/'
    if not os.path.exists(folder):
        print("INFO: Creating folder:",folder)
        os.makedirs(folder)

def read_model_run_hour(mr, hour=0):
    """
    Read wind,temperature,pressure,altitude data from model run
    ARGUMENTS:
        mr: model run name, will match folder in ../data/
            can also be full path to parent folder to /fire and /atmos for access-fire model run output
        hour: integer   # 0 is first hour, 1 is second hour, ...
    """
    atmosdir=DATADIR+mr+"/atmos/"
    if not os.path.isdir(atmosdir):
        print("ERROR:",atmosdir," is not a directory")
        assert False, "mr needs to point to dir holding /atmos/ folder"
   
    allfiles=glob(atmosdir+"*.nc")
    allfiles.sort()
   
    # 4 files per model hour
    hourfiles = allfiles[hour*4:hour*4+4]
    print("INFO: will read files:")
    print("    :",hourfiles)
   
    DS = xr.open_mfdataset(hourfiles,compat=COMPAT)
    return DS

def read_model_run_fire(mr):
    """
    """
    fdir=DATADIR+mr+"/fire/"
    firepaths=glob(fdir+"*00Z.nc")
    DS = xr.open_mfdataset(firepaths,compat=COMPAT)
       
    return(DS)


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
        
        DS_atmos = read_model_run_hour(mr,hour)
        
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
    
    # add wind speed
    U = DS.wnd_ucmp
    V = DS.wnd_vcmp
    S = xr.DataArray(np.hypot(U.data,V.data), 
                     dims=U.dims, coords=U.coords)
    DS["wind_speed"]=S
    
    # add wind direction
    WD = xr.DataArray(wind_dir_from_uv(U.data,V.data),
                      dims=U.dims, coords=U.coords)
    WD.attrs["formulation"] = "wind_dir_rads = np.arctan2(v,u);   wind_dir = (-1*wind_dir_rads*180/np.pi - 90) % 360"
    DS["wind_direction"]=WD
    
    # add RH
    Q=DS.spec_hum
    Temp=DS.air_temp
    Press=DS.pressure
    Press_mb = Press.data/100.0 # Pa to hPa (millibars)
    RH = xr.DataArray(relative_humidity_from_specific(Q.data,Temp.data,Press_mb),
                      dims=Q.dims, coords=Q.coords)
    RH.attrs["formulation"] = "es =  6.112 * np.exp((17.67 * tempC)/(tempC + 243.5)); e  = qair * press / (0.378 * qair + 0.622);    rh = e / es;    rh[rh > 1] = 1;    rh[rh < 0] = 0"
    DS["relative_humidity"] = RH
    utc=Temp.time.values
    localtime = local_time_from_time_lats_lons(utc,[lat],[lon])
    DS_plus_lt=DS.assign_coords(localtime=("time",localtime))
    
    make_folder(fname)
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
    
    DS_fire = read_model_run_fire(mr)
    lats = DS_fire.lat.values
    lons = DS_fire.lon.values
    utc = DS_fire.time.values
    # constructed local time and area
    time = local_time_from_time_lats_lons(utc,lats,lons)
    area = lat_lon_grid_area(lats,lons)
    
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
    
    make_folder(fname)
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

def plot_fireseries(mr):
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
    plt.savefig(mr+"_fire_series.png")


if __name__ == '__main__':
    mr="badja_run2"
    latlon=[-36.12,149.425]
    lat,lon = latlon
    
    plot_fireseries(mr)

    #DS=read_model_timeseries(mr,latlon)
    #DS_fire = read_fire_time_series(mr)
    #print(DS)
    #print(DS_fire)
    

    

