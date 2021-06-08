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

import pandas
from glob import glob
from datetime import datetime, timedelta
from utilities import fio, utils, plotting,constants
from utilities.fio_iris import read_PFT_timeseries

AWS_rename_columns = {
            "Latitude to four decimal places in degrees":"latitude",
            "Longitude to four decimal places in degrees":"longitude",
            "Day/Month/Year Hour24:Minutes in DD/MM/YYYY HH24:MI format in Local time":"localtime",
            "Day/Month/Year Hour24:Minutes in DD/MM/YYYY HH24:MI format in Universal coordinated time":"utc",
            #"Precipitation in last 10 minutes in mm":,
            #"Quality of precipitation in last 10 minutes",
            #"Precipitation since 9am local time in mm",
            #"Quality of precipitation since 9am local time",
            "Air Temperature in degrees C":"temperature",
            #"Quality of air temperature":,
            "Wet bulb temperature in degrees C":"temperature_bulb",
            #"Quality of Wet bulb temperature",
            "Dew point temperature in degrees C":"temperature_dew",
            #"Quality of dew point temperature,
            "Relative humidity in percentage %":"RH",
            #"Quality of relative humidity",
            #"Vapour pressure in hPa",Quality of vapour pressure,
            #Saturated vapour pressure in hPa,Quality of saturated vapour pressure,
            "Wind speed in m/s":"windspeed_ms-1",
            #Wind speed quality,
            "Wind direction in degrees true":"winddir",
            #Wind direction quality,
            "Speed of maximum windgust in last 10 minutes in m/s":"wind_gust_10minute_ms-1",
            #Quality of speed of maximum windgust in last 10 minutes
            "Mean sea level pressure in hPa":"mslp_hPa",
            #Quality of mean sea level pressure,
            "Station level pressure in hPa":"pressure_hPa",
            #Quality of station level pressure,
            #QNH pressure in hPa,
            #Quality of QNH pressure,
            #AWS Flag,Error Flag,#
#            }
#PAWS_rename_columns = {
    #Station Name	
    "UTC DateTime":"utc",
    "EDST Date Time":"localtime",
    "Dew Point Temp C":"temperature_dew",
    #Day
    #latitude
    #longitude
    "Temp C":"temperature",
    "Calc RH %":"RH",
    #"Direct RH %"
    "Wind Direction (Average 10min)":"winddir",
    "10 Metre Wind 10min Average (Km/h)":"windspeed_kmh-1",
    "2 Metre Wind 10min Average (Km/h)":"windspeed_2metre_kmh-1",
    #Wind 10min Average (Knots)	"Rain to 0900am mm"
    #Rain mm 
    #Rain 10min mm
    #Solar Radiation  W/m2
    #Wind Gust (knots)
    #2m Wind Gust km/h
    "10m Wind Gust km/h":"wind_gust_10minute_kmh-1",
    #Battery Voltage Volts
    #Internal Temp C
    "FDI Forest McArthur":"FFDI",
    }



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
    
    # ADD local time
    utc=Temp.time.values
    localtime = utils.local_time_from_time_lats_lons(utc,[lat],[lon])
    DS_plus_lt=DS.assign_coords(localtime=("time",localtime))
    
    fio.make_folder(fname)
    DS_plus_lt.to_netcdf(fname)
    print("INFO: SAVED NETCDF:",fname)
    return DS_plus_lt

def read_fire_time_series(mr, 
                          latlon=None, 
                          force_recreate=False,
                          interp_method="nearest"):
    """
    Read/save model run time series for fire metrics, using 5 number summary and mean:
        fire power, fire speed, heat flux, (10m wind speeds?), 
    Note from Harvey:
        The speed will be zero before the fire start. 
        And its minimum will be 0.001 (I think) after the fire start. 
        I believe the issue is not harmful, but worth fixing.
    """
    if latlon is None:
        fname = "../data/timeseries/"+mr+"_fire.nc"
    else:
        lat,lon = latlon
        fname = "../data/timeseries/"+mr+str(lat)+","+str(lon)+".nc"
        
    if os.path.isfile(fname) and not force_recreate:
        print("INFO: Reading already created netcdf fire timeseries:",fname)
        return xr.open_dataset(fname)
    
    fire_rename_dict={
            "HUMID_2":"surface_RH",
            "LHEAT_2":"latent_heat",
            "SHEAT_2":"sensible_heat",
            "TEMPE_2":"surface_temperature",
            "UWIND_2":"u_10m",
            "VWIND_2":"v_10m",
            }
    
    
    DS_fire = fio.read_model_run_fire(mr)
    lats = DS_fire.lat.values
    lons = DS_fire.lon.values
    utc = DS_fire.time.values
    # constructed local time and area
    time = utils.local_time_from_time_lats_lons(utc,lats,lons)
    area = utils.lat_lon_grid_area(lats,lons)
    
    ## firepower in W/m2
    DA_SH =  DS_fire['SHEAT_2'] # [t, lons, lats]
    DA_u = DS_fire['UWIND_2'] # [t, lons, lats]
    DA_v = DS_fire['UWIND_2'] # [t, lons, lats]
    DA_s = xr.DataArray(np.hypot(DA_u.values,DA_v.values), coords=DA_v.coords, dims=DA_v.dims, name="windspeed_10m")
    DA_s_quantiles = DA_s.quantile([0,.25,.5,.6,.7,.75,.8,.9,.95,.96,.97,.98,.99,1],dim=("lat","lon"))
    
    # [t, lons, lats] broadcasts with [lons,lats] to repeat along time dim
    firepower = np.sum(DA_SH.values * area.T * 1e-9,axis=(1,2)) # W/m2 * m2 * GW/W
    DA_firepower = xr.DataArray(firepower, dims=["time"],coords=[DS_fire.time])
    
    
    # LHEAT_2 comes from water_vapour output (units???)
    # less than SH by 6 magnitudes
    #DA_LH =  DS_fire['LHEAT_2'] # [t, lons, lats] 
    
    # fire_speed in m/s?
    # min val is .001
    ## MASK OUT VALUES BELOW .00105
    DA_FS = DS_fire['fire_speed'] # [t, lons, lats]
    DA_FS.load() # need to load for quantile
    # quantiles have shape [q, time]
    #plt.figure()
    #plt.plot(DA_FS.quantile(0.5,dim=("lat","lon")),label="median")
    #plt.plot(DA_FS.quantile(0.95,dim=("lat","lon")),label="95th pctile")
    #plt.plot(DA_FS.where(DA_FS.values>0.00105).quantile(0.5,dim=("lat","lon")),label="median (masked)")
    #plt.plot(DA_FS.where(DA_FS.values>0.00105).quantile(0.95,dim=("lat","lon")),label="95th pctile (masked)")
    #plt.legend()
    #plt.savefig("firespeed_masking_check")
    #plt.close()
    #print("DEBUG: saved figure:","firespeed_masking_check")
    DA_FS_quantiles=DA_FS.where(DA_FS.values>0.00105).quantile([0,.25,.5,.6,.7,.75,.8,.9,.95,.96,.97,.98,.99,1],dim=("lat","lon"))
    
    ## If we have a latlon we add 10m winds
    DS_fire_timeseries=xr.Dataset()
    if latlon is not None:

        # will interpolate onto new dimension "latlon" with associated coordinate
        DA_lat = xr.DataArray([lat], dims="latlon", coords={"lat":lat,"lon":lon})
        DA_lon = xr.DataArray([lon], dims="latlon", coords={"lat":lat,"lon":lon})
        
        DS_fire_timeseries=DS_fire.interp(
                lat=DA_lat,
                lon=DA_lon,
                method=interp_method,
                )
        DS_fire_timeseries = DS_fire_timeseries.rename_vars(fire_rename_dict)
        u = DS_fire_timeseries.u_10m.values
        v = DS_fire_timeseries.v_10m.values
        wdir = utils.wind_dir_from_uv(u,v)
        DS_fire_timeseries["wdir_10m"] = xr.DataArray(wdir,
                          dims=DS_fire_timeseries.u_10m.dims,
                          coords=DS_fire_timeseries.u_10m.coords)
    
    DS_fire_timeseries["windspeed_10m_quantiles"]=DA_s_quantiles
    DS_fire_timeseries["firespeed_quantiles"]=DA_FS_quantiles
    DS_fire_timeseries["firepower"]=DA_firepower
    
    DS_fire_timeseries_plus_lt=DS_fire_timeseries.assign_coords(localtime=("time",time))
    
    
    ## Save file
    fio.make_folder(fname)
    DS_fire_timeseries_plus_lt.to_netcdf(fname)
    print("INFO: SAVED NETCDF:",fname)
    return DS_fire_timeseries_plus_lt
    

def fireseries(mr,extent=None,subdir=None,
        GW_max=None,):
    """
    show model run firepower, maximum fire speed, and 95th pctile of fire speed
    ARGS:
        GW_max: maximum gigawatts for left y axis
    """
    ## Read/create time series
    DS = read_fire_time_series(mr)
    time=DS.localtime.values
    firepower=DS.firepower.values
    DA_FS=DS['firespeed_quantiles']
    DA_WS=DS['windspeed_10m_quantiles']
    
    ## Plot stuff
    ## Two subplots, top is fire power and fire speed
    ## bottom is wind speed distribution at 10m altitude
    fig,axes = plt.subplots(nrows=2,ncols=1,sharex=True,sharey=False)
    plt.sca(axes[0])
    
    plt.plot_date(time,firepower,color='r',fmt='-',label='firepower')
    plt.ylabel('Gigawatts',color='r')
    if (GW_max is not None) and (np.max(firepower) > GW_max):
        plt.ylim(0,GW_max)
    plt.twinx()
    for pctile,fscolor in zip([ .95, .99, 1.0],['grey','grey','k']):
        plt.plot_date(time, DA_FS.sel(quantile=pctile).values * 3.6, 
                      color=fscolor,
                      fmt='-',
                      label=pctile*100,
                      )
    plt.legend(title="percentile",
               prop={'size': 4},
               )
    plt.ylabel("firespeed (km/h)")
    plt.xticks([],[])
    plt.title(mr+" fire")
    
    ## second subplot
    plt.sca(axes[1])
    for pctile,wscolor in zip([0.5,0.75,0.9,0.95,1.00],['grey','grey','grey','grey','k']):
        plt.plot_date(time, DA_WS.sel(quantile=pctile).values*3.6,
                      color=wscolor,
                      fmt='-',
                      label=pctile*100,
                      )
    plt.legend(
        prop={'size': 5},
        #title="percentile",
        )
    plt.ylabel("10m wind speed (km/h)")
    windspeed_max=DA_WS.sel(quantile=1.0).values*3.6
    windmax=np.max([136.3636, 
                    np.max(windspeed_max),
                    ])
    # annotate maximum value
    ymax = np.max(windspeed_max)
    xpos, = np.where(windspeed_max==ymax)
    xmax = time[xpos]
    plt.annotate("%.2f"%ymax,
        xy=(xmax, ymax), 
        xytext=(xmax, ymax*1.02),
        fontsize=7,
        )
    

    plt.ylim(0,windmax*1.1)
    if windmax > 180:
        windticks = [0,30,60,90,120, 150, 180]
    elif windmax > 150:
        windticks = [0,30,60,90,120, 150]
    else:
        windticks = [0,30,60,90,120]
    
    plt.yticks(windticks,windticks)
    
    plt.gcf().autofmt_xdate()
    plt.xlabel('local time')
    plt.subplots_adjust(hspace=0.0)
    fio.save_fig(mr, 
            plot_name="fire_series", 
            plot_time="fire_series",
            subdir=subdir,
            plt=plt,
            )

def DF_subset_time(DF, dt0=None, dt1=None, timename='localtime'):
    """
    subset AWS to time window dt0,dt1
    can do utc if you change timename to 'utc'
    """
    if dt0 is not None:
        time_mask = DF[timename] >= dt0
        DF = DF.loc[time_mask]
    
    if dt1 is not None:
        time_mask = DF[timename] <= dt1
        DF = DF.loc[time_mask]
    
    return DF


def read_PAWS_belowra(lt0=None, lt1=None):
    """
    Read PAWS data
    can subset to date window [lt0, lt1]
    """
    ddir="../data/AWS/"
    #AWS_allfiles=glob(ddir+"PAWS*")
    paws_file=ddir+"PAWS_belowra.csv"
    
    
    DF_PAWS = pandas.read_csv(paws_file, index_col=None, header=0)
    DF_PAWS.rename(columns=AWS_rename_columns,inplace = True)
    
    # convert the 'Date' column to datetime format
    for dtcolumn in ["localtime","utc"]:
        DF_PAWS[dtcolumn] = pandas.to_datetime(DF_PAWS[dtcolumn],dayfirst=True)

    # Ensure numerical columns are numerical
    for numcolumn in ["temperature","temperature_dew","RH","windspeed_kmh-1","winddir",
            "wind_gust_10minute_kmh-1"]:
        DF_PAWS[numcolumn] = pandas.to_numeric(DF_PAWS[numcolumn],errors='coerce')
        # errors='coerce' turns bad strings into NaN
    for kmh,ms in zip(['windspeed_kmh-1','wind_gust_10minute_kmh-1'],
                       ['windspeed_ms-1','wind_gust_10minute_ms-1']):
        DF_PAWS[ms] = DF_PAWS[kmh] / 3.6 # add new ms-1 column to match AWS outputs
    
    # if station_name is not None:
    #     DF_PAWS = DF_PAWS.loc[DF_AWS['Station Name'].str.contains(str.upper(station_name))]
    
    if (lt0 is not None) or (lt1 is not None):
        DF_PAWS = DF_subset_time(DF_PAWS,dt0=lt0,dt1=lt1,timename='localtime')
    
    return DF_PAWS

def read_AWS(extent=None,station_name=None, lt0=None, lt1=None):
    """
    Read all AWS data
    can subset to extent: [W,E,S,N]
    can subset to single station
    can subset to date window [lt0, lt1]
    """
    ddir="../data/AWS/"
    AWS_allfiles=glob(ddir+"HM01X_Data*")
    
    AWS_drop_columns=['hm','Station Number',
                      "Precipitation in last 10 minutes in mm",
                      "Quality of precipitation in last 10 minutes",
                      "Precipitation since 9am local time in mm",
                      "Quality of precipitation since 9am local time",
                      "Quality of air temperature",
                      "Quality of Wet bulb temperature",
                      "Quality of relative humidity",
                      "Vapour pressure in hPa",
                      "Quality of vapour pressure",
                      "Saturated vapour pressure in hPa",
                      "Quality of saturated vapour pressure",
                      "Wind speed quality",
                      "Wind direction quality",
                      "Quality of speed of maximum windgust in last 10 minutes",
                      "Quality of mean sea level pressure",
                      "Quality of station level pressure",
                      "QNH pressure in hPa",
                      "Quality of QNH pressure",
                      "AWS Flag","Error Flag","#"
                      ]
    
    
    li = []
    for filename in AWS_allfiles:
        df = pandas.read_csv(filename, index_col=None, header=0)
        li.append(df)
    DF_AWS = pandas.concat(li, axis=0, ignore_index=True)
    #DF_AWS = pandas.read_csv(AWS_allfiles)
    DF_AWS.rename(columns=AWS_rename_columns,inplace = True)
    DF_AWS.drop(AWS_drop_columns, 
                axis=1, 
                inplace=True)
    
    # convert the 'Date' column to datetime format
    for dtcolumn in ["localtime","utc"]:
        DF_AWS[dtcolumn] = pandas.to_datetime(DF_AWS[dtcolumn],dayfirst=True)

    # Ensure numerical columns are numerical
    for numcolumn in ["temperature","temperature_bulb","temperature_dew","RH","windspeed_ms-1","winddir",
            "wind_gust_10minute_ms-1","mslp_hPa","pressure_hPa"]:
        DF_AWS[numcolumn] = pandas.to_numeric(DF_AWS[numcolumn],errors='coerce')
        # errors='coerce' turns bad strings into NaN

    
    ## Select by station name
    #df.loc[(df['column_name'] >= A) & (df['column_name'] <= B)]
    #DF_Parndana = DF_AWS.loc[DF_AWS['Station Name'].str.contains("PARNDANA")]
    #lat,lon = DF_Parndana.latitude.values[0], DF_Parndana.longitude.values[0]
    
    if station_name is not None:
        DF_AWS = DF_AWS.loc[DF_AWS['Station Name'].str.contains(str.upper(station_name))]
    
    if extent is not None:
        lon0,lon1,lat0,lat1 = extent
        DF_AWS = DF_AWS.loc[(DF_AWS["latitude"] > lat0) & (DF_AWS["latitude"] < lat1)]
        DF_AWS = DF_AWS.loc[(DF_AWS["longitude"] > lon0) & (DF_AWS["longitude"] < lon1)]
    
    if (lt0 is not None) or (lt1 is not None):
        DF_AWS = DF_subset_time(DF_AWS,dt0=lt0,dt1=lt1,timename='localtime')
    
    return DF_AWS
    

def AWS_compare_10m(mr, station_name, buffer_hours=1,
                    no_model=False,
                    no_aws=False,
                    latlon=None,):
    """
    """
    
    ## READ AWS
    if "PAWS_Belowra" in station_name:
        DF_AWS = read_PAWS_belowra()
    elif not no_aws:
        DF_AWS = read_AWS(station_name=station_name)
    
    if no_aws:
        assert latlon is not None, "latlon needs to be entered if running with no aws file"
        lat,lon = latlon
    else:
        lat = DF_AWS.latitude.values[0]
        lon = DF_AWS.longitude.values[0]
        latlon=[lat,lon]
    print("INFO: compare AWS running for lat,lon:",lat,lon)

    if not no_model:
        ## Read fire series at that spot
        DS_fire = read_fire_time_series(mr,latlon=[lat,lon],)
        lt_fire = DS_fire.localtime.values
        firepower = DS_fire.firepower.values # [t] GWatts
        fire_u10 = DS_fire['u_10m'].values.squeeze()
        fire_v10 = DS_fire['v_10m'].values.squeeze()
        fire_s10 = np.hypot(fire_u10,fire_v10)
        fire_wdir10 = DS_fire['wdir_10m'].values.squeeze()
        fire_t_surf = DS_fire['surface_temperature'].values.squeeze() # [t] Celcius
        fire_RH_surf = DS_fire['surface_RH'].values.squeeze() # [t] %
        
        lt0 = lt_fire[0]-pandas.Timedelta(buffer_hours,'h')
        lt1 = lt_fire[-1]+pandas.Timedelta(buffer_hours,'h')
    else:
        avail_times=fio.hours_available(mr)
        avail_lt = utils.local_time_from_time_lats_lons(avail_times,[lat],[lon])
        lt0 = avail_lt[0]
        lt1 = avail_lt[-1]
        
    if not no_aws:
        ## subset to model output +- buffer time
        DF_AWS = DF_subset_time(DF_AWS, dt0=lt0,dt1=lt1,timename='localtime')
        AWS_s = DF_AWS['windspeed_ms-1'].values
        AWS_gusts = DF_AWS['wind_gust_10minute_ms-1'].values
        AWS_wdir = DF_AWS['winddir'].values
        AWS_RH = DF_AWS['RH'].values
        AWS_T = DF_AWS['temperature'].values # Celcius
        lt_AWS = DF_AWS.localtime.values

    ## Plot stuff
    mc='darkgreen' # model colour
    dc='k' # data colour
    fig,axes = plt.subplots(nrows=4+(not no_model),ncols=1,sharex=True,sharey=False)
    # first plot shows wind speed and fire power
    plt.sca(axes[0])
    if not no_model:
        plt.plot_date(lt_fire,fire_s10, color=mc, fmt='-', 
                      label='model (10 metre)')
    if not no_aws:
        plt.plot_date(lt_AWS, AWS_s, color=dc,fmt='-', 
                      label='AWS')
        plt.plot_date(lt_AWS, AWS_gusts, color=dc,fmt='x', alpha=0.7,
                      label='AWS gusts (10 minute)')

    plt.xticks([],[])
    plt.ylabel("wind speed (m/s)")

    
    # wind direction
    plt.sca(axes[1])
    if not no_model:
        plt.plot_date(lt_fire,fire_wdir10, color=mc, fmt='o', label='model (10m)')
    if not no_aws:
        plt.plot_date(lt_AWS, AWS_wdir, color=dc,fmt='o', label='AWS')
    # swap axis side
    plt.ylabel("WDir (deg)")
    plt.ylim(0,360) # ylimits should be 0 to 360 for wdir
    axes[1].yaxis.set_label_position("right")
    axes[1].yaxis.tick_right()
    wdir_ticks=[90,180,270]
    plt.yticks(wdir_ticks,wdir_ticks)
    plt.xticks([],[])
    # other side axis for RH
    plt.sca(axes[2])
    if not no_model:
        plt.plot_date(lt_fire,fire_RH_surf, color=mc,fmt='--',
                      label='model (surface)')
    if not no_aws:
        plt.plot_date(lt_AWS, AWS_RH, color=dc,fmt='--', 
                      label='AWS')
    #plt.legend()
    plt.ylabel('RH (%)')
    rh_ticks = [25,50,75]
    plt.ylim(0,100) # ylimits should be 0 to 100 for RH
    plt.yticks(rh_ticks,rh_ticks)
    plt.xticks([],[])
    
    ## temperature, pressure
    plt.sca(axes[3])
    if not no_model:
        plt.plot_date(lt_fire, fire_t_surf, color=mc, fmt='-', label='model (surface)')
    if not no_aws:
        plt.plot_date(lt_AWS, AWS_T, color=dc,fmt='-', label='AWS')
    plt.ylabel("T (C)")
    axes[3].yaxis.set_label_position("right")
    axes[3].yaxis.tick_right()
    
    if not no_model:
        plt.xticks([],[])
        plt.sca(axes[4])
        plt.plot_date(lt_fire,firepower, color='r',fmt='-',label='firepower')
        plt.ylabel('FP (GW)',color='r')

        # Add PFT
        pft_ts=read_PFT_timeseries(mr, latlon=[lat,lon])
        pft=pft_ts.extract("PFT")[0]
        utc = utils.dates_from_iris(pft)
        ltoffset = utils.local_time_offset_from_lats_lons([lat],[lon])
        lt_pft = np.array([utci+timedelta(hours=ltoffset) for utci in utc])
        plt.plot_date(lt_pft,pft.data, color='m',fmt='-',label="PFT")
        # Limit y axis to 2x maximum firepower
        plt.legend()

    
    # fix date formatting
    plt.gcf().autofmt_xdate()
    plt.xlabel('local time')
    plt.suptitle(mr+" vs "+station_name)
    plt.subplots_adjust(hspace=0.0)
    fio.save_fig(mr, 
            plot_name="AWS_vs_station", 
            plot_time=station_name,
            plt=plt,
            )
    
def AWS_sites(mr=None, WESN=None):
    """
        return name,lat,lon for all sites
        optionally subset to those within a particular model run inner nest
        optionally subset to within a [WESN] boundary
    """
    aws=read_AWS()
    station_names= [str.strip(name) for name in aws["Station Name"].unique()]
    
    if (mr is not None) and (WESN is None):
        topog = fio.model_run_topography(mr)
        WESN = topog.longitude.values[0],topog.longitude.values[-1],topog.latitude.values[0],topog.latitude.values[-1],
    
    #print(station_names)
    namelatlon = []
    for station_name in station_names:
        df = aws.loc[aws['Station Name'].str.contains(str.upper(station_name))]
        lat,lon = df.latitude.values[0],df.longitude.values[0]
        if WESN is not None:
            if (WESN[0]<lon) and (WESN[1]>lon) and (WESN[2]<lat) and (WESN[3]>lat):
                namelatlon.append([station_name,lat,lon])
            else:
                continue
        else:
            namelatlon.append([station_name,lat,lon])
        
    return namelatlon
        

if __name__ == '__main__':
    
    
    #DF=read_PAWS_belowra()
    #print(DF)
    
    if False:
        fireseries("KI_run1_exploratory",)
    
    if False:
        #for mr in ["yanchep_run3", "yanchep_run4","stanthorpe_run1","stanthorpe_run2"]:
        #for mr in ["corryong_run4", "corryong_run4_fix","green_valley_run2","gren_valley_run2_fix"]:
        for mr in ["green_valley_run2_fix"]:
            fireseries(mr)

    if False:
        for mr in ['KI_run1','KI_run2','badja_run3','badja_run1','badja_run2',]:
            #read_fire_time_series(mr, force_recreate=True)
            fireseries(mr)

    if False:
        for mr in ['badja_run3','badja_run1','badja_run2']:
            AWS_compare_10m(mr, "Belowra",no_aws=True,latlon=constants.latlons['Belowra'])
            AWS_compare_10m(mr,'PAWS_Belowra',no_model=True)
            AWS_compare_10m(mr,'Moruya')
            

    if True:
        for mr in ['KI_run2','KI_run1',]:
            for site in ['CAPE BORDA','KINGSCOTE AERO','PARNDANA CFS AWS']:
                AWS_compare_10m(mr,site)
                #fireseries(mr)
            AWS_compare_10m(mr,'CAPE WILLOUGHBY',no_model=True)

    

