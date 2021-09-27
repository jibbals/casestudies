#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 10:25:13 2021
    PFT calculations for a model run
    
@author: jesse
"""

import iris
import numpy as np
import os
import xarray
import pandas
from datetime import datetime, timedelta
from timeit import default_timer as timer
from utilities import fio_iris as fio
from utilities import utils, constants
from timeseries_stuff import read_fire_time_series





def compare_PFT_to_firepower(mr, latlon, firespeed=False, name=None, FPrun=None):
    """
        Timesries of PFT and firepower
    """
    
    lat,lon = latlon

    ## Read fire series at that spot

    if FPrun is None:
        DS_fire = read_fire_time_series(mr,latlon=[lat,lon],)
    else:
        DS_fire = read_fire_time_series(FPrun,latlon=[lat,lon],)

    lt_fire = DS_fire.localtime.values
    firepower = DS_fire.firepower.values # [t] GWatts
    fire_u10 = DS_fire['u_10m'].values.squeeze()
    fire_v10 = DS_fire['v_10m'].values.squeeze()
    fire_s10 = np.hypot(fire_u10,fire_v10)
    fire_wdir10 = DS_fire['wdir_10m'].values.squeeze()

    # Create/read pft timeseries
    pft_ts=fio.read_PFT_timeseries(mr, latlon=latlon)
    # pull out data, sample plot
    pft=pft_ts.extract("PFT")[0]
    #utc = pft.coord('time').points
    utc = utils.dates_from_iris(pft)
    ltoffset = utils.local_time_offset_from_lats_lons([lat],[lon])
    lt_pft = np.array([utci+timedelta(hours=ltoffset) for utci in utc])
        
    buffer_hours=1 # make plots a bit wider
    lt0 = lt_fire[0]-pandas.Timedelta(buffer_hours,'h')
    lt1 = lt_fire[-1]+pandas.Timedelta(buffer_hours,'h')
        
    ## Plot stuff
    mc='darkgreen' # model colour
    
    # show firepower and PFT
    plt.plot_date(lt_fire,firepower, color='r',fmt='-',label='firepower')
    plt.plot_date(lt_pft,pft.data, color='m',fmt='-',label="PFT")
    plt.ylabel('Gigawatts',color='r')
    plt.legend()
    
    if firespeed:
        # other side axis for firepower
        plt.twinx()
        plt.plot_date(lt_fire,fire_s10, color=mc, fmt='-', 
                      label='model (10 metre)')
        plt.ylabel("wind speed (m/s)")

    if name is None:
        name = "%.2f,%.2f"%(lat,lon)
    # fix date formatting
    plt.gcf().autofmt_xdate()
    plt.xlabel('local time')
    plt.suptitle(mr+" at "+name)
    plot_name="%s_%s"%(mr,name)
    if FPrun is not None:
        plot_name = plot_name+"_FP_from_"+FPrun.split("_")[-1]
    fio.save_fig(mr, 
            plot_name="PFT_vs_firepower", # folder name
            plot_time= plot_name,# plot name (is normally time)
            plt=plt,
            )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    Borda_latlon=constants.latlons['Cape Borda']
    Parndana_latlon=constants.latlons['Parndana']
    Yowrie_latlon=constants.latlons['Yowrie']
    if True:
        # look at run3, run4, and run4 against firepower from run3
        compare_PFT_to_firepower("badja_run3", Yowrie_latlon, 
                firespeed=False, 
                name="Yowrie")
        compare_PFT_to_firepower("badja_run4", Yowrie_latlon, 
                firespeed=False, 
                name="Yowrie")
        compare_PFT_to_firepower("badja_run4", Yowrie_latlon, 
                firespeed=False, 
                name="Yowrie",
                FPrun="badja_run3",)


    if False:
        mr="KI_run2"
        compare_PFT_to_firepower(mr,Borda_latlon,firespeed=False,name="Cape Borda")
        compare_PFT_to_firepower(mr,Parndana_latlon,firespeed=False, name="Parndana")


    # badja firepower vs upwind pft
    if False:
        mr = "badja_run3"
        badja_latlon=[-36.0,149.4]
        compare_PFT_to_firepower(mr,badja_latlon,firespeed=False)
    
    # Create/read pft timeseries
    #pft_ts=fio.read_PFT_timeseries(mr, latlon=latlon)
    # pull out data, sample plot
    #pft=pft_ts.extract("PFT")[0]
    #time = pft.coord('time').points
    #plt.plot_date(time,pft.data)
    #plt.title(mr + " PFT at %.2f,%.2f"%(lat,lon))
    #plt.savefig("pft_test.png")
    #print("INFO: SAVED pft_test.png")
