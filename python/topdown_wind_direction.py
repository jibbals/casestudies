# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 21:11:21 2021

@author: jgreensl
"""

import matplotlib
matplotlib.use('Agg')

# plotting stuff
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, LinearLocator#, LogFormatter
#import matplotlib.patheffects as PathEffects
import numpy as np
import warnings

#from datetime import datetime,timedelta
from pandas import Timedelta
#from scipy import interpolate
#import cartopy.crs as ccrs

# local modules
from utilities import plotting, utils, constants, fio

###
## GLOBALS
###

##
## METHODS
###
def topdown_wind_plot(DS):
    """
    """
    print(':TODO:')

def topdown_winds(
        mr,
        hours=range(24),
        extent=None,
        levels=[0,1,3,5,10],
        subdir=None,
        ):
    """
    
    ARGUMENTS:
        mr: model run name
        hours: optional run for subset of model hours
        extent: subset extent
        levels: model levels to plot
        subdir: savefolder in case of specific extent
    """
    # Defaults
    if hours is None:
        hours=range(24)

    # read fire model output
    DS_fire=fio.read_model_run_fire(mr)
    
    for hour in hours:
        DS=fio.read_model_run_hour(mr,hour=hour)
        if extent is not None:
            DS = fio.extract_extent(DS,extent)
        lats=DS.latitude.data
        lons=DS.longitude.data
        houroffset=utils.local_time_offset_from_lats_lons(lats,lons)
         
        times=DS.time.data # np.datetime64 array
        
        # loop over timesteps
        for ti,time_utc in enumerate(times):
            # create square figure
            plt.figure(figsize=[13,13])
            
            # add fire
            DS_fire_timeslice = DS_fire.loc[dict(time=time_utc)]
            #print("DEBUG: transect: fire timeslice:", DS_fire_timeslice)
            FF = DS_fire_timeslice['firefront'].data
            plotting.map_fire(FF.T,lats,lons)
            
            # get local time
            time_lt = utils.local_time_from_time_lats_lons(time_utc,lats,lons)
            time_str=time_lt.strftime("%Y%m%d %H%M")+"(UTC+%.2f)"%houroffset
            
            DS_timeslice=DS.loc[dict(time=time_utc)]
            

            plt.suptitle(time_str+ "wind transect")
            fio.save_fig(mr,"topdown_wind_dirs",time_utc,plt,)
    

if __name__ == '__main__':
    mr="KI_run1_exploratory"
    topdown_winds(mr)