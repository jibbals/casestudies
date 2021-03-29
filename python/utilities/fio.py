# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 10:04:17 2021
    Simplified FIO using xarray on casestudies output
    Eventually fio_iris.py will be obselete
@author: jgreensl
"""

import xarray as xr # uses dask to lazy read, can handle big data files
import numpy as np
import timeit # for timing stuff
from datetime import datetime, timedelta
import pandas # for csv reading (AWS)

from glob import glob
import os

###
## GLOBALS
###
DATADIR="../data/"
## override will handle missmatched attributes using info from first file containing attribute
COMPAT='override'
## not sure what minimal does but it works on the work laptop
if xr.__version__<"0.12.4":
    COMPAT='minimal'

# This script is only run from parent folder, so this relative path should work
#from utilities import utils, constants

def atmos_paths(mr, 
                hours=None, 
                suffix=None):
    """
    return list of file paths with data from model run <run_name>.
    Optionally subset to a list of specific hours

    INPUTS:
        hours (optional): list of datetimes
        suffix (optional): only return files with *suffix* in the name 
    """

    ddir = DATADIR + run_name + '/atmos/'
    # return list of fpaths
    fpaths=[]
    
    ## no suffix? set it to empty
    if suffix is None: 
        suffix=""

    ## No ftimes? return all available data paths
    if hours is None:
        fpaths=glob(ddir+"*%s.nc"%suffix)
            
    else: 
        # make sure it's iterable
        if not hasattr(hours,'__iter__'):
            hours = [hours]
        hours = np.array(hours)
    
        for hour in hours:
            dstamp=hour.strftime('%Y%m%d%H')
            fpaths0 = glob(ddir+"*%s*%s.nc"%(dstamp,suffix))
            fpaths.extend(fpaths0)
    if len(fpaths) < 1:
        print("WARNING: No files found (after any constraints applied) in %s "%ddir)
    fpaths.sort()
    return fpaths

def extract_extent(DS,WESN):
    """
    subset xarray DataSet to within [West,East,South,North] boundaries
    needs latitude/longitude dimensions
    ARGS:
        DS: xarray dataset (or DataArray?)
        WESN: [W,E,S,N] latlons of extent
    """
    lon0,lon1,lat0,lat1=WESN
    
    mask_lon = (DS.longitude >= lon0) & (DS.longitude <= lon1)
    mask_lat = (DS.latitude >= lat0) & (DS.latitude <= lat1)
    
    if hasattr(DS,"longitude_0"):
        mask_lon = mask_lon & (DS.longitude_0 >= lon0) & (DS.longitude_0 <= lon1)
    if hasattr(DS,"latitude_0"):
        mask_lat = mask_lat & (DS.latitude_0 >= lat0) & (DS.latitude_0 <= lat1)
        
    #Finally, it is just a matter of using the where() method and specifying drop=True as an argument.
    print("DEBUG: DS before spatial subset:")
    print(DS.head())
    cropped_ds = DS.where(mask_lon & mask_lat, drop=True)
    print("DEBUG: DS after spatial subset:")
    print(cropped_ds.head())
    return cropped_ds

def fire_path(mr, prefix):
    """
    return fire output for model run "mr" matching "prefix" such as firefront|10m_uwind|...
    """
    fdir = DATADIR+mr+'/fire/'
    
    paths=glob(fdir+prefix+'.'+'*')
    return paths

def fire_paths(mr,datestr="",affix=".nc"):
    """
    return sorted list of files within model run fire output directory
    ARGS:
        mr: model run
        datestr: files look like "...YYYYMMDDThhmmZ.nc" if you just want specifice datestring match add it here
        affix: ".nc"
    RETURNS: [ffpaths, fluxpaths, fspaths, u10paths, v10paths]
    """
    fdir = DATADIR+mr+'fire/'
    
    #print("DEBUG: looking for file:",fdir+'firefront.'+affix+'*')
    ffpaths=glob(fdir+'firefront.'+datestr+"*"+affix)
    fluxpaths=glob(fdir+'sensible_heat.'+datestr+"*"+affix)
    fspaths=glob(fdir+'fire_speed.'+datestr+"*"+affix)
    v10paths=glob(fdir+'10m_vwind.'+datestr+"*"+affix)
    u10paths=glob(fdir+'10m_uwind.'+datestr+"*"+affix)
    ffpaths.sort()
    fluxpaths.sort()
    fspaths.sort()
    v10paths.sort()
    u10paths.sort()
    return [ffpaths, fluxpaths, fspaths, u10paths, v10paths]

def make_folder(pname):
    folder = '/'.join(pname.split('/')[:-1]) + '/'
    if not os.path.exists(folder):
        print("INFO: Creating folder:",folder)
        os.makedirs(folder)

def model_run_datetimes(mr, hours=None):
    """
    return np array of datetimes available to model run mr
    """
    allhours=np.array(hours_available(mr))
    if hours is not None:
        # make sure it's iterable
        if not hasattr(hours,'__iter__'):
            hours = [hours]
        allhours=allhours[hours]
    return allhours

def model_run_topography(mr):
    '''
    Read first model output file with "surface_altitude"
    return DataArray surface_altitude
    '''

    ddir = DATADIR+mr+'/atmos/'
    files= glob(ddir+"*_mdl_th1.nc")
    files.sort()
    
    topog_varname='surface_altitude'
    assert len(files) > 0, "NO FILES FOUND IN "+ddir
    
    DS = xr.open_dataset(files[0])
    topog=DS[topog_varname] 
    return topog

def hours_available(mr):
    """
    Return list of hours where model output exists
    """
    ddir = '../data/'+mr+'/atmos/'
    # glob _slv files and build dates list from filenames
    fpattern=ddir+'*mdl_th1.nc'
    filepaths=glob(fpattern)
    filepaths.sort()
    # just last 28 chars "umnsaa_....nc"
    files = [f[-28:] for f in filepaths]
    datetimes=[datetime.strptime(f, "umnsaa_%Y%m%d%H_mdl_th1.nc") for f in files]
    return np.array(datetimes)

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
        print("INFO:",atmosdir," is not a directory")
        atmosdir=mr+"/atmos/"
        print("INFO: reading from %s"%atmosdir)
    
    allfiles=glob(atmosdir+"*.nc")
    allfiles.sort()
    
    # 4 files per model hour
    hourfiles = allfiles[hour*4:hour*4+4]
    print("INFO: will read files:")
    print("    :",hourfiles)
    
    DS = xr.open_mfdataset(hourfiles,compat=COMPAT)
    #lats = DS['latitude']
    #lons = DS['longitude']
    #topog=DS["surface_altitude"]
    #print(DS.head())
    return DS
    
def read_model_run_fire(mr):
    """
    """
    fdir=DATADIR+mr+"/fire/"
    firepaths=glob(fdir+"*.nc")
    DS = xr.open_mfdataset(firepaths,compat=COMPAT)
    #print(DS)
    #    if datetimes is not None:
    #        dt64=datetimes
    #        if isinstance(datetimes[0],datetime):
    #            dt64 = [np.datetime64(dt) for dt in datetimes]
    #        print(dt64)
    #        print(DS.time)
    #        DS = DS.loc[dict(time=dt64)]
        
    return(DS)
    
def save_fig_to_path(pname,plt, **savefigargs):
    '''
    Create dir if necessary
    Save figure
    example: save_fig('my/path/plot.png',plt)
    INPUTS:
        pname = path/to/plotname.png
        plt = matplotlib.pyplot instance

    '''
    # Defaults:
    if 'dpi' not in savefigargs:
        savefigargs['dpi']=200
    if 'transparent' not in savefigargs:
        savefigargs['transparent']=False
        
    make_folder(pname)
    print ("INFO: Saving figure:",pname)
    plt.savefig(pname, **savefigargs)
    plt.close()

def standard_fig_name(model_run, plot_name, plot_time,
                      subdir=None, 
                      ext='.png'):
    if isinstance(plot_time,datetime):
        dstamp=plot_time.strftime('%dT%H%M')
    elif isinstance(plot_time,np.datetime64):
        dstamp=(plot_time.astype(datetime)).strftime('%dT%H%M')
    else:
        dstamp=plot_time

    if subdir is None:
        subdir = ""
    elif subdir[0] not in "/\\":
        subdir = "/"+subdir

    path='../figures/%s/%s%s/%s'%(model_run, plot_name, subdir, dstamp) + ext
    return path

def save_fig(mr, plot_name, plot_time, plt,
             subdir=None, 
             ext='.png', 
             **savefigargs):
    """
    create figurename as figures/plot_name/model_run/extent_name/ddTHHMM.png

    INPUTS:
        model_run, plot_name : strings
        plot_time : can be datetime64 or string
            if datetime, pname is set to ddTHHMM
        plt : is instance of matplotlib.pyplot
        extent_name : name of extent used making plot
    """
    path = standard_fig_name(mr, plot_name, plot_time, 
                             subdir, ext, **savefigargs)

    save_fig_to_path(path, plt, **savefigargs)