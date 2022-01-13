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

from utilities.utils import extra_DataArrays, vorticity

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

    # If we're looking at a xr.Dataset, we need to use sel
    if isinstance(DS,xr.Dataset):
        if hasattr(DS,"longitude_0"): # atmos DS has some fields on staggered grid
            DS_cut = DS.sel(
                latitude = slice(lat0,lat1),
                longitude = slice(lon0,lon1),
                latitude_0 = slice(lat0,lat1),
                longitude_0 = slice(lon0,lon1),
                )
        elif hasattr(DS,"lon"): # maybe fire DS
            DS_cut = DS.sel(
                lat = slice(lat0,lat1),
                lon = slice(lon0,lon1),
                )
        else:
            DS_cut = DS.sel(
                latitude = slice(lat0,lat1),
                longitude = slice(lon0,lon1),
                )

        return DS_cut

    
    # make mask for lats/lons (or latitude/longitude)
    if hasattr(DS,"longitude"):
        mask_lat = (DS.latitude >= lat0) & (DS.latitude <= lat1)
        mask_lon = (DS.longitude >= lon0) & (DS.longitude <= lon1)
    elif hasattr(DS,"lon"):
        mask_lat = (DS.lat >= lat0) & (DS.lat <= lat1)
        mask_lon = (DS.lon >= lon0) & (DS.lon <= lon1)
    else:
        print(DS)
        print("ERROR: COULDN'T FIND dimensions (lon)gitude, or (lat)itude")
    
    # Handle staggered dims also
    if hasattr(DS,"longitude_0"):
        mask_lon = mask_lon & (DS.longitude_0 >= lon0) & (DS.longitude_0 <= lon1)
    if hasattr(DS,"latitude_0"):
        mask_lat = mask_lat & (DS.latitude_0 >= lat0) & (DS.latitude_0 <= lat1)
    
    #Finally, it is just a matter of using the where() method and specifying drop=True as an argument.
    print("DEBUG: DS before spatial subset:")
    print(DS)
    cropped_ds = DS.where(mask_lon & mask_lat, drop=True)

    print("DEBUG: DS after spatial subset:")
    print(cropped_ds)
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

def read_model_run_hour(mr, extent=None, hour=0):
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
    
    if extent is not None:
        DS = extract_extent(DS,extent)
    
    return DS

def read_extra_data(mr, 
                    hour=0,
                    force_recreate=False,
                    topind=90
                    ):
    """
    Read or Create/Save set of data extra to the model output
    Includes: potential temp, destaggered winds, z_th, vorticity, OW, OWZ, OWN, updraft helicity, rotation
    
    ARGUMENTS:
        mr: model run name
        hour: integer - which hour to read/create
        force_recreate: bool - create from atmos output even if already done
    RETURNS:
        xarray DataSet
    """
    
    hour_dt = hours_available(mr)[hour]
    hour_str = hour_dt.strftime("%Y%m%d%H")
    
    fname = "../data/"+mr+"/extra_data/"+hour_str+".nc"
    if os.path.isfile(fname) and not force_recreate:
        # Read the file, return DS
        print("INFO: Reading already created extra_data:",fname)
        return xr.open_dataset(fname)
    #else:
    
    # make the file, save the file
    atmos=read_model_run_hour(mr,hour=hour)
    make_folder(fname)
    
    DA_dict={}
    # add extra data
    extra_DataArrays(atmos,add_winds=True,add_theta=True,add_z=True)
    DA_dict['potential_temperature']= atmos['potential_temperature']
    DA_dict['u'] = atmos['u']
    DA_dict['v'] = atmos['v']
    DA_dict['s'] = atmos['s']
    DA_dict['wind_direction'] = atmos['wind_direction']
    DA_dict["z_th"] = atmos["z_th"]
    # add vorticity
    u=atmos['u'].data
    v=atmos['v'].data
    lats=atmos['u'].latitude
    lons=atmos['u'].longitude
    zeta,OW,OWN,OWZ = vorticity(u,v,lats,lons)
    DA_zeta = xr.DataArray(data=zeta,
                           coords=atmos['u'].coords,
                           dims=atmos['u'].dims,
                           name="vorticity",
                           attrs={"units":"1/s",
                                  "desc":"Vorticity = zeta = v_x - u_y",
                                  },
                          )
    DA_OW = xr.DataArray(data=OW,
                         coords=atmos['u'].coords,
                         dims=atmos['u'].dims,
                         name="OW",
                         attrs={"units":"1/s",
                                "desc":"Shear def = F = v_x + U_y, Stretch def = E = u_x - v_y, OW = zeta^2 - (E^2 + F^2)",
                                },
                          )
    
    DA_dict['vorticity']=DA_zeta
    DA_dict['OW'] = DA_OW
    # ADD local time
    #    utc=Temp.time.values
    #    localtime = utils.local_time_from_time_lats_lons(utc,[lat],[lon])
    #    DS_plus_lt=DS.assign_coords(localtime=("time",localtime))
    # 
    # Add rotation
    

    print("INFO: SAVING NETCDF:",fname)
    DS = xr.Dataset(DA_dict)
    DS.to_netcdf(fname)
    print("INFO: SAVED NETCDF:",fname)
    

def read_model_run_fire(mr, 
                        extent=None,
                        #dtimes=None,
                        ):
    """
    """
    fdir=DATADIR+mr+"/fire/"
    firepaths=glob(fdir+"*00Z.nc")
    DS = xr.open_mfdataset(firepaths,compat=COMPAT)
    
    if extent is not None:
        DS = extract_extent(DS,extent)
    
    #subselect based on datetime list input
    # does not quite work...
    #    if dtimes is not None:
    #        # convert to numpy.datetime64?
    #        for DA in DS:
    #            if 'time' in DS[DA].dims:
    #                print("DEBUG: BEFORE:")
    #                print(DS[DA].loc[dict(time=dtimes)])
    #                DS.update({DA:DS[DA].loc[dict(time=dtimes)]})
    #                print("DEBUG: AFTER:")
    #                print(DS[DA])
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
        #dstamp=(plot_time.astype(datetime)).strftime('%dT%H%M')
        dstamp = pandas.to_datetime(str(plot_time)).strftime('%dT%H%M')
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
