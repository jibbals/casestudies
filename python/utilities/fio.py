#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 13:52:09 2019

  READING AND WRITING NETCDF AND GRIB(?) FILES

@author: jesse
"""

###
## IMPORTS
###

import iris
from iris.experimental.equalise_cubes import equalise_attributes
import xarray as xr # better fio
import numpy as np
import timeit # for timing stuff
import warnings
from datetime import datetime, timedelta
import pandas # for csv reading (AWS)

from glob import glob
import os

# This script is only run from parent folder, so this relative path should work
from utilities import utils, constants

###
## GLOBALS
###
__VERBOSE__=True
__DATADIR__="../data/"

## Info that is the same for most runs
#
sim_info={
    'badja':{
        'topog':'umnsaa_2019123003_slv.nc',
        'filedates':np.array([datetime(2019,12,30,3) + timedelta(hours=x) for x in range(24)]), # in UTC
        'UTC_offset':11, # UTC + 11 hours AEDT (daylight savings)
        },
    'corryong':{
        'topog':'umnsaa_2019123015_slv.nc',
        'filedates':np.array([datetime(2019,12,30,15) + timedelta(hours=x) for x in range(24)]), # in UTC
        'UTC_offset':11, # UTC + 11 hours AEDT (daylight savings)
        },
    'KI':{
        'topog':'umnsaa_2020010215_slv.nc',
        'filedates':np.array([datetime(2020,1,2,15) + timedelta(hours=x) for x in range(24)]),
        'UTC_offset':10.5, # utc + 10.5 in summer
        }
    }


## Where are model outputs located, and run specific info
run_info = {
    'badja_run1':{
        'dir':__DATADIR__+'badja_run1/',
        'WESN':[149.26,150.075,-36.52,-35.914],
        'desc':{ # any description stuff in here
            "fuel":"constant = 2 everywhere",
            },
        },
    'corryong_run1':{
        'dir':__DATADIR__+'corryong_run1/',
        'WESN':[147.0196,148.6296,-37.1244,-35.5144],
        'desc':{ # any description stuff in here
            "fuel":"constant = 2 everywhere",
            },
        },
    'KI_run1':{ # run1 uses approximately correct fire polygon start
        'dir':__DATADIR__+'KI_run1/',
        'WESN':[136.1922,137.8036,-36.1774,-35.5054],
        'desc':{ # any description stuff in here
            "fuel":"constant = 2 everywhere",
            },
        },
    'KI_run0':{ # test run to get working, fire polygon not accurate
        'dir':__DATADIR__+'KI_run0/',
        'WESN':[136.1922,137.8036,-36.1774,-35.5054],
        'desc':"Test run to get suite working",
        },
    }

# populate run_info with sim_info
for run in run_info.keys():
    loc=run.split('_')[0]
    for k,v in sim_info[loc].items():
        # if sim location info not in run_info[run] dict, add it
        if k not in run_info[run].keys():
            run_info[run][k] = v

###
## METHODS
###

def _constraints_from_extent_(extent, constraints=None, tol = 0.0001):
    """
        Return iris constraints based on [WESN] lon,lat limits (with a tolerance)
        this only looks at the cell centres, so data with or without cell bounds compares equally
        additional constraints can be ampersanded to the lat lon constraints
    """
    West,East,South,North = extent
    # NEED TO COMPARE TO CELL MIDPOINT SO THAT BEHAVIOUR IS CONSTANT WHETHER OR NOT THERE ARE BOUNDS
    constr_lons = iris.Constraint(longitude = lambda cell: West-tol <= cell.point <= East+tol)
    constr_lats = iris.Constraint(latitude = lambda cell: South-tol <= cell.point <= North+tol)
    if constraints is not None:
        constraints = constraints & constr_lats & constr_lons
    else:
        constraints = constr_lats & constr_lons
    return constraints

def create_wind_profile_csv(model_run, name, latlon):
    """ TODO: Update for AWS comparison data creation
    """
    # just want tiny area to interpolate within
    extent = [latlon[0]-.01, latlon[1]-.01, latlon[0]+.01, latlon[1]+.01] 
    
    ## add_flags no longer used
    cubes = read_model_run(model_run,fdtime=None,
                           extent=extent,
                           add_winds=True, add_z=True, add_topog=True,
                           add_RH=True)
    
    x,y,wdir = cubes.extract(['u','v','wind_direction'])
    z, topog = cubes.extract(['z_th','surface_altitude'])
    T, RH = cubes.extract(['air_temperature','relative_humidity'])
    
    # extract single lat lon
    cubes1 = iris.cube.CubeList([x,y,wdir,z,topog, T, RH])
    lat,lon = -32.84,115.93
    for i in range(len(cubes1)):
        cubes1[i] = cubes1[i].interpolate([('longitude',lon), ('latitude',lat)],
                                          iris.analysis.Linear())
    
    x1,y1,wdir1,z1,topog1,T1,RH1 = cubes1
    Tg = T1[:,0].data
    RHg = RH1[:,0].data
    wdg = wdir1[:,0].data
    wsg = np.sqrt(x1[:,0].data**2+y1[:,0].data**2)
    
    # height above ground level
    agl = z1.data - topog1.data
    agl_coord = iris.coords.AuxCoord(agl, var_name='height_agl', units='m')
    agl_coord.guess_bounds()
    
    # interpolate to our heights
    heights = np.arange(500,2001,500)
    cubes2 = iris.cube.CubeList([x1,y1,wdir1])
    ## interpolate to 500,1000,1500,2000m above ground level
    for i in range(len(cubes2)):
        # add agl coord
        cubes2[i].add_aux_coord(agl_coord,1) 
        cubes2[i] = cubes2[i].interpolate([('height_agl',heights)], iris.analysis.Linear())
    
    ### Create csv with gridded horizontal wind speed and direction
    ## 
    ## Sample headers:
    headers=['Time',
             'Temperature',
             'Relative humidity',
             'Wind direction ground',
             'Wind direction 500m',
             'Wind direction 1000m',
             'Wind direction 1500m',
             'Wind direction 2000m',
             'Wind speed ground',
             'Wind speed 500m',
             'Wind speed 1000m',
             'Wind speed 1500m',
             'Wind speed 2000m']
    # time format (UTC)
    times = utils.dates_from_iris(cubes[0],remove_seconds=True)
    tstrings = [dt.strftime("%Y-%m-%dT%H:%M:%S")for dt in times]
    
    import csv
    [u2,v2,wd2c] = cubes2
    ws2 = np.sqrt(u2.data**2+v2.data**2)
    wd2 = wd2c.data
    with open('outputs/waroona_winds.csv',mode='w',newline='') as winds_file:
        fwriter=csv.writer(winds_file, delimiter=',')
        fwriter.writerow(headers)
        for i in range(len(times)):
            fwriter.writerow([tstrings[i],Tg[i]-273.15,RHg[i], 
                              wdg[i], wd2[i,0], wd2[i,1], wd2[i,2], wd2[i,3],
                              wsg[i], ws2[i,0], ws2[i,1], ws2[i,2], ws2[i,3]])
    print("INFO: SAVED FILE outputs/waroona_winds.csv")


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

def read_nc_iris(fpath, constraints=None, keepvars=None, HSkip=None):
    '''
    Read netcdf file using iris, returning cubeslist
    actual data is not read until you call cube.data
    constraints can be applied, or variable names, or both
    '''

    print("INFO: Reading(iris) ",fpath)
    # First read all cubes in file using constraints
    if constraints is not None:
        cubes = iris.load(fpath, constraints)
    else:
        cubes = iris.load(fpath)

    # If just some variables are wanted, pull them out
    if keepvars is not None:
        cubes = cubes.extract(keepvars)

    # Maybe we want to cut down on the horizontal resolution
    if HSkip is not None:
        if not (HSkip == False):
            # For each cube, apply ::HSkip to lon/lat dimension
            small_cubes = iris.cube.CubeList()
            for cube in cubes:
                if cube.ndim == 2:
                    mini = cube[::HSkip,::HSkip]
                elif cube.ndim == 3:
                    mini = cube[...,::HSkip,::HSkip]
                elif cube.ndim == 4:
                    mini = cube[...,...,::HSkip,::HSkip]
                else:
                    print("ERROR: HSKIP is missing the cube: ")
                    print(cube)
                    assert False, "shouldn't miss any cubes"
                small_cubes.append(mini)
            cubes = small_cubes
    if len(cubes) < 1:
        print("ERROR: NO CUBES RETURNED IN fio.read_nc_iris()")
        print("     :",constraints)
    return cubes

def fire_paths(model_run):
    fdir = run_info[model_run]['dir']+'fire/'
    affix='20' # if there is no affix, look for name.YYYYMMDDTHHmmZ.nc
    
    #print("DEBUG: looking for file:",fdir+'firefront.'+affix+'*')
    ffpaths=glob(fdir+'firefront.'+affix+'*')
    fluxpaths=glob(fdir+'sensible_heat.'+affix+'*')
    fspaths=glob(fdir+'fire_speed.'+affix+'*')
    v10paths=glob(fdir+'10m_vwind.'+affix+'*')
    u10paths=glob(fdir+'10m_uwind.'+affix+'*')
    ffpaths.sort()
    fluxpaths.sort()
    fspaths.sort()
    v10paths.sort()
    u10paths.sort()
    return [ffpaths, fluxpaths, fspaths, u10paths, v10paths]

def fire_path(model_run, filename):
    """
    """
    fdir = __DATADIR__+model_run+'/fire/'
    affix='20' # if there is no affix, look for name.YYYYMMDDTHHmmZ.nc
    
    #print("DEBUG: looking for file:",fdir+'firefront.'+affix+'*')
    paths=glob(fdir+filename+'.'+affix+'*')
    return paths

def read_fire(model_run,
              dtimes=None, 
              constraints=None, 
              extent=None,
              filenames='firefront',
              HSkip=None):
    '''
    Read fire output file
    output is transposed from [time,lon,lat] -> [time, lat, lon] to match model output
    ARGUMENTS:
        model_run: name of model run to read
        dtimes:list of datetimes to read (default is all)
        extent: [WESN]
        filename: string matching one of {'firefront', 'sensible_heat','fire_speed','10m_uwind','10m_vwind','fuel_burning','relative_humidity','water_vapor','surface_temp'}
    '''
    ## make sure filenames is list of strings
    if isinstance(filenames, str):
        filenames=[filenames]
    
    ## If no fire exists for model run, return [None,...]
    fdir = __DATADIR__+model_run+'/fire/'
    if not os.path.exists(fdir):
        print("ERROR: no such filepath:",fdir)
        # needs to be iterable to match cubelist return type (with wind counted twice) 
        return [None]*len(filenames)
    
    if extent is not None:
        constraints = _constraints_from_extent_(extent,constraints)

    # Build up cubelist based on which files you want to read
    cubelist = iris.cube.CubeList()

    for fname in filenames:
        paths = fire_path(model_run,fname)
        
        #flags = [firefront, sensibleheat, firespeed, wind, wind]
        units = {"firefront":None,
                "sensible_heat":'Watts/m2', 
                "fire_speed":'m/s', 
                "10m_vwind":'m/s', 
                "10m_uwind":'m/s',
                }
        unit = units[fname]
    
        #print("DEBUG: flag, paths, unit",flag,paths,unit)
        
        if len(paths) < 1:
            print("ERROR: missing files")
            print("     :", paths)
        cube, = read_nc_iris(paths, constraints=constraints, HSkip=HSkip)
        if unit is not None:
            cube.units=unit
        cubelist.append(cube)

    # Subset by time if argument exists
    if dtimes is not None:
        for i in range(len(cubelist)):
            cubelist[i] = subset_time_iris(cubelist[i], dtimes)
    
    # finally put latitude before longitude
    for cube in cubelist:
        
        nx = len(cube.coord('longitude').points)
        if len(cube.shape) == 2:
            if cube.shape[0] == nx:
                cube.transpose()
        if len(cube.shape) == 3:
            if cube.shape[1] == nx:
                # t, lon, lat -> t, lat, lon
                cube.transpose([0, 2, 1])
        
    return cubelist

def model_run_filepaths(run_name, 
        hours=None, suffix=None):
    """
    return list of file paths with data from model run <run_name>.
    Optionally subset to a list of specific hours

    INPUTS:
        hours (optional): list of datetimes
        suffix (optional): only return files with *suffix* in the name 
    """

    ddir = __DATADIR__ + run_name + '/atmos/'
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
    


def read_model_run(mr,
                   hours=None, extent=None, constraints=None, HSkip=None):
    '''
    Read output from particular model run into cubelist, generally concatenates
    along the time dimension.

    INPUTS:
        model_version: string (see model_outputs keys)
        hours: which hours to read, optional
            file output datetime[s] to be read, if None then read all files
        subdtimes: iterable of datetimes, optional
            after reading files, just keep these datetime slices
            None will return all datetimes
        extent: list, optional
            [West, East, South, North] lon,lon,lat,lat list used to spatially
            subset the data (if desired)
        HSkip: integer, optional
            slice horizontal dimension [::HSkip] to reduce resolution

    RETURNS:
        iris.cube.CubeList with standardised dimensions [time, level, latitude, longitude]
    
    0: air_pressure / (Pa)                 (time: 6; model_level_number: 140; latitude: 14; longitude: 14)
    1: air_pressure_at_sea_level / (Pa)    (time: 6; latitude: 14; longitude: 14)
    2: air_pressure_rho / (Pa)             (time: 6; model_level_number: 140; latitude: 14; longitude: 14)
    3: air_temperature / (K)               (time: 6; model_level_number: 140; latitude: 14; longitude: 14)
    4: cld_ice / (kg kg-1)                 (time: 6; model_level_number: 140; latitude: 14; longitude: 14)
    5: cld_water / (kg kg-1)               (time: 6; model_level_number: 140; latitude: 14; longitude: 14)
    6: specific_humidity / (kg kg-1)       (time: 6; model_level_number: 140; latitude: 14; longitude: 14)
    7: surface_air_pressure / (Pa)         (time: 6; latitude: 14; longitude: 14)
    8: surface_temperature / (K)           (time: 6; latitude: 14; longitude: 14)
    9: upward_air_velocity / (m s-1)       (time: 6; model_level_number: 140; latitude: 14; longitude: 14)
    10: x_wind / (m s-1)                    (time: 6; model_level_number: 140; latitude: 14; longitude: 14)
    11: y_wind / (m s-1)                    (time: 6; model_level_number: 140; latitude: 14; longitude: 14)
    '''

    ## make sure we have model run data
    #assert model_version in run_info.keys(), "%s not yet supported by 'read_model_run'"%model_version
    ddir = __DATADIR__ + mr + '/atmos/'
    
    timelesscubes=[]
    allcubes=None

    ## No ftimes? set to all ftimes
    if hours is None:
        #hours = run_info[mr]['filedates']
        hours=hours_available(mr)
        if len(hours)<1:
            print("ERROR: No files in "+ddir)
            assert False, "NO FILES FOR "+mr
            
    # make sure it's iterable
    if not hasattr(hours,'__iter__'):
        hours = [hours]
    hours = np.array(hours)

    ## First read the basics, before combining along time dim
    for dtime in hours:
        slv,ro1,th1,th2 = read_standard_run(mr, dtime,
                                            HSkip=HSkip,
                                            extent=extent,
                                            constraints=constraints)

        # Remove specific humidity from slv, since we get the whole array from th1
        if len(slv.extract('specific_humidity')) > 0:
            slv.remove(slv.extract('specific_humidity')[0])

        if allcubes is None:
            allcubes=slv
        else:
            allcubes.extend(slv)
        # model output on rho levels and theta levels are slightly different
        if len(ro1.extract('air_pressure')) > 0:
            # rename the rho levels air pressure (or else it's repeated)
            ro1.extract('air_pressure')[0].rename('air_pressure_rho')

        allcubes.extend(ro1)
        allcubes.extend(th1)
        allcubes.extend(th2)


    ## Now because we may be reading multiple files 
    ## we Concatenate along time dimension
    ## First need to unify time dimension:
    iris.util.unify_time_units(allcubes)
    ## Also need to equalise the attributes list
    # I think this just deletes attributes which are not the same between matching cubes..
    equalise_attributes(allcubes)
    ## Join along the time dimension
    allcubes = allcubes.concatenate()
    ## subset to our subdtimes
    
    # surface stuff may have no time dim...
    for varname in ['surface_air_pressure','surface_temperature']:
        saps = allcubes.extract(varname)
        if len(saps) > 1:
            sap = saps.merge_cube()
            for cube in saps:
                allcubes.remove(cube)
            tube = allcubes[0]
            time = tube.coord('time')
            # add time steps for easyness
            sap0 = sap
            if tube.shape[0] != sap.shape[0]:
                sap0 = sap.interpolate([('time',time.points)],
                                        iris.analysis.Linear())
            # add single cube with time dim
            allcubes.append(sap0)
            
    ## NOW add any timeless cubes
    allcubes.extend(timelesscubes)

    ## extras
    water_and_ice = allcubes.extract(['cld_water',
                                      'cld_ice'])
    if len(water_and_ice) == 2:
        water,ice=water_and_ice
        qc = (water+ice) * 1000
        qc.units = 'g kg-1'
        qc.rename('qc')
        allcubes.append(qc)

    return allcubes


def read_standard_run(model_run, hour, constraints=None, extent=None, HSkip=None):
    '''
        Read converted model output files
        returns list of 4 iris cube lists, matching the model output files: slv, ro1, th1, th2
        ========
        0: specific_humidity / (1)             (time: 6; latitude: 576; longitude: 576)
        1: surface_air_pressure / (Pa)         (time: 6; latitude: 576; longitude: 576)
        2: air_pressure_at_sea_level / (Pa)    (time: 6; latitude: 576; longitude: 576)
        3: surface_temperature / (K)           (time: 6; latitude: 576; longitude: 576)
        ========
        0: air_pressure / (Pa)                 (time: 6; model_level_number: 140; latitude: 576; longitude: 576)
        1: x_wind / (m s-1)                    (time: 6; model_level_number: 140; latitude: 576; longitude: 576)
        2: y_wind / (m s-1)                    (time: 6; model_level_number: 140; latitude: 577; longitude: 576)
        ========
        0: air_pressure / (Pa)                 (time: 6; model_level_number: 140; latitude: 576; longitude: 576)
        1: air_temperature / (K)               (time: 6; model_level_number: 140; latitude: 576; longitude: 576)
        2: specific_humidity / (kg kg-1)       (time: 6; model_level_number: 140; latitude: 576; longitude: 576)
        3: upward_air_velocity / (m s-1)       (time: 6; model_level_number: 140; latitude: 576; longitude: 576)
        ========
        0: mass_fraction_of_cloud_ice_in_air / (kg kg-1) (time: 6; model_level_number: 140; latitude: 576; longitude: 576)
        1: mass_fraction_of_cloud_liquid_water_in_air / (kg kg-1) (time: 6; model_level_number: 140; latitude: 576; longitude: 576)
    '''

    dstamp=hour.strftime('%Y%m%d%H')
    ddir = __DATADIR__ + model_run+'/atmos/'

    # If we just want a particular extent, subset to that extent using constraints
    if extent is not None:
        constraints = _constraints_from_extent_(extent,constraints)

    # 3 file types for new waroona output
    ## slx - single level output
    ## mdl_ro1 - multi level, on ro dimension (winds)
    ## mdl_th1 - multi level, on theta dimension
    ## mdl_th2 - multi level, on theta dimension (since um output file size is limited)
    _standard_run_vars_ = {
        'slv':[
            'specific_humidity',  # kg/kg [t,lat,lon]
            #'spec_hum', # kg/kg [t,lat,lon]
            'av_soil_mois', # kg/m-2 [t,depth,lat,lon] #soil moisture content
            'surface_air_pressure', # Pa
            'mslp', # Pa [t,lat,lon]
            'surface_temperature', # [t,y,x]
            # x_wind and y_wind are here also...
            ],
        'mdl_ro1':[
            'air_pressure', # [t,z,lat,lon]
            'x_wind', # [t,z,y,x]
            'y_wind', # [t,z,y,x]
            #'height_above_reference_ellipsoid', # m [z,y,x]
            ],
        'mdl_th1':[
            'pressure', # Pa [tzyx] # air_pressure
            'air_temp', # K [tzyx]
            'specific_humidity', # kg/kg [tzyx]
            'vertical_wnd', # m/s [tzyx] # upward_air_velocity
            ],
        'mdl_th2':[
            #'mass_fraction_of_cloud_ice_in_air', # kg/kg [tzyx]
            #'mass_fraction_of_cloud_liquid_water_in_air',
            'cld_water',
            'cld_ice',
            ],
        }
    cubelists = []
    for filetype,varnames in _standard_run_vars_.items():
        path = ddir+'umnsaa_%s_%s.nc'%(dstamp,filetype)
        cubelists.append(read_nc_iris(path,constraints=constraints,keepvars=varnames,HSkip=HSkip))

    return cubelists

def remove_duplicate_cubes(dupes, cubeslist, ndims_expected=3):
    flag = False
    if len(dupes) > 1:
        for dupe in dupes:
            if flag or (len(dupe.shape) != ndims_expected):
                cubeslist.remove(dupe)
            else:
                # keep first 3d instance of sh
                flag = True

def read_topog(model_version, extent=None, HSkip=None):
    '''
    Read topography cube
    '''

    ddir = '../data/' + model_version + '/atmos/'
    files= glob(ddir+"*_mdl_th1.nc")
    files.sort()
    
    constraints='surface_altitude'
    if extent is not None:
        constraints = _constraints_from_extent_(extent,constraints)
        
    assert len(files) > 0, "NO FILES FOUND IN "+ddir
    topog = read_nc_iris(files[0],
                         constraints = constraints, 
                         HSkip=HSkip,
                         )[0]
    
    # don't want time dim in topog
    topog = iris.util.squeeze(topog) 
    return topog

def subset_time_iris(cube,dtimes,seccheck=121):
    '''
    take a cube with the time dimension and subset it to just have dtimes
    can handle iris seconds, minutes, or hours time dim formats
    assert times are available within seccheck seconds of dtimes
    '''
    tdim  = cube.coord('time')
    secmult = 1
    grain = str(tdim.units).split(' ')[0]
    unitformat = '%s since %%Y-%%m-%%d %%H:%%M:00'%grain
    if grain == 'minutes':
        secmult=60
    elif grain == 'hours':
        secmult=3600

    d0 = datetime.strptime(str(tdim.units),unitformat)
    # datetimes from ff
    dt = np.array([d0 + timedelta(seconds=secs*secmult) for secs in tdim.points])
    # for each datetime in dtimes argument, find closest index in dt
    tinds = []
    for dtime in dtimes:
        tinds.append(np.argmin(abs(dt-dtime)))
    tinds = np.array(tinds)

    # Check that fire times are within 2 minutes of desired dtimes
    #print("DEBUG: diffs")
    #print([(ffdt[tinds][i] - dtimes[i]).seconds < 121 for i in range(len(dtimes))])
    assert np.all([(dt[tinds][i] - dtimes[i]).total_seconds() < seccheck for i in range(len(dtimes))]), "fire times are > 2 minutes from requested dtimes"

    # subset cube to desired times
    return cube[tinds]

def make_folder(pname):
    folder = '/'.join(pname.split('/')[:-1]) + '/'
    if not os.path.exists(folder):
        print("INFO: Creating folder:",folder)
        os.makedirs(folder)

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
    else:
        dstamp=plot_time

    if subdir is None:
        subdir = ""
    elif subdir[0] not in "/\\":
        subdir = "/"+subdir

    path='../figures/%s/%s%s/%s'%(model_run, plot_name, subdir, dstamp) + ext
    return path

def save_fig(model_run, plot_name, plot_time, plt,
             subdir=None, 
             ext='.png', 
             **savefigargs):
    """
    create figurename as figures/plot_name/model_run/extent_name/ddTHHMM.png

    INPUTS:
        model_run, plot_name : strings
        plot_time : can be datetime or string
            if datetime, pname is set to ddTHHMM
        plt : is instance of matplotlib.pyplot
        extent_name : name of extent used making plot
    """
    path = standard_fig_name(model_run, plot_name, plot_time, 
                             subdir, ext, **savefigargs)

    save_fig_to_path(path, plt, **savefigargs)



def read_model_timeseries(model_run,latlon,
                          horizontal_avg_in_degrees=None,
                          d0=None,dN=None,
                          wind_10m=True,
                          ):
    """
    TODO: Update this to work on casestudies dataset
    ARGUMENTS:
        horizontal_avg_in_degrees: instead of interpolating to a lat/lon, average horizontally up to this many degrees away
            currently this is a rectangle
        d0,dN: start and end date (optional)
        wind_10m flag tells method to add 10m winds from fire output files
    """
    lat,lon = latlon
    extent = [lon-.02, lon+.02, lat-.02, lat+.02] # just grab real close to latlon
    if horizontal_avg_in_degrees is not None:
        #assert horizontal_avg_in_degrees>0.01, "radial average too small!"
        extent = [lon-horizontal_avg_in_degrees, 
                  lon+horizontal_avg_in_degrees,
                  lat-horizontal_avg_in_degrees, 
                  lat+horizontal_avg_in_degrees]
    
    umhours = run_info[model_run]['filedates']
    
    # limit hours desired to d0-dN
    
    if d0 is not None:
        di=max(utils.date_index(d0,umhours)-1, 0)
        umhours=umhours[di:]
    if dN is not None:
        di=utils.date_index(dN,umhours)
        umhours=umhours[:di+1]
    ## Read Model output:
    
    cubes = read_model_run(model_run, fdtime=umhours, extent=extent, 
                                 add_topog=True, add_winds=True, 
                                 add_RH=True, add_z=True, 
                                 add_dewpoint=True, add_theta=True)
    
    ctimes = utils.dates_from_iris(cubes.extract('upward_air_velocity')[0])
    
    if wind_10m:
        u10,v10=read_fire(model_run, extent=extent, dtimes=ctimes,
                          firefront=False, wind=True)
        u10.rename('u_10m')
        v10.rename('v_10m')
        s10=utils.wind_speed_from_uv_cubes(u10,v10)
        s10.units='m s-1'
        s10.var_name='s_10m'
        cubes.append(u10)
        cubes.append(v10)
        cubes.append(s10)
    
    # remove unwanted times
    # if d0 is not None:
    #     di=utils.date_index(d0,ctimes)
    #     for ci,cube in enumerate(cubes):
    #        if 'time' in cube.coords():
    #            cubes[ci] = cube[di:]
    #     ctimes=ctimes[di:]
    # if dN is not None:
    #     di=utils.date_index(dN,ctimes)
    #     for ci,cube in enumerate(cubes):
    #        if 'time' in cube.coords():
    #            cubes[ci] = cube[:(di+1)]
    #     ctimes=ctimes[:di]
    # Interpolate or average the horizontal component
    for ci,cube in enumerate(cubes):
        #print("DEBUG: interp step:",ci,cube.name(), cube.shape)
        if 'latitude' in [coord.name() for coord in cube.coords()]:
            if horizontal_avg_in_degrees:
                # dimensions to collapse will be 'latitude' and 'longitude'
                cubes[ci] = cube.collapsed(['latitude','longitude'], iris.analysis.MEAN)
            else:
                cubes[ci] = utils.profile_interpolation(cube,latlon,)
        #print("     :",ci,cubes[ci].name(),cubes[ci].shape)
    return cubes
