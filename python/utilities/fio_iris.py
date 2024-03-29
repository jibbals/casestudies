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
if iris.__version__ < '3':
    from iris.experimental.equalise_cubes import equalise_attributes
else:
    from iris.util import equalise_attributes

from iris.cube import CubeList as icCubeList
from iris.cube import Cube as icCube
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
## MOVING STUFF TO fio.py, still want them available here until this file completely refactored
###
from utilities.fio import hours_available, fire_paths, fire_path
from utilities.fio import atmos_paths as model_run_paths

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
    cubes1 = icCubeList([x,y,wdir,z,topog, T, RH])
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
    cubes2 = icCubeList([x1,y1,wdir1])
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
            small_cubes = icCubeList()
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
        print("ERROR: no data for:")
        print("     : fpath: ",fpath)
        print("     :",constraints)
    return cubes

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
    cubelist = icCubeList()

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
        
    if len(files) < 1:
        print("ERROR: no data for:")
        print("     : model run: ",model_version)
        print("     : extent:", str(extent))
    assert len(files) > 0, "NO FILES FOUND IN "+ddir
    topog = read_nc_iris(files[0],
                         constraints = constraints, 
                         HSkip=HSkip,
                         )[0]
    
    # don't want time dim in topog
    topog = iris.util.squeeze(topog) 
    return topog

def read_PFT_timeseries(mr,latlon,
                        force_recreate=False,
                        interp_method="nearest",):
    lat,lon = latlon
    extent=[lon-.01, lon+.01, lat-.01, lat+.01] # WESN
    
    fname = "../data/PFT/"+mr+str(lat)+","+str(lon)+".nc"
    if os.path.isfile(fname) and not force_recreate:
        print("INFO: Reading already created netcdf timeseries:",fname)
        return iris.load(fname)
    print("INFO: creating netcdf timeseries:",fname)
    
    
    # let's time how long it takes
    hours = hours_available(mr)
    
    PFT_full=[]
    dtimes=[]
    cubes=None
    topog=read_topog(mr,extent=extent)
    for hour in hours:
        del cubes
        # Read the cubes for one hour at a time
        cubes = read_model_run(mr, hours=[hour], extent=extent,)
        utils.extra_cubes(cubes,
                          add_z=True, 
                          add_RH=True,
                          #add_topog=True, 
                          add_winds=True,
                          add_theta=True,
                          )
        cubes.append(topog) # add topog to CubeList
        #print("DEBUG: topog", topog) 
        #print(cubes)
        PFT_full.append(utils.PFT_from_cubelist(cubes, latlon=latlon))
        dtimes.append(cubes[0].dim_coords[0].points)
    
    ## COMBINE PFT ARRAY along time dimension...
    PFT=np.concatenate(PFT_full,axis=0)
    dtimes = np.concatenate(dtimes,axis=0) # array of seconds since 1970-01-01 00:00:00
    ## Fix coordinates
    ## Copy existing coords and attributes
    lats = cubes[0].coord('latitude').points
    lons = cubes[0].coord('longitude').points
    Pa, = cubes.extract('air_pressure')
    Pa = Pa[:,0] # surface only
    
    # copy dimensions
    timecoord = iris.coords.DimCoord(dtimes, 'time', units=Pa.coord('time').units)
    # time is only dimension since we have single lat,lon
    dim_coords = [(timecoord,0),]

    # create cube from PFT array
    #print("DEBUG: dim_coords",dim_coords)
    #print("DEBUG: dim_coords[0]",dim_coords[0])
    #print("DEBUG: type PFT", type(PFT))
    #print("DEBUG: shape PFT", np.shape(PFT))
    #print("DEBUG: PFT", PFT)
    PFTcube = icCube(PFT,
                             var_name='PFT',
                             units='Gigawatts',
                             dim_coords_and_dims=dim_coords)
    # Keep some attributes too
    attrkeys=['source', 'um_version', 'institution', 'title', 
              'summary', 'project', 'acknowledgment', 
              'license', 
              'geospatial_lon_units',
              'geospatial_lat_units',
              'publisher_institution', 'publisher_name', 'publisher_type', 
              'publisher_url', 'naming_authority']
    attributes={}
    for k in attrkeys:
        if k in Pa.attributes.keys():
            attributes[k] = Pa.attributes[k]
    #attributes = { k:Pa.attributes[k] for k in attrkeys }
    attributes['geospatial_lon_max'] = np.max(lons)
    attributes['geospatial_lat_max'] = np.max(lats)
    PFTcube.attributes=attributes
    
    # save file
    make_folder(fname)
    iris.save(PFTcube,fname)
    
    # test file:
    f = iris.load(fname)
    print("INFO: SAVED ",fname)
    #print(f)
    return f

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




