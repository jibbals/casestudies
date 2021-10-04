#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 11:39:09 2019

@author: jesse
"""

import numpy as np
from datetime import datetime,timedelta

# interpolation package
from scipy import interpolate, signal
from pandas import Timedelta

import iris
import xarray as xr
from timeit import default_timer as timer
from utilities import constants
from utilities import PFT as PFT_calc

# more print statements for testing
__VERBOSE__=False

def distance_between_points(latlon0,latlon1):
    """
    return distance between lat0,lon0 and lat1,lon1
        IN METRES
    
    calculated using haversine formula, shortest path on a great-circle
     - see https://www.movable-type.co.uk/scripts/latlong.html
    """
    R = 6371e3 # metres (earth radius)
    lat0, lon0 = latlon0
    lat1, lon1 = latlon1
    latr0 = np.deg2rad(lat0)
    latr1 = np.deg2rad(lat1)
    dlatr = np.deg2rad(lat1-lat0)
    dlonr = np.deg2rad(lon1-lon0)
    a = np.sin(dlatr/2.0)**2 + np.cos(latr0)*np.cos(latr1)*(np.sin(dlonr/2.0)**2)
    c = 2*np.arctan2(np.sqrt(a),np.sqrt(1-a))
    return R*c

def extra_DataArrays(DS,
                add_z=False,
                add_winds=False,
                add_dewpoint=False,
                add_RH=False,
                add_theta=False,):
    """
    Additional DAs added to model DataSet (DS)
    """
    if 'air_pressure' in DS:
        p = DS['air_pressure']
    else:
        p = DS['pressure']
    if 'air_temperature' in DS:
        Ta = DS['air_temperature']
    elif 'air_temp' in DS:
        Ta = DS['air_temp']
    else:
        Ta = DS['temperature']
    if 'air_pressure_at_sea_level' in DS:
        pmsl = DS['air_pressure_at_sea_level']
    else:
        pmsl = DS['mslp']
    if 'x_wind' in DS:
        u1, v1 = DS['x_wind'],DS['y_wind']
    else:
        u1,v1 = DS['wnd_ucmp'],DS['wnd_vcmp']
    
    # add z_theta
    if add_z:
        
        nt,nz,ny,nx = p.shape
        # repeat surface pressure along new z axis
        reppmsl0 = np.repeat(pmsl.data[np.newaxis,:,:,:],nz, axis=0)
        # put time dim first to match air pressure
        reppmsl = np.transpose(reppmsl0,(1,0,2,3))
        zth = -(287*300/9.8)*np.log(p.data/reppmsl)
        zth_DA = xr.DataArray(data=zth,
                              coords=p.coords,
                              dims=p.dims,
                              name="z_th",
                              attrs={"units":"m",
                                     "desc":"-(287*300/9.8)*np.log(P/mslp)",
                                     },
                              )
        
        DS["z_th"]=zth_DA

    if add_winds:
        # wind speeds need to be interpolated onto non-staggered latlons
        
        
        ### DESTAGGER u and v using iris interpolate
        ### (this will trigger the delayed read)
        # u1: [t,z, lat, lon1]
        # v1: [t,z, lat1, lon]  # put these both onto [t,z,lat,lon]
        u=u1.interp(longitude_0=v1.coords['longitude'])
        #print(u)
        #print(u.coords)
        #print(u.dims)
        #u.rename({"longitude_0":"longitude"})
        v=v1.interp(latitude_0=u1.coords['latitude'])
        #v.rename({"latitude_0":"latitude"})
        
        # Get wind speed cube using hypotenuse of u,v
        s = wind_speed(u.data,v.data)
        s_DA = xr.DataArray(data=s,
                            coords=u.coords,
                            dims=u.dims,
                            name="s",
                            attrs={"units":"m s-1",
                                   "desc":"hypot(u,v)",
                                   },
                           )
        # Get wind direction using arctan of y/x
        wind_dir = wind_dir_from_uv(u.data,v.data)
        wd_DA = xr.DataArray(data=wind_dir,
                            coords=u.coords,
                            dims=u.dims,
                            name="wind_direction",
                            attrs={"units":"degrees",
                                   "desc":"arctan(u,v) flipped and rotated to met standard",
                                   },
                           )
        DS['u'] = u
        DS['v'] = v
        DS['s']=s_DA
        DS['wind_direction']=wd_DA

    #    if add_dewpoint:
    #        # Take pressure and relative humidity
    #        #print("DEBUG: add_dewpoint", allcubes)
    #        p,q = allcubes.extract(['air_pressure','specific_humidity'])
    #        p_orig_units = p.units
    #        q_orig_units = q.units
    #        p.convert_units('hPa')
    #        q.convert_units('kg kg-1')
    #        
    #        # calculate vapour pressure:
    #        epsilon = 0.6220 # gas constant ratio for dry air to water vapour
    #        e = p*q/(epsilon+(1-epsilon)*q)
    #        e.rename('vapour_pressure')
    #        e.units = 'hPa'
    #        #e_correct=np.mean(e.data)
    #        p.convert_units(p_orig_units)
    #        q.convert_units(q_orig_units)
    #        #assert np.isclose(np.mean(e.data),e_correct), "Changing units back messes with vapour_presssure"
    #
    #        allcubes.append(e)
    #        # calculate dewpoint from vapour pressure
    #        Td = 234.5 / ((17.67/np.log(e.data/6.112))-1) # in celcius
    #        Td = Td + 273.15 # celcius to kelvin
    #
    #        # change Td to a Cube
    #        iris.std_names.STD_NAMES['dewpoint_temperature'] = {'canonical_units': 'K'}
    #        cubeTd = iris.cube.Cube(Td, standard_name="dewpoint_temperature",
    #                                   var_name="Td", units="K",
    #                                   dim_coords_and_dims=[(p.coord('time'),0),
    #                                                        (p.coord('model_level_number'),1),
    #                                                        (p.coord('latitude'),2),
    #                                                        (p.coord('longitude'),3)])
    #
    #        allcubes.append(cubeTd)

    if add_theta:
        # Estimate potential temp
        
        theta = potential_temperature(p.data,Ta.data)
        theta_DA = xr.DataArray(data=theta,
                            coords=p.coords,
                            dims=p.dims,
                            name="theta",
                            attrs={"units":"K",
                                   "desc":"T*(1e5/p)**(287.05/1004.64)",
                                   },
                           )
        DS['potential_temperature']=theta_DA
    
    #    if add_RH:
    #        # estimate relative humidity
    #        q,T = allcubes.extract(['specific_humidity','air_temperature'])
    #        # compute RH from specific and T in kelvin
    #        orig_Tunits=T.units
    #        T.convert_units('K')
    #        RH = relative_humidity_from_specific(q.data, T.data)
    #        # restore T units (just in case)
    #        T.convert_units(orig_Tunits)
    #        # turn RH into a cube and add to return list
    #        iris.std_names.STD_NAMES['relative_humidity'] = {'canonical_units': '1'}
    #        cubeRH = iris.cube.Cube(RH, standard_name="relative_humidity",
    #                                   var_name="RH", units="1",
    #                                   dim_coords_and_dims=[(q.coord('time'),0),
    #                                                        (q.coord('model_level_number'),1),
    #                                                        (q.coord('latitude'),2),
    #                                                        (q.coord('longitude'),3)])
    #        allcubes.append(cubeRH)
    return DS

def extra_cubes(allcubes,
                add_z=False,
                add_winds=False,
                add_dewpoint=False,
                add_RH=False,
                add_theta=False,):
    """
    Additional cubes added to model cubes
    """

    if add_z:
        # add zth cube
        p, pmsl = allcubes.extract(['air_pressure','air_pressure_at_sea_level'])
        ## DONT take out time dimension
        ##p, pmsl = p[0], pmsl[0]
        nt,nz,ny,nx = p.shape
        # repeat surface pressure along new z axis
        reppmsl0 = np.repeat(pmsl.data[np.newaxis,:,:,:],nz, axis=0)
        # put time dim first to match air pressure
        reppmsl = np.transpose(reppmsl0,(1,0,2,3)) 
        zth = -(287*300/9.8)*np.log(p.data/reppmsl)
        iris.std_names.STD_NAMES['z_th'] = {'canonical_units': 'm'}
        zthcube=iris.cube.Cube(zth, standard_name='z_th',
                               var_name="zth", units="m",
                               dim_coords_and_dims=[(p.coord('time'),0),
                                                    (p.coord('model_level_number'),1),
                                                    (p.coord('latitude'),2),
                                                    (p.coord('longitude'),3)])
        allcubes.append(zthcube)

    if add_winds:
        # wind speeds need to be interpolated onto non-staggered latlons
        u1, v1 = allcubes.extract(['x_wind','y_wind'])
        
        ### DESTAGGER u and v using iris interpolate
        ### (this will trigger the delayed read)
        # u1: [t,z, lat, lon1]
        # v1: [t,z, lat1, lon]  # put these both onto [t,z,lat,lon]
        u = u1.interpolate([('longitude',v1.coord('longitude').points)],
                           iris.analysis.Linear())
        v = v1.interpolate([('latitude',u1.coord('latitude').points)],
                           iris.analysis.Linear())
        
        # add standard names for these altered variables:
        iris.std_names.STD_NAMES['u'] = {'canonical_units': 'm s-1'}
        iris.std_names.STD_NAMES['v'] = {'canonical_units': 'm s-1'}
        u.standard_name='u'
        v.standard_name='v'
        # Get wind speed cube using hypotenuse of u,v
        s = wind_speed_from_uv_cubes(u,v)
        s.units = 'm s-1'
        s.var_name='s' # s doesn't come from a var with a std name so can just use var_name
        
        # Get wind direction using arctan of y/x
        wind_dir = wind_dir_from_uv(u.data,v.data)
        
        wdcube = iris.cube.Cube(wind_dir,
                                var_name='wind_direction',
                                units='degrees',
                                #long_name='degrees clockwise from due North',
                                dim_coords_and_dims=[(s.coord('time'),0),
                                                     (s.coord('model_level_number'),1),
                                                     (s.coord('latitude'),2),
                                                     (s.coord('longitude'),3)])
        wdcube.units = 'degrees'
        wdcube.var_name='wind_direction'
        
        # add cubes to list
        allcubes.append(u)
        allcubes.append(v)
        allcubes.append(s)
        allcubes.append(wdcube)

    if add_dewpoint:
        # Take pressure and relative humidity
        #print("DEBUG: add_dewpoint", allcubes)
        p,q = allcubes.extract(['air_pressure','specific_humidity'])
        p_orig_units = p.units
        q_orig_units = q.units
        p.convert_units('hPa')
        q.convert_units('kg kg-1')
        
        # calculate vapour pressure:
        epsilon = 0.6220 # gas constant ratio for dry air to water vapour
        e = p*q/(epsilon+(1-epsilon)*q)
        e.rename('vapour_pressure')
        e.units = 'hPa'
        #e_correct=np.mean(e.data)
        p.convert_units(p_orig_units)
        q.convert_units(q_orig_units)
        #assert np.isclose(np.mean(e.data),e_correct), "Changing units back messes with vapour_presssure"

        allcubes.append(e)
        # calculate dewpoint from vapour pressure
        Td = 234.5 / ((17.67/np.log(e.data/6.112))-1) # in celcius
        Td = Td + 273.15 # celcius to kelvin

        # change Td to a Cube
        iris.std_names.STD_NAMES['dewpoint_temperature'] = {'canonical_units': 'K'}
        cubeTd = iris.cube.Cube(Td, standard_name="dewpoint_temperature",
                                   var_name="Td", units="K",
                                   dim_coords_and_dims=[(p.coord('time'),0),
                                                        (p.coord('model_level_number'),1),
                                                        (p.coord('latitude'),2),
                                                        (p.coord('longitude'),3)])

        allcubes.append(cubeTd)

    if add_theta:
        # Estimate potential temp
        p, Ta = allcubes.extract(['air_pressure','air_temperature'])
        theta = potential_temperature(p.data,Ta.data)
        # create cube
        iris.std_names.STD_NAMES['potential_temperature'] = {'canonical_units': 'K'}
        cubetheta = iris.cube.Cube(theta, standard_name="potential_temperature",
                                   var_name="theta", units="K",
                                   dim_coords_and_dims=[(p.coord('time'),0),
                                                        (p.coord('model_level_number'),1),
                                                        (p.coord('latitude'),2),
                                                        (p.coord('longitude'),3)])
        allcubes.append(cubetheta)
    
    if add_RH:
        # estimate relative humidity
        q,T = allcubes.extract(['specific_humidity','air_temperature'])
        # compute RH from specific and T in kelvin
        orig_Tunits=T.units
        T.convert_units('K')
        RH = relative_humidity_from_specific(q.data, T.data)
        # restore T units (just in case)
        T.convert_units(orig_Tunits)
        # turn RH into a cube and add to return list
        iris.std_names.STD_NAMES['relative_humidity'] = {'canonical_units': '1'}
        cubeRH = iris.cube.Cube(RH, standard_name="relative_humidity",
                                   var_name="RH", units="1",
                                   dim_coords_and_dims=[(q.coord('time'),0),
                                                        (q.coord('model_level_number'),1),
                                                        (q.coord('latitude'),2),
                                                        (q.coord('longitude'),3)])
        allcubes.append(cubeRH)
    return allcubes

def FFDI(DF,RH,T,v):
    """
    The FFDI is a key tool for assessing fire danger in Australia. 
    The formulation of the FFDI (e.g. Noble et al. 1980) is based on the 
    temperature (C), T, wind speed (km h-1), v, relative humidity (%), RH,
    and  a  component  representing  fuel  availability  called  the  
    Drought  Factor,  DF.
    The  Drought  Factor  is  given  as  a  number  between  0  and  10  and  
    represents  the  influence  of  recent temperatures and rainfall events on 
    fuel availability (see Griffiths 1998 for details).
    # From https://www.bushfirecrc.com/sites/default/files/managed/resource/ctr_010_0.pdf
    ARGUMENTS:
        DF: Drought factor (0 to 10)
        RH: rel humidity as a %
        T: Temperature (Celcius)
        v: wind speed (km/h)
    """
    # https://www.bushfirecrc.com/sites/default/files/managed/resource/ctr_010_0.pdf
    ffdi=2*np.exp(-0.45 + 0.987*np.log(DF) - .0345*RH+.0338*T+.0234*v)
    return ffdi

def profile_interpolation(cube, latlon, average=False):
    """
    interpolate iris cube along vertical dimension at given [lat,lon] pair
    optionally use average instead of bilinear interp
    ARGS:
        cube: iris cube [[time],lev,lat,lon]
        latlon: [lat, lon]
        average (optional): how many kilometers to use for average
            TO BE IMPLEMENTED
    """
    
    # Direct interpolation
    data0 = iris.util.squeeze(
        cube.interpolate(
            [('longitude',[latlon[1]]), ('latitude',[latlon[0]])],
            iris.analysis.Linear()
            )
        )
    #if average>0:
        #do stuff
    return data0

def number_of_interp_points(lats,lons,start,end, factor=1.5):
    """
    Returns how many points should be interpolated to between start and end on latlon grid
    Based on min grid size, multiplied by some factor
        The factor drastically reduces a runtime warning stating "too many knots added"
    """
    lat0,lon0 = start
    lat1,lon1 = end
    
    # min grid spacing
    min_dy = np.min(np.abs(np.diff(lats)))
    min_dx = np.min(np.abs(np.diff(lons)))
    
    # minimum resolvable lateral distance
    dgrid = np.hypot(min_dx,min_dy) * factor
    
    # line length (start to end)
    dline = np.sqrt((lon1-lon0)**2 + (lat1-lat0)**2)
    
    nx = int(np.ceil(dline/dgrid))
    if nx < 2:
        print("ERROR: start and end are too close for interpolation")
        print("     : lats[0:3]:", lats[0:3])
        print("     : lons[0:3]:", lons[0:3])
        print("     : start, end:", start, end)
        assert False, "Start and end are too close"
    if nx == 2:
        nx = 3 # best to have at least 3 points
    return nx

def latslons_axes_along_transect(lats,lons,start,end,nx):
    """
    INPUTS:
        lats, lons: 1darrays of degrees
        start: [lat0,lon0] # start of transect
        end: [lat1,lon1] # end of transect
        
    return lons,lats
        interpolated degrees from start to end along lats,lons grid
    """
    x_factor = np.linspace(0,1,nx)
    lat0,lon0 = start
    lat1,lon1 = end
    lons_axis = lon0 + (lon1-lon0)*x_factor
    lats_axis = lat0 + (lat1-lat0)*x_factor
    return lats_axis,lons_axis

def interp_cube_to_altitudes(cube, altitudes, model_heights=None, closest=False):
    """
    take cube, interpolate to altitude using approximated "level_height" coordinate
    if that fails (or if closest is True), use closest indices from model_heights
    """
    coord_names=[coord.name() for coord in cube.coords()]
    height_coord_name = None
    if 'level_height' in coord_names:
        height_coord_name='level_height'
    elif 'atmosphere_hybrid_height_coordinate' in coord_names:
        height_coord_name='atmosphere_hybrid_height_coordinate'
    
    if (height_coord_name is None) or closest:
        # get closest height indices
        hinds=np.zeros(len(altitudes)).astype(int)
        for i,wanted_height in enumerate(altitudes):
            hind = np.argmin(np.abs(model_heights-wanted_height))
            hinds[i]=hind
        
        # subset to height indices
        ret_cube  = cube[:,hinds,:,:]
    else:
        ret_cube = iris.util.squeeze(
            cube.interpolate(
                [(height_coord_name, altitudes)],
                iris.analysis.Linear()
                )
            )
    return ret_cube

def transect_slicex(lats,lons,start,end,nx=None, nz=1,
                    latlons=False):
    """
    Horizontal transect is based on lat,lon start and end
    This method returns the resultant xaxis as "metres from start"
    if latlons==True, return list of latlons between start and end point
    """
    if nx is None:
        nx = number_of_interp_points(lats,lons,start,end)
        
    if latlons:
        lllist = []
        D_lat = lats[-1]-lats[0]
        D_lon = lons[-1]-lons[0]
        for xfactor in np.linspace(0,1,nx):
            lllist.append([start[0]+D_lat*xfactor,start[1]+D_lon*xfactor])
        return lllist
        
    xlen = distance_between_points(start,end) 
    xaxis = np.linspace(0,xlen,nx)
    # multiply nrows by nz, ncols by 1 so [1,nx] -> [nz, nx]
    return np.tile(xaxis,[nz,1]) 

#def cross_section(data,lats,lons,start,end,npoints=None):
#    '''
#    interpolate along horisontal transect (cross section)
#    x,y axes will be in degrees
#    inputs: data = array[[z],lats,lons]
#            start = [lat0,lon0]
#            end = [lat1,lon1]
#            nx = how many points along horizontal?
#              If this is None, use resolution of grid
#    '''
#  
#    lat1,lon1=start
#    lat2,lon2=end
#      
#    # add z axis if there is none
#    if len(data.shape) < 3:
#        data=data[np.newaxis,:,:]
#    nz = data.shape[0]
#    
#    # base interp points on grid size
#    if npoints is None:
#        dy = lats[1] - lats[0]
#        dx = lons[1] - lons[0]
#        #print (dy, dx)
#        dgrid = np.hypot(dx,dy)
#        #print(dgrid)
#        dline = np.sqrt((lon2-lon1)**2 + (lat2-lat1)**2) # dist from start to end
#        npoints = int(np.ceil(dline/dgrid))
#      
#    # Define grid for horizontal interpolation. x increases from 0 to 1 along the
#    # desired line segment    
#    x_factor = np.linspace(0,1,npoints)
#    slicelon = lon1 + (lon2-lon1)*x_factor
#    slicelat = lat1 + (lat2-lat1)*x_factor
#    
#    # Interpolate data along slice (in model height coordinate). Note that the
#    # transpose is needed because RectBivariateSpline assumes the axis order (x,y)
#    slicedata = np.tile(np.nan, [nz, npoints])
#    for k in range(0,nz):
#        f = interpolate.RectBivariateSpline(lons,lats,data[k,:,:].transpose())
#        slicedata[k,:] = f.ev(slicelon,slicelat)
#      
#    return np.squeeze(slicedata)

def transect(data, lats, lons, start, end, nx=None, z=None,
             interpmethod='linear'):
    '''
    interpolate along cross section
    USES XARRAY INTERPOLATION 
    inputs: 
        data:[[z], lats, lons] array
        lats, lons: horizontal dims 1d arrays
        start = [lat0,lon0]
        end = [lat1,lon1]
        nx = how many points along horizontal. defaults to grid size
        z_th = optional altitude array [z, lats, lons]
    RETURNS: 
        struct: {
            'transect': vertical cross section of data 
            'x': 0,1,...,len(X axis)-1
            'y': y axis [Y,X] in terms of z
            'lats': [X] lats along horizontal axis
            'lons': [X] lons along horizontal axis
        } 
        xaxis: x points in metres
        yaxis: y points in metres or None if no z provided
    '''
    lat1,lon1 = start
    lat2,lon2 = end
    
    # base interp points on grid size
    if nx is None:
        nx = number_of_interp_points(lats,lons,start,end)
    
    # Interpolation line is really a list of latlons
    lataxis,lonaxis = latslons_axes_along_transect(lats,lons,start,end,nx=nx)
    # Create label to help interpret output
    label=["(%.2f, %.2f)"%(lat,lon) for lat,lon in zip(lataxis,lonaxis)]
    xdistance = np.array([distance_between_points(start, latlon) for latlon in zip(lataxis,lonaxis)])
    
    # Lets put our data into an xarray data array 
    coords = []
    if len(data.shape) ==3:
        coords = [("z",np.arange(data.shape[0]))]    
    coords.extend([("lats",lats),("lons",lons)])
    da = xr.DataArray(data,
                      coords)
    # we also put lat and lon list into data array with new "X" dimension
    da_lats = xr.DataArray(lataxis,dims="X")
    da_lons = xr.DataArray(lonaxis,dims="X")
    
    # interpolat to our lat,lon list
    slicedata = np.squeeze(da.interp(lats=da_lats,lons=da_lons,method=interpmethod).values)
    X=xdistance
    Y=None
    
    if z is not None:
        NZ=data.shape[0] # levels is first dimension
        da_z = xr.DataArray(z,
                            coords)
        # Y in 2d: Y [y,x]
        Y = np.squeeze(da_z.interp(lats=da_lats,lons=da_lons,method=interpmethod).values)
        # X in 2d: X [y,x]
        X = np.repeat(xdistance[np.newaxis,:],NZ,axis=0)
        
    return {'transect':slicedata, # interpolation of data along transect
            'xdistance':xdistance, # [x] metres from start
            'x':X, # [[y,]x] # metres from start, repeated along z dimension
            'y':Y, # [y,x] # interpolation of z input along transect
            'xlats':lataxis, # [x] 
            'xlons':lonaxis, # [x]
            'xlabel':label, # [x]
            }


def transect_old(data, lats, lons, start, end, nx=None, z_th=None):
    '''
    interpolate along cross section
    inputs: 
        wind: data [[z], lats, lons]
        lats, lons: horizontal dims 1d arrays
        start = [lat0,lon0]
        end = [lat1,lon1]
        nx = how many points along horizontal. defaults to grid size
        z_th = optional altitude array [z, lats, lons]
    RETURNS: 
        struct: {
            'transect': vertical cross section of data 
            'x': x axis [Y,X] in metres from start point 
            'y': y axis [Y,X] in terms of z_th
            'lats': [X] lats along horizontal axis
            'lons': [X] lons along horizontal axis
        } 
        xaxis: x points in metres
        yaxis: y points in metres or None if no z_th provided
    '''
    lat1,lon1 = start
    lat2,lon2 = end
    
    # add z axis if there is none
    if len(data.shape) < 3:
        data=data[np.newaxis,:,:]
    nz = data.shape[0]
    
    # base interp points on grid size
    if nx is None:
        nx = number_of_interp_points(lats,lons,start,end)
    
    # Define grid for horizontal interpolation.
    lataxis,lonaxis = latslons_axes_along_transect(lats,lons,start,end,nx=nx)
    label=["(%.2f, %.2f)"%(lat,lon) for lat,lon in zip(lataxis,lonaxis)]
    
    # Interpolate data along slice (in model height coordinate). Note that the
    # transpose is needed because RectBivariateSpline assumes the axis order (x,y)
    slicedata = np.tile(np.nan, [nz, nx])
    slicez = np.tile(np.nan, [nz, nx])
    for k in range(0,nz):
        f = interpolate.RectBivariateSpline(lons,lats,data[k,:,:].transpose())
        slicedata[k,:] = f.ev(lonaxis,lataxis)
        if z_th is not None:
            f2 = interpolate.RectBivariateSpline(lons,lats,z_th[k,:,:].transpose())
            slicez[k,:] = f2.ev(lonaxis, lataxis)
    
    # X, Y axes need to be on metres dimension!
    slicex=transect_slicex(lats,lons,start,end,nx=nx,nz=nz)
    
    return {'transect':np.squeeze(slicedata),
            'x':slicex,
            'y':slicez,
            'lats':lataxis,
            'lons':lonaxis,
            'label':label,
            }

def transect_winds(u,v,lats,lons,start,end,nx=None,z=None):
    """
    Get wind speed along arbitrary transect line
    ARGUMENTS:
        u[...,lev,lat,lon]: east-west wind speed
        v[...,lev,lat,lon]: north_south wind speed
        lats[lat]: latitudes
        lons[lon]: longitudes
        start[2]: lat,lon start point for transect
        end[2]: lat,lon end point for transect
        nx: optional number of interpolation points along transect
        z[...,lev,lat,lon]: optional altitude or pressure levels
        
    RETURNS: structure containing:
        'transect_angle': transect line angle (counter clockwise positive, east=0 degrees)
        'transect_wind':wind along transect (left to right is positive),
        'transect_v':v cross section,
        'transect_u':u cross section,
        'x': metres from start point along transect,
        'y': metres above surface along transect,
        'lats':transect latitudes,
        'lons':transect longitudes,
        'label':[x,'lat,lon'] list for nicer xticks in cross section
        
    """
    lat0,lon0=start
    lat1,lon1=end
    # signed angle in radians for transect line
    theta_rads=np.arctan2(lat1-lat0,lon1-lon0)
    theta_degs=np.rad2deg(theta_rads)
    #print("CHECK: angle between", start, end)
    #print("     : is ",theta_degs, "degrees?")
    # base interp points on grid size
    if nx is None:
        nx = number_of_interp_points(lats,lons,start,end)
        
    ucross_str=transect(u,lats,lons,
                    start=[lat0,lon0],
                    end=[lat1,lon1],
                    nx=nx,
                    z=z)
    ucross = ucross_str['transect']
    vcross_str=transect(v,lats,lons,
                    start=[lat0,lon0],
                    end=[lat1,lon1],
                    nx=nx,
                    z=z)
    vcross = vcross_str['transect']
    wind_mag = ucross * np.cos(theta_rads) + vcross * np.sin(theta_rads)
    
    ret={
        'transect_angle':theta_degs,
        'transect_wind':wind_mag,
        'transect_v':vcross,
        'transect_u':ucross,
        'x':ucross_str['x'],
        'y':ucross_str['y'],
        'xdistance':ucross_str['xdistance'],
        'xlats':ucross_str['xlats'],
        'xlons':ucross_str['xlons'],
        'xlabel':ucross_str['xlabel'],
        'nx':nx,
        }
    return ret

def cube_to_xyz(cube,
                ztop=-1):
    """
    take iris cube [lev, lat, lon]
    pull out the data and reshape it to [lon,lat,lev]
    """
    assert len(cube.shape)==3, "cube is not 3-D"
    data = cube[:ztop,:,:].data
    if np.ma.is_masked(data):
        data=data.data
    # data is now a level, lat, lon array... change to lon, lat, lev
    xyz = np.moveaxis(np.moveaxis(data,0,2),0,1)
    return xyz

def date_index(date,dates, dn=None, ignore_hours=False):
    """
    ARGUMENTS: date,dates,dn=None,ignore_hours=False
        closest index available within dates that matches date
        if dn is passed in, indices will list from date up to dn
    """
    new_date=date
    new_dn=dn
    new_dates=dates
    if ignore_hours:
        new_date=datetime(date.year,date.month,date.day)
        if dn is not None:
            new_dn=datetime(dn.year,dn.month,dn.day)
        new_dates = [datetime(d.year,d.month,d.day) for d in dates]
    whr=np.where(np.array(new_dates) == new_date) # returns (matches_array,something)
    if len(whr[0])==0:
        print ("ERROR: ",date, 'not in', new_dates[0], '...', new_dates[-1])
    elif dn is None:
        return whr[0][0] # We just want the match
    else:
        whrn=np.where(np.array(new_dates) == new_dn) # returns last date match
        if len(whrn[0])==0: # last date not in dataset
            print ("ERROR: ",new_dn, 'not in', new_dates[0], '...', new_dates[-1])
        return np.arange(whr[0][0],whrn[0][0]+1)

def date_from_gregorian(greg, d0=datetime(1970,1,1,0,0,0)):
    '''
        gregorian = "hours since 19700101 00:00:00"
        Returns nparray of datetimes
    '''
    greg=np.array(greg)
    if greg.ndim==0:
        return np.array( [d0+timedelta(seconds=int(greg*3600)),])
    return np.array([d0+timedelta(seconds=int(hr*3600)) for hr in greg])

def height_from_iris(cube,bounds=False):
    """
    Return estimate of altitudes from cube: numpy array [levels]
        looks for atmosphere_hybrid_height_coordinate, or level_height auxiliary coord
    """
    height=None
    coord_names=[coord.name() for coord in cube.coords()]
    if 'level_height' in coord_names:
        height = cube.coord('level_height').points
        zbounds = cube.coord('level_height').bounds
    elif 'atmosphere_hybrid_height_coordinate' in coord_names:
        height = cube.coord('atmosphere_hybrid_height_coordinate').points
        zbounds = cube.coord('atmosphere_hybrid_height_coordinate').bounds
    else:
        print("ERROR: no height estimate in ", cube.name())
        print("     : coord names: ",coord_names)
        print("     : cube: ", cube)
        assert False, "no height coords"
    if bounds:
        height = zbounds
        #print(type(height),np.shape(height))
    return height

def unmask(arr):
    if np.ma.isMaskedArray(arr):
        #print("     : returning arr.data")
        return arr.data
    return arr


def dates_from_iris(timedim, remove_seconds=True):
    '''
    input is coord('time') and grain
    or else input a cube with a 'time' dim
    output is array of datetimes
    '''
    tdim=timedim
    if isinstance(timedim, iris.cube.Cube):
        tdim  = timedim.coord('time')
        grain = str(tdim.units).split(' ')[0]
    #elif isinstance(timedim, [iris.coords.DimCoord,iris.coords.Coord]):
    elif isinstance(timedim, iris.coords.Coord):
        grain = str(timedim.units).split(' ')[0]
    
    unitformat = '%s since %%Y-%%m-%%d %%H:%%M:%%S'%grain
    d0 = datetime.strptime(str(tdim.units),unitformat)
    secmult=1
    if grain=='minutes':
        secmult=60
    elif grain=='hours':
        secmult=60*60
    elif grain=='days':
        secmult=60*60*24
    
    dt = np.array([d0 + timedelta(seconds=secs*secmult) for secs in tdim.points])
    dtm = dt
    if remove_seconds:
        for i,d in enumerate(dt):
            iday = d.day
            ihour = d.hour
            iminute = d.minute+int(d.second>30)
            if iminute == 60:
                ihour+=1
                iminute=0
            if ihour == 24:
                iday+=1
                ihour=0
            dtm[i] = datetime(d.year,d.month, iday, ihour, iminute)
    return dtm

def firepower_from_cube(shcube):
    """
    calculate and return firepower over time in GWatts
    
    Inputs
    ======
        shcube: sensible heat flux in Watts/m2
    """
    
    lon,lat = shcube.coord('longitude'), shcube.coord('latitude')
    ### get areas in m2
    # Add boundaries to grid
    if lat.bounds is None:
        lat.guess_bounds()
    if lon.bounds is None:
        lon.guess_bounds()
    grid_areas = iris.analysis.cartography.area_weights(shcube)

    firepower = shcube.data * grid_areas # W/m2 * m2
    if np.ma.is_masked(firepower):
        firepower=firepower.data
    return firepower/1e9 # Watts to Gigawatts

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

def test_lat_lon_grid_area():
    """
    """
    lats=np.array([1,2,3,13])
    lons=np.array([2,3,13])
    lat_edges,lon_edges=lat_lon_grid_edges(lats,lons)
    print("TEST: lats, lat edges:",lats, lat_edges)
    assert np.all(np.isclose(lat_edges,np.array([0.5,1.5,2.5,8,18]))), "lat edges are wrong"
    print("TEST: lons, lon edges:",lons, lon_edges)
    assert np.all(np.isclose(lon_edges,np.array([1.5,2.5,8,18]))), "lon edges are wrong"
    area=lat_lon_grid_area(lats,lons)
    print("TEST: area(km2)=",area/1e6)
    print("TEST: shape(area)=",np.shape(area))
    assert np.isclose(10*area[0,0], area[0,2]), "area[0,0] should be a 10th of area[0,2] (10 times the longitude)"
    assert area[0,0] > area[1,0], "area[0,0] should be a greater than area[1,0] (higher latitude)"
    print("CHECK:make sense?")
    
    

def lat_lon_index(lat,lon,lats,lons):
    ''' lat,lon index from lats,lons    '''
    with np.errstate(invalid='ignore'):
        latind=(np.abs(lats-lat)).argmin()
        lonind=(np.abs(lons-lon)).argmin()
    return latind,lonind

def nearest_date_index(date, dates, allowed_seconds=120):
    """
    Return date index that is within allowewd_seconds of date
    """
    # Need to use total seconds, close negative time deltas have -1 day + ~80k seconds
    secs_diff = np.abs([ tdelta.total_seconds() for tdelta in (np.array(dates) - date)])
    ind = np.argmin(secs_diff)
    assert secs_diff[ind] <= allowed_seconds, "%s not within %d seconds of %s ... %s. \n "%(date.strftime("%Y%m%d-%H:%M"), allowed_seconds, dates[0].strftime("%Y%m%d-%H:%M"), dates[-1].strftime("%Y%m%d-%H:%M")) + str(dates[:])
    return ind

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

def zth_calc(pmsl,p):
    '''
    Calculate zth from pmsl and p
    pmsl : [t,lat,lon]
    p    : [t,lev,lat,lon]
    
    '''
    nt,nz,ny,nx = p.shape
    reppmsl = np.repeat(pmsl[:,np.newaxis,:,:],nz, axis=1) # repeat surface pressure along z axis
    return -(287*300/9.8)*np.log(p/reppmsl)

def potential_temperature(p,T):
    '''
    calculate theta from pressure and air temperature
    # Potential temperature based on https://en.wikipedia.org/wiki/Potential_temperature
    # with gas constant R = 287.05 and specific heat capacity c_p = 1004.64
    '''
    # maybe check for units
    if np.max(p) < 50000:
        print("WARNING: Potential temperature assumes pressure is in Pascals, highest input pressure is only ", np.max(p))
    if np.min(T) < 50:
        print("WARNING: Potential temperature assumes Temperature is in Kelvin, lowest input temp is only ", np.min(T))
    return T*(1e5/p)**(287.05/1004.64)

def destagger_winds_DA(DA_u, DA_v):
    """
    ## Staggered grid:
    ## u[latitude,longitude_edges] # minus one edge for some reason
    ## v[latitude_edges,longitude]
    ## fix by linear interpolation
    ARGS:
        DA_u = DataArray 'xwind' from model output with staggered longitudes
        DA_v = DataArray 'ywind' from model output with staggered latitudes
    RETURNS:
        u,v DataArrays on destaggered coordinates ['latitude','longitude']
    """
    if hasattr(DA_u,"latitude"):
        # move x wind onto y wind longitudes
        u = DA_u.interp(longitude_0=DA_v['longitude'].values)
        # then move y wind onto xwind latitudes
        v = DA_v.interp(latitude_0=DA_u['latitude'].values)
        # rename dimensions
        u=u.rename({"longitude_0":"longitude"})
        v=v.rename({"latitude_0":"latitude"})
    elif hasattr(DA_u,"lat"):
        # move x wind onto y wind longitudes
        u = DA_u.interp(lon_0=DA_v['lon'].values)
        # then move y wind onto xwind latitudes
        v = DA_v.interp(lat_0=DA_u['lat'].values)
        # rename dimensions
        u=u.rename({"lon_0":"lon"})
        v=v.rename({"lat_0":"lat"})
    else:
        print(DA_u)
        print("ERROR: COULDN'T FIND L(ong/at)ITUDE")
    
    
    return u,v

def destagger_winds(u1,v1,lats=None,lons=None,lats1=None,lons1=None):
    '''
    destagger winds from ACCESS um output
    wind speeds are on their directional grid edges
    #u1 = [time,levs,lat,lon1] 
    #v1 = [time,levs,lat1,lons]
    
    '''
    
    # can check assumptions  and fix !! todo
    if lats is not None:
        print("INFO: Destagger check:")
        arrlens=[len(arr) for arr in [lats,lats1,lons,lons1]]
        print("    : len(lats)=%d  len(lats1)=%d  len(lons)=%d  len(lons1)=%d"%(arrlens[0],arrlens[1],arrlens[2],arrlens[3]))
        print("    : lats1[0], lats[0], lats1[1]: ",lats1[0],lats[0],lats1[1])
        print("    : lats1[-2], lats[-1], lats1[-1]: ",lats1[-2],lats[-1],lats1[-1])
        print("    : lons1[0], lons[0], lons1[1]: ",lons1[0],lons[0],lons1[1])
        print("    : lons1[-2], lons[-1], lons1[-1]: ",lons1[-2],lons[-1],lons1[-1])
        print("INFO: Destagger ASSUMES that lons are edges but that they miss the LAST edge")
        ## TODO since we have actual dims here, can do proper interpolation
        
    
    #N_DIMS=len(np.shape(u1))
    u = np.tile(np.nan,u1.shape) # tile repeats the nan accross nz,ny,nx dimensions
    # if N_DIMS == 4:
    #     # interpolation of edges
    #     u[:,:,:,1:] = 0.5*(u1[:,:,:,1:] + u1[:,:,:,:-1])
    #     u[:,:,:,0] = u1[:,:,:,0]
    #     v = 0.5*(v1[:,:,1:,:] + v1[:,:,:-1,:])
    # elif N_DIMS == 3:
    #     u[:,:,1:] = 0.5*(u1[:,:,1:] + u1[:,:,:-1])
    #     u[:,:,0] = u1[:,:,0]
    #     v = 0.5*(v1[:,1:,:] + v1[:,:-1,:])
    # elif N_DIMS == 2:
        # u[:,1:] = 0.5*(u1[:,1:] + u1[:,:-1])
        # u[:,0] = u1[:,0]
        # v = 0.5*(v1[1:,:] + v1[:-1,:])
        
    ## model output misses eastmost edge(?)
    u[...,:,:-1] = 0.5*(u1[...,:,:-1] + u1[...,:,1:])
    u[...,:,-1] = u1[...,:,-1]
    ## latitude edges behave as expected
    v = 0.5*(v1[...,1:,:] + v1[...,:-1,:])
    
    return u,v

def destagger_wind_cubes(cubes):
    """
    """
    ## Staggered grid:
    ## u[latitude,longitude_edges]
    ## v[latitude_edges,longitude]
    ## fix by interpolation

    # assume length 2 input lists are just [x_wind,y_wind]
    if len(cubes) == 2:
        u1, v1 = cubes
    else:
        u1, v1 = cubes.extract(['x_wind','y_wind'])
    
    # interpolate xwind longitudes onto vwind longitude dim
    u = u1.interpolate([('longitude',v1.coord('longitude').points)],
                       iris.analysis.Linear())
    v = v1.interpolate([('latitude',u1.coord('latitude').points)],
                       iris.analysis.Linear())
        
    # add standard names for these altered variables:
    iris.std_names.STD_NAMES['u'] = {'canonical_units': 'm s-1'}
    iris.std_names.STD_NAMES['v'] = {'canonical_units': 'm s-1'}
    u.standard_name='u'
    v.standard_name='v'
    return u,v

def uv_from_wind_degrees(wd,met_convention=True):
    """
    return u,v vectors from wind direction argument (degrees)
    Assume wd in degrees clockwise from north, pointing to where the wind is coming from, 
    unless met_convention is set to False. 
    If false, wd is just math convention: 
        pointing direction of wind flow in degrees anticlockwise from due east
    """
    # change to math degrees
    wd_math = 270-wd if met_convention else wd
    # x vector is cos (angle)
    # numpy uses radians as input so degrees need to be converted
    u=np.cos(np.deg2rad(wd_math))
    # y vector is sin (angle)
    v=np.sin(np.deg2rad(wd_math))
    return u,v

def wind_speed(u,v):
    '''
    horizontal wind speed from u,v vectors
    '''
    
    s = np.hypot(u,v) # Speed is hypotenuse of u and v
    
    if np.sum(s==-5000)>0:
        print("ERROR: u and v are probably not on the same grid")
        print("     : This was occurring for me before I destaggered model x and y wind output")
        s[s==-5000] = np.NaN
        assert np.sum(np.isnan(s[:,:,:,1:]))==0, "Some nans are left in the wind_speed calculation"
        assert np.sum(s==-5000)==0, "Some -5000 values remain in the wind_speed calculation"
    assert np.sum(s<0) == 0, "Negative wind speeds have occurred"
    
    return s

def find_max_index_2d(field):
    """
    Find maximum location from 2d field
    Arguments:
        field [y,x] : 2d field 
    
    return yi,xi
    """
    # find max windspeed, put into same shape as winds [y,x]
    mloc = np.unravel_index(np.argmax(field,axis=None),field.shape)
    return mloc

def local_time_from_time_lats_lons(time_utc,lats,lons):
    if isinstance(time_utc,datetime):
        time_utc=np.datetime64(time_utc)
        
    houroffset=local_time_offset_from_lats_lons(lats,lons)
    if hasattr(time_utc,'__iter__'):
        time_lt = [ np.datetime64((utc+Timedelta(houroffset,'h'))).astype(datetime) for utc in time_utc]
    else:
        time_lt=np.datetime64((time_utc+Timedelta(houroffset,'h'))).astype(datetime)
    return time_lt

def local_time_offset_from_lats_lons(lats,lons):
    """
    Guess time based on mean longitude
    Assume summer
    return integer (hours offset from UTC+0)
    """
    meanlon=np.mean(lons)
    meanlat=np.mean(lats)
    
    if meanlon>141:
        # daylight savings in NSW and VIC, but not QLD
        off=11.0 if meanlat<-29.0 else 10.0
    elif meanlon > 129:
        off=10.5
    else:
        off=8.0 #WA... should be 8.75 for daylight savings???
        
    return off

def locations_from_extent(extent):
    """
    find matching location names and latlons from constants.latlons using an extent
    """
    W,E,S,N = extent
    names,latlons=[],[]
    for k,v in constants.latlons.items():
        lat,lon = v
        if lat > S and lat < N and lon > W and lon < E:
            names.append(k)
            latlons.append(v)
    return names,latlons


def PFT_from_cubelist(cubes0, latlon=None, tskip=None, latskip=None, lonskip=None):
    """
    Wrapper to call the PFT function using cubelist of data
    
    Loop over required dimensions, 200 PFT calculations take ~ 1 minutes.
    Subsetting or striding dimensions is recommended for larger data sets.
    cubelist requires these named cubes:
        'air_temperature', 'specific_humidity', 'potential_temperature',
        'air_pressure','u','v','upward_air_velocity',
        'surface_altitude', 'surface_air_pressure', 'surface_temperature'
    if latlon is specified, just one point is extracted using linear interpolation
    
    """
    # local cubelist copy, so that original cubes aren't modified
    cubes = iris.cube.CubeList()
    # if there's no surface data, use first level as surf
    if len(cubes0.extract('surface_temperature')) == 0:
        st = cubes0.extract('air_temperature')[0].copy()
        st.rename("surface_temperature")
        st = st[:,0,:,:]
        cubes0.append(st)
    if len(cubes0.extract('surface_air_pressure')) == 0:
        sap = cubes0.extract('air_pressure')[0].copy()
        sap.rename("surface_air_pressure")
        sap = sap[:,0,:,:]
        cubes0.append(sap)
        
    for cube in cubes0.extract(['air_temperature', 'specific_humidity', 
                    'potential_temperature', 'air_pressure', 'u', 'v',
                    'upward_air_velocity', 'surface_altitude', 
                    'surface_air_pressure', 'surface_temperature'], strict=True):
        cubes.append(cube.copy())
    
    
    ## first interpolate everything to latlon
    if latlon is not None:
        lat,lon = latlon
        for i in range(len(cubes)):
            cubes[i] = cubes[i].interpolate([('longitude',lon), ('latitude',lat)],
                                            iris.analysis.Linear())
    
    has_time_dim = len(cubes.extract('u')[0].coords('time')) == 1
    
    # Now subset everything based on skips
    if ((latskip is not None) or (lonskip is not None)) and (latlon is None):
        latslice = slice(None, None, latskip)
        lonslice = slice(None, None, lonskip)
        
        for i in range(len(cubes)):
            # cube needs to have spacial grid
            if len(cubes[i].coords('longitude')) == 1:
                # cube may have vertical grid and or temporal grid
                cshape=cubes[i].shape
                if len(cshape) == 2:
                    cubes[i] = cubes[i][latslice,lonslice]
                elif len(cshape) == 3:
                    cubes[i] = cubes[i][:,latslice,lonslice]
                elif len(cshape) == 4:
                    cubes[i] = cubes[i][:,:,latslice,lonslice]
    if has_time_dim and (tskip is not None):
        tslice = slice(None, None, tskip)
        # for each cube
        for i in range(len(cubes)):
            # if the cube has a time dim, of length greater than 1, slice it
            if len(cubes[i].coords('time')) == 1:
                if len(cubes[i].coord('time').points) > 1:
                    cubes[i] = cubes[i][tslice]
    
    if has_time_dim:
        cubedtimes = dates_from_iris(cubes.extract('u')[0])
    
    # now for easy reading pull out cubes
    TTcube, qqcube, thcube = cubes.extract(['air_temperature', 
                                            'specific_humidity',
                                            'potential_temperature'],strict=True)
    prcube, uucube, vvcube = cubes.extract(['air_pressure', 'u','v'],strict=True)
    wwcube = cubes.extract('upward_air_velocity',strict=True)
    
    # surface metrics
    # surface values in old run are on different time dimension...!?!
    zsfc, psfc, Tsfc = cubes.extract(['surface_altitude', 
                                      'surface_air_pressure', 
                                      'surface_temperature'])
    #print(zsfc.shape, psfc.shape, Tsfc.shape, has_time_dim, len(cubedtimes), latlon)
    #print(wwcube)
    #print(wwcube.coord('time'))
    zsfc = zsfc.data # m
    if len(zsfc.shape) == 0:
        zsfc = float(zsfc)
    
    # psfc and Tsfc may not have time dim.
    # if they do, or if nothing has time dims, just treat normally
    #if (len(wwcube.coord('time').points) == 1) or (not has_time_dim):
    #    psfc = psfc.data # Pa
    #    Tsfc = Tsfc.data # K
    #    if len(psfc.shape) == 0:
    #        psfc = float(psfc)
    #        Tsfc = float(Tsfc)
    # if they don't, and our other data has a time dim, repeat these over the time dim
    if psfc.shape[0] != wwcube.shape[0]:
        # repeat along time dim
        if latlon is None:
            psfc = np.repeat(psfc.data[np.newaxis,:,:], len(cubedtimes),axis=0)
            Tsfc = np.repeat(Tsfc.data[np.newaxis,:,:], len(cubedtimes),axis=0)
        else:
            psfc = np.repeat(float(psfc.data), len(cubedtimes),axis=0)
            Tsfc = np.repeat(float(Tsfc.data), len(cubedtimes),axis=0)
    else:
        psfc = psfc.data # Pa
        Tsfc = Tsfc.data # K
        if len(psfc.shape) == 0:
            psfc = float(psfc)
            Tsfc = float(Tsfc)
            
    # Return array
    PFT = np.zeros(Tsfc.shape)+np.NaN 
    
    # let's time how long it takes
    start = timer()
    
    ## Loop over time, lat, lon dimensions
    if latlon is None:
        lats,lons = TTcube.coord('latitude').points,TTcube.coord('longitude').points
        for yi in range(len(lats)):
            for xi in range(len(lons)):
                zsfc0 = zsfc[yi,xi] # zsfc never has time dim
                if has_time_dim:
                    for ti in range (len(cubedtimes)):
                        TT, qq = TTcube[ti,:,yi,xi].data.data, qqcube[ti,:,yi,xi].data.data
                        uu,vv,ww = uucube[ti,:,yi,xi].data.data, vvcube[ti,:,yi,xi].data.data, wwcube[ti,:,yi,xi].data.data
                        th,pr = thcube[ti,:,yi,xi].data.data, prcube[ti,:,yi,xi].data.data
                        psfc0, Tsfc0 = psfc[ti,yi,xi], Tsfc[ti,yi,xi]

                        # Find pressure level at which T = Tmin (-20 degr C)
                        # get first instance where TT is less than Tmin
                        Tmin_indices = TT < 253.15
                        pmin = pr[Tmin_indices][0]
                        
                        frets = PFT_calc.PFT(TT,qq,uu,vv,ww,th,pr, 
                                        zsfc0,psfc0,Tsfc0,
                                        Pmin=pmin)
                        PFT[ti,yi,xi] = frets[8]/1e9 # G Watts
                # if no time dimension
                else:
                    TT, qq = TTcube[:,yi,xi].data.data, qqcube[:,yi,xi].data.data
                    uu,vv,ww = uucube[:,yi,xi].data.data, vvcube[:,yi,xi].data.data, wwcube[:,yi,xi].data.data
                    th,pr = thcube[:,yi,xi].data.data, prcube[:,yi,xi].data.data
                    psfc0, Tsfc0 = psfc[yi,xi], Tsfc[yi,xi]

                    # Find pressure level at which T = Tmin (-20 degr C)
                    # get first instance where TT is less than Tmin
                    Tmin_indices = TT < 253.15
                    pmin = pr[Tmin_indices][0]
                    
                    frets = PFT_calc.PFT(TT,qq,uu,vv,ww,th,pr, 
                                    zsfc0,psfc0,Tsfc0,
                                    Pmin=pmin)
                    PFT[yi,xi] = frets[8]/1e9 # G Watts
    # If we have no lat, lon dim
    else:
        zsfc0 = zsfc
        if has_time_dim:
            for ti in range (len(cubedtimes)):
                TT, qq = TTcube[ti,:].data.data, qqcube[ti,:].data.data
                uu,vv,ww = uucube[ti,:].data.data, vvcube[ti,:].data.data, wwcube[ti,:].data.data
                th,pr = thcube[ti,:].data.data, prcube[ti,:].data.data
                psfc0, Tsfc0 = psfc[ti], Tsfc[ti]

                # Find pressure level at which T = Tmin (-20 degr C)
                # get first instance where TT is less than Tmin
                Tmin_indices = TT < 253.15
                if np.sum(Tmin_indices) < 1:
                    print("ERROR: PROFILE HAS NO TEMPERATURES LOWER THAN 253.15 KELVIN")
                    print("     : Maybe not using whole profile? or wrongly using celcius?")
                    print("     : profile follows:")
                    print(TT)
                    assert np.sum(Tmin_indices) > 0, "not a good profile"
                pmin = pr[Tmin_indices][0]
                
                frets = PFT_calc.PFT(TT,qq,uu,vv,ww,th,pr, 
                                zsfc0,psfc0,Tsfc0,
                                Pmin=pmin)
                PFT[ti] = frets[8]/1e9 # G Watts
        # if no time dimension and no spatial dim
        else:
            TT, qq = TTcube.data.data, qqcube.data.data
            uu,vv,ww = uucube.data.data, vvcube.data.data, wwcube.data.data
            th,pr = thcube.data.data, prcube.data.data
            psfc0, Tsfc0 = psfc, Tsfc

            # Find pressure level at which T = Tmin (-20 degr C)
            # get first instance where TT is less than Tmin
            Tmin_indices = TT < 253.15
            pmin = pr[Tmin_indices][0]
            
            frets = PFT_calc.PFT(TT,qq,uu,vv,ww,th,pr, 
                            zsfc0,psfc0,Tsfc0,
                            Pmin=pmin)
            PFT = frets[8]/1e9 # G Watts
    end = timer()
    print("Info: time to produce PFT( shape = %s): %.2f minutes"%(str(PFT.shape), (end-start)/60.0))
    return PFT


def dragana_vorticity(lat,lon,u_wind,v_wind, smoothing=False):
    """
    From dragana's script
    """
    # Necessary for vorticity calculation
    lon2 = lon[np.newaxis,:]
    lat2 = lat[:,np.newaxis]
    lon2_rad = np.radians(lon2)
    lat2_rad = np.radians(lat2)
    re = 6370e3 # m
    ##### For Coriolis parameter (planetary vort.)
    #omega = 7.292115e-5 # 1/s
    #R = 287
    #cor_term = 2 * omega * np.sin(lat2_rad)
    
    # looks like u and v need to be [lat,lon]
    if u_wind.shape[0] == len(lon):
        u_wind = u_wind.T
        v_wind = v_wind.T
        
    rel_vor = np.zeros(u_wind.shape,dtype=np.float32)

    rel_vor[1:-1,1:-1] = ( (v_wind[1:-1,2:]-v_wind[1:-1,:-2])/( (lon2_rad[:,2:]-lon2_rad[:,:-2])*re*np.cos(lat2_rad[1:-1,:])) )-\
                               ( (u_wind[2:,1:-1]-u_wind[:-2,1:-1])/( (lat2_rad[2:,:]-lat2_rad[:-2,:])*re) ) +\
                               ((u_wind[1:-1,1:-1]/re) * np.tan(lat2_rad[1:-1,:]))

    # Setup for smoothing
    if smoothing:
        m2 = 5 #convrad
        m = 2*m2+1
        y = np.zeros((m,m))
        [y1,y2] = np.meshgrid(np.arange(0,m)-m2,np.arange(0,m)-m2)
        yr = np.hypot(y1,y2)
        y[yr <= m2] = 1
        y /= y.sum()
        rel_vor = signal.fftconvolve(rel_vor,y,mode='full')[m2:-m2,m2:-m2]
    
    return rel_vor

def vorticity(u,v,lats,lons,nans_to_zeros=False):
    """
    
    ARGUMENTS:
        u = longitudinal wind [lats,lons] (m/s)
        v = latitudinal wind [lats,lons] (m/s)
        lats (deg)
        lons (deg)
        nans_to_zeros flag to change nans to zero
        
    RETURNS: zeta is transformed from m/s/deg to 1/s
        zeta, OW, OW_norm, OWZ
        scales: 
            zeta(vorticity) ~ -0.05 to +.05
            OW (similar)
            OW_norm ~ 0 to 1
            OWZ ~ also 0 to 1?
        
    NOTES:
        derivatives calculated using numpy gradient function
            >>> f = np.array([1, 2, 4, 7, 11, 16], dtype=float)
            >>> np.gradient(f)
            array([1. , 1.5, 2.5, 3.5, 4.5, 5. ])
            ## array locations can be inserted , essentially used as devision terms
            ## gradient at each point is mean of gradient on either side
            ## eg: 2 dim array
            np.gradient(np.array([[1, 2, 6], [3, 4, 5]], dtype=float))
                [array([[ 2.,  2., -1.], [ 2.,  2., -1.]]), # first output is along rows
                array([[1. , 2.5, 4. ], [1. , 1. , 1. ]])] # second output is along columns
            # This method has a dummy test in tests.py -> vorticity_test()
        Hi Jesse,
        Vorticity = zeta = v_x - u_y (where v_x = dv/dx. u_y = du/dy).
        Shearing deformation = F = v_x + U_y
        Stretching deformation = E = u_x - v_y
        OW = zeta^2 - (E^2 + F^2)        Here "^2" means squared
        OW_norm = OW/zeta^2
        OWZ = OW/zeta
        Try plotting zeta, OW_norm and OWZ.  They should all be insightful.
        Cheers,
        Kevin.
        
    """
    lat_deg_per_metre = 1/111.32e3 # 111.32km per degree
    lat_mean = np.mean(lats)
    lon_deg_per_metre = lat_deg_per_metre * np.cos(np.deg2rad(lat_mean))
    
    mlats = lats / lat_deg_per_metre # convert lats into metres
    mlons = lons / lon_deg_per_metre # convert lons into metres
    
    # u[lat,lon]
    u_lat, u_lon = np.gradient(u,mlats,mlons)
    v_lat, v_lon = np.gradient(v,mlats,mlons)
    # u is left to right (longitudinal wind)
    # v is south to north (latitudinal wind)
    zeta = v_lon - u_lat
    F = v_lon + u_lat
    E = u_lon - v_lat
    OW = zeta**2 - (E**2 + F**2)
    # ignore div by zero warning
    with np.errstate(divide='ignore',invalid='ignore'):
        zetanan = np.copy(zeta)
        zetanan[np.isclose(zeta,0)]=np.NaN
        OW_norm = OW/(zetanan**2)
        OWZ = OW/zetanan
    # fix nans to zero
    if nans_to_zeros:
        OWZ[np.isnan(OWZ)]=0 
        OW_norm[np.isnan(OW_norm)]=0 
        
    return zeta, OW, OW_norm, OWZ

def wind_dir_from_uv(u,v):
    """
        input u and v cubes, to have trig applied to them
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

def wind_speed_from_uv_cubes(u,v):
    """
    returns wind direction as a cube
    """
    return iris.analysis.maths.apply_ufunc(np.hypot,u,v)

def wind_speed_to_linewidth(speed, lwmin=0.5, lwmax=5, speedmax=None):
    ''' 
    Function to convert windspeed into a sensible linewidth
        returns lwmin + (lwmax-lwmin)*(speed/speedmax)
    if speedmax is set and lower than max(speed), speed[speed>speedmax]=speedmax
    ARGUMENTS:
    '''
    cpspeed=np.copy(speed)
    if speedmax is not None:
        cpspeed[speed>speedmax]=speedmax
    else:
        speedmax=np.nanmax(speed)
    return lwmin + (lwmax-lwmin)*(speed / speedmax)
