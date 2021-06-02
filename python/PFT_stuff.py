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

from timeit import default_timer as timer
from utilities import fio_iris as fio
from utilities import utils
from utilities import PFT as PFT_calc



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
        cubedtimes = utils.dates_from_iris(cubes.extract('u')[0])
    
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
                for thing in [uu,vv,ww,qq,TT,th,pr]:
                    print(type(thing),np.shape(thing))
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
    print("Info: time to produce PFT(%s): %.2f minutes"%(str(PFT.shape), (end-start)/60.0))
    return PFT


def read_PFT_timeseries(mr,latlon,
                        force_recreate=False,
                        interp_method="nearest",):
    lat,lon = latlon
    extent=[lon-.01, lon+.01, lat-.01, lat+.01] # WESN
    
    fname = "../data/PFT/"+mr+str(lat)+","+str(lon)+".nc"
    if os.path.isfile(fname) and not force_recreate:
        print("INFO: Reading already created netcdf timeseries:",fname)
        return iris.load(fname)
    
    
    # let's time how long it takes
    start = timer()
    hours = fio.hours_available(mr)
    
    PFT_full=[]
    dtimes=[]
    cubes=None
    topog=fio.read_topog(mr,extent=extent)
    for hour in hours:
        del cubes
        # Read the cubes for one hour at a time
        cubes = fio.read_model_run(mr, hours=[hour], extent=extent,)
        utils.extra_cubes(cubes,
                          add_z=True, 
                          add_RH=True,
                          #add_topog=True, 
                          add_winds=True,
                          add_theta=True,
                          )
        cubes.append(topog) # add topog to CubeList
        print(cubes)
        PFT_full.append(PFT_from_cubelist(cubes, latlon=latlon))
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
    dim_coords = [(b,a) for [a,b] in enumerate(Pa.dim_coords)]
    # update time dimension
    dim_coords[0] = (timecoord,0)
    # create cube from PFT array
    PFTcube = iris.cube.Cube(PFT,
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
    #ddir=fio.run_info[mr]['dir']
    fname= "data/PFT/"+mr+".nc"
    fio.make_folder(fname)
    iris.save(PFTcube,fname)
    
    
    end = timer()
    print("Info: time to read model run and calculate %s : %.2f minutes"%(fname, (end-start)/60.0))
    
    # test file:
    f = iris.load(fname)
    print("INFO: SAVED ",fname)
    print(f)
    return f

if __name__ == "__main__":
    read_PFT_timeseries("KI_run2_exploratory", latlon=[-35.8,137.3])
    