# -*- coding: utf-8 -*-

# 31/03/2021
# Vertical cross-section script, along the line.
# This is based on Jeff's script (plot_xsec2_jdk.py), save interpolated
# data into separate *.nc file that is later used for plotting.

# Used for ACCESS-fire case studies

from netCDF4 import Dataset
import numpy as np
import argparse
#import scipy.signal as sig
import scipy.io as sio
from scipy import interpolate

def copyattrs(ncv1,ncv2):
    ats = ncv1.ncattrs()
    for at in ats:
        ncv2.setncattr(at,ncv1.getncattr(at))
    return 0

# ===============================================================
# Interpolation function
def xsec_interp(lon,lat,data,z,lont,latt,topog,end1,end2,nx=100):
    """Get a cross-section of some model data on model levels
    
    lon,lat: grid that data is on
    data: 3-d data array on model levels
    z: 3-d array of physical height, same grid as data
    lont,latt: grid that topography is on
    topog: 2-d array of topography
    end1=(lon1,lat1): left-hand endpoint of section
    end2=(lon2,lat2): right-hand endpoint of section
    nx: no of points used for horizontal interpolation to slice
    
    return values: interpolated data to save and plot

    Author: Based on Jeff Kepert's plot_xsec function (involves plotting).
    """

    lon1,lat1=end1
    lon2,lat2=end2
    nz = data.shape[0]

    # Define grid for horizontal interpolation. x increases from 0 to 1 along the
    # desired line segment
    x_plane = np.linspace(0.0,1.0,nx)
    lon_plane = lon1 + (lon2-lon1)*x_plane
    lat_plane = lat1 + (lat2-lat1)*x_plane
    
    # Interpolate topography along slice
    f = interpolate.RectBivariateSpline(lont,latt,topog.transpose())
    topog_plane = f.ev(lon_plane,lat_plane)
    
    # Physical height along slice, model height coordinates. Note that the
    # transpose is needed because RectBivariateSpline assumes the axis order (x,y)
    z_plane = np.tile(np.nan, [nlev,nx])
    for k in range(0,nlev):
        f = interpolate.RectBivariateSpline(lont,latt,z[k,:,:].transpose())
        z_plane[k,:] = f.ev(lon_plane,lat_plane)

    # Interpolate data along slice (in model height coordinate). Note that the
    # transpose is needed because RectBivariateSpline assumes the axis order (x,y)
    data_plane = np.tile(np.nan, [nlev,nx])
    for k in range(0,nlev):
        f = interpolate.RectBivariateSpline(lont,latt,data[k,:,:].transpose())
        data_plane[k,:] = f.ev(lon_plane,lat_plane)

    # Reality check on the interpolation, eventually remove
    i1 = np.argmin(np.abs(lont-lon1))        
    i2 = np.argmin(np.abs(lont-lon2))        
    j1 = np.argmin(np.abs(latt-lat1))        
    j2 = np.argmin(np.abs(latt-lat2))        
    print('Nearest neighbour vs interp topog:')
    print('   {:9.2f} {:9.2f}'.format(topog_plane[0 ],topog[j1,i1]))
    print('   {:9.2f} {:9.2f}'.format(topog_plane[-1],topog[j2,i2]))
    
    i1 = np.argmin(np.abs(lon-lon1))        
    i2 = np.argmin(np.abs(lon-lon2))        
    j1 = np.argmin(np.abs(lat-lat1))        
    j2 = np.argmin(np.abs(lat-lat2))        
    print('Nearest neighbour vs interp data:')
    print('   {:9.2f} {:9.2f}'.format(data_plane[0,0 ],data[0,j1,i1]))
    print('   {:9.2f} {:9.2f}'.format(data_plane[0,-1],data[0,j2,i2]))

    return x_plane, lon_plane, lat_plane, topog_plane, z_plane, data_plane
# ===============================================================

# Process command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('field',help='field to process (e.g. spd)')
args = parser.parse_args()

# Corryong fire 
datadir = '/g/data/en0/dzr563/ACCESS-fire/green_valley2/2021-05-03/20191230T0300Z/0p3/atmos/'
datadir_out = '/g/data/en0/dzr563/green_valley2_nc_data/'

time_int = '2019123006'
time_start = '2019123003'

for fld in [args.field]:  #'u','v','w','temp', 'p'
    if fld == 'w':
        infile1 = 'umnsaa_' + time_int + '_mdl_th1.nc'
        infile = 'umnsaa_' + time_start + '_mdl_th1.nc'
        invar = 'vertical_wnd'
        outvar = 'w_plane'
        outfile = 'w_' + time_int + '_CS1_plane_300m_NWSE_testing.nc'
        outvar_units = 'm/s'
    elif fld == 'temp':
        infile1 = 'umnsaa_' + time_int + '_mdl_th1.nc'
        infile = 'umnsaa_' + time_start + '_mdl_th1.nc'
        invar = 'air_temp'
        outvar = 'temp_plane'
        outfile = 'temp_' + time_int + '_CS1_plane_300m_NWSE.nc'
        outvar_units = 'K'
    elif fld == 'p':
        infile1 = 'umnsaa_' + time_int + '_mdl_th1.nc'
        infile = 'umnsaa_' + time_start + '_mdl_th1.nc'
        invar = 'pressure'
        outvar = 'p_plane'
        outfile = 'p_' + time_int + '_CS1_plane_300m_NWSE.nc'
        outvar_units = 'Pa'
    elif fld == 'u':
        infile1 = 'umnsaa_' + time_int + '_mdl_ro1_reg.nc'
        infile = 'umnsaa_' + time_start + '_mdl_ro1.nc'
        invar = 'x-wind'
        outvar = 'u_plane'
        outfile = 'u_' + time_int + '_CS1_plane_300m_NWSE_testing.nc'
        outvar_units = 'm/s'
    elif fld == 'v':
        infile1 = 'umnsaa_' + time_int + '_mdl_ro1_reg.nc'
        infile = 'umnsaa_' + time_start + '_mdl_ro1.nc'
        invar = 'y-wind'
        outvar = 'v_plane'
        outfile = 'v_' + time_int + '_CS1_plane_300m_NWSE.nc'
        outvar_units = 'm/s'

    # Read physical ht of model levels, topography, lat, lon
    ncfile = Dataset(datadir+infile,'r')

    nlon = len(ncfile.dimensions['longitude'])
    nlat = len(ncfile.dimensions['latitude'])
    nlev = len(ncfile.dimensions['model_level_number'])

    lon  = ncfile.variables['longitude'][:]
    lat  = ncfile.variables['latitude' ][:]
    zlev = ncfile.variables['level_height'][:]
    if fld == 'u' or fld =='v':
        z = ncfile.variables['height_rho'][:]
    else:
        z = ncfile.variables['height_theta'][:]

    topog = ncfile.variables['topog'][:,:]
    lont = lon
    latt = lat
    ncfile.close()

    # Read correct time data
    if fld =='u' or fld=='v':
        ncfile = Dataset(datadir_out + infile1,'r')
        nt   = len(ncfile.dimensions['time'])
        t    = ncfile.variables['time'        ][:]
    else:
        ncfile = Dataset(datadir + infile1,'r')
        nt   = len(ncfile.dimensions['time'])
        t    = ncfile.variables['time'        ][:]
# ===============================================================

    # Set up output file
    ncoutf = Dataset(datadir_out+outfile,'w',format='NETCDF3_CLASSIC')
    print("Initialising u")
    # Dimensions
    d_t     = ncoutf.createDimension('time',        None)
    d_ht    = ncoutf.createDimension('height',nlev)
    d_xp     = ncoutf.createDimension('x_plane', 100)

    # Dimension variables
    v_t   = ncoutf.createVariable('time',        'f8',('time'))
    v_ht  = ncoutf.createVariable('height', 'f4',('height'))
    v_xp = ncoutf.createVariable('x_len', 'f4',('x_plane'))

    copyattrs(ncfile.variables['level_height'],v_ht)
    copyattrs(ncfile.variables['time'        ],v_t)

    v_ht[:]  = zlev.astype(np.float32)
    v_ht.setncattr('long_name','Interpolated height')
    v_ht.setncattr('units','m')

    # Variables
    x_plane = ncoutf.createVariable('x_plane','f8',('time', 'x_plane'))
    x_plane.setncattr('name','x_plane')
    lon_plane = ncoutf.createVariable('lon_plane','f8',('time', 'x_plane'))
    lon_plane.setncattr('name','lon_plane')
    lat_plane = ncoutf.createVariable('lat_plane','f8',('time', 'x_plane'))
    lat_plane.setncattr('name','lat_plane')

    z_plane = ncoutf.createVariable('z_plane','f8',('time', 'height', 'x_plane'))
    z_plane.setncattr('name','z_plane')
    z_plane.setncattr('units','m')

    topog_plane = ncoutf.createVariable('topog_plane','f8',('time', 'x_plane'))
    topog_plane.setncattr('name','topog_plane')
    topog_plane.setncattr('units','m')

    v_t = ncoutf.variables['time']
    z_plane = ncoutf.variables['z_plane']
    topog_plane = ncoutf.variables['topog_plane']

    print("creating " + outvar)
    data_plane = ncoutf.createVariable(outvar,'f8',('time','height', 'x_plane'))
    data_plane.setncattr('name',outvar)
    data_plane.setncattr('units',outvar_units)
    
    ncfile.close()
    print('Finished setup of output files')
# ===============================================================

# Call interpolation function and save data
    
    nx=100
    data_int = np.tile(np.nan, [nlev,nx])
    z_int = np.tile(np.nan, [nlev,nx])
    topog_int = np.tile(np.nan, [nlev,nx])
    x_int = np.tile(np.nan, [nx])
    lat_int = np.tile(np.nan, [nx])
    lon_int = np.tile(np.nan, [nx])

    #==========================================
    # Cross-section start and end points
    # Green Valley
    # NW-SE, CS1
    end1 = (147.5,-35.85)
    end2 = (147.8,-36.05)
    #==========================================

    for txout,tx in enumerate(range(0,nt)):

        print('time index = {:d}'.format(tx))
        if fld=='u' or fld=='v':
            ncfile = Dataset(datadir_out + infile1, 'r')

        else:
            ncfile = Dataset(datadir + infile1, 'r')
        data = ncfile.variables[invar][tx,:,:,:]  # time, ht, lat, lon

        x_int,lon_int,lat_int,topog_int,z_int,data_int = xsec_interp(lon,lat,data,z,lont,latt,topog,end1,end2)

        v_t[txout] = t[tx].astype(np.float64)
        data_plane[txout,:,:] = data_int.astype(np.float64)
        z_plane[txout,:,:] = z_int.astype(np.float64)
        topog_plane[txout,:] = topog_int.astype(np.float64)
        x_plane[txout,:] = x_int.astype(np.float64)
        lon_plane[txout,:] = lon_int.astype(np.float64)
        lat_plane[txout,:] = lat_int.astype(np.float64) 
        ncoutf.sync()
        ncfile.close()

    ncoutf.close()







    









