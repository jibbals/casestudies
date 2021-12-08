# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 12:03:38 2021

@author: jgreensl
"""

import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter #,FormatStrFormatter,LinearLocator, LogFormatter
#import matplotlib.patheffects as PathEffects

import numpy as np
import warnings
from datetime import datetime,timedelta
#from scipy import interpolate
import xarray as xr

# local modules
from utilities.plotting import quiverwinds, _cmaps_
from utilities.utils import distance_between_points
from utilities import constants# ,plotting, utils
#from utilities import fio_iris as fio

###
## GLOBALS
###

# {"model run": {"transect name" : [lat0,lon0,lat1,lon1] ,}, }
list_of_transects={
    "badja_am1":{ 
        "Cobargo_Plume_NWSE":[-36.26,149.675, -36.4,150.0],
        },
    "KI_run2":{},
    "badja_run3":{},
    }

###
## METHODS
###



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



def transect_interp(data, lats, lons, start, end, nx=None, z=None,
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

def transect_ticks_labels(start,end, metres=True):
  '''
    return xticks and xlabels for a cross section
  '''
  lat1,lon1=start
  lat2,lon2=end
  # Set up a tuple of strings for the labels. Very crude!
  xticks = (0.0,0.5,1.0)
  if metres:
      xlen = distance_between_points(start,end)
      xticks = (0, 0.5*xlen, xlen)
  fmt = '{:.1f}S {:.1f}E'
  xlabels = (fmt.format(-lat1,lon1),fmt.format(-0.5*(lat1+lat2),0.5*(lon1+lon2)),fmt.format(-lat2,lon2))
  return xticks,xlabels


def transect_winds_interp(u,v,lats,lons,start,end,nx=None,z=None):
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

    ucross_str=transect_interp(u,lats,lons,
                    start=[lat0,lon0],
                    end=[lat1,lon1],
                    nx=nx,
                    z=z)
    ucross = ucross_str['transect']
    vcross_str=transect_interp(v,lats,lons,
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


def plot_transect_winds(u,v,w,z,
                  lats,lons,
                  transect,
                  ztop=5000,
                  npoints=None,
                  n_arrows=20,
                  wind_contourargs={},
                  ):
    """
    Plot transect winds: uses utils.transect_winds to get winds along transect plane
    
    ARGUMENTS:
        u,v,w,z: arrays [lev,lats,lons] of East wind, N wind, Z wind, and level altitudes
        lats,lons: dims, in degrees
        transect: [[lat0, lon0], [lat1,lon1]]
        topog: [lats,lons] surface altitude array
        ztop: top altitude to look at, defualt 5000m
    """
    retdict = {} # return info for whatever use
    
    # First we subset all the arrays so to be below the z limit
    zmin = np.min(z,axis=(1,2)) # lowest altitude on each model level
    ztopi = np.argmax(ztop<zmin)+1 # highest index where ztop is less than model level altitude
    u,v,w,z = [u[:ztopi],v[:ztopi],w[:ztopi],z[:ztopi]]
    
    # interpolation points
    start,end = transect
    if npoints is None:
        npoints=number_of_interp_points(lats,lons,start,end)
    
    # vertical wind speed along transect
    transect_w_struct = transect_interp(w,lats,lons,start,end,nx=npoints,z=z)
    transect_w = transect_w_struct['transect']
    
    # transect direction left to right wind speed
    transect_winds_struct = transect_winds_interp(u,v,lats,lons,start,end,nx=npoints,z=z)
    transect_s = transect_winds_struct['transect_wind']
    slicex = transect_winds_struct['x']
    slicez = transect_winds_struct['y']
    # scale for Y axis based on axes
    Yscale=ztop/transect_winds_struct['xdistance'][-1]
    
    retdict['xlabel'] = transect_winds_struct['xlabel']
    # left to right wind speed along transect
    retdict['s'] = transect_s
    # vertical wind motion along transect
    retdict['w'] = transect_w
    # x and y axis coordinates along transect
    retdict['x'] = slicex
    retdict['y'] = slicez
    # east to west, south to north wind speeds along transect
    retdict['u'] = transect_winds_struct['transect_u']
    retdict['v'] = transect_winds_struct['transect_v']
    retdict['yscale'] = Yscale
    quiverwinds(slicez,slicex,transect_s,transect_w,
                n_arrows=n_arrows,
                add_quiver_key=False,
                alpha=0.5,
                )
    plt.xlim(np.nanmin(slicex),np.nanmax(slicex))
    plt.ylim(np.nanmin(slicez),ztop)
    
    return retdict



###
## PLOTTING METHODS
###

def plot_transect(data, z, lat, lon, start, end, npoints=None, 
             topog=None, sh=None, latt=None, lont=None, ztop=4000,
             title="", ax=None, colorbar=True,
             lines=None, contours=None,
             cbar_args={},
             **contourfargs):
    '''
    Draw cross section
    ARGUMENTS:
        data is 3d [levs,lats,lons]
        z (3d): height [lev,lats,lons]
        lat(1d), lon(1d): data coordinates
        start, end: [lat0,lon0], [lat1,lon1]
        lines (list): add black lines to contourf
        cbar_args = dict of options for colorbar, drawn if colorbar==True
    return slicedata, slicex, slicez, cmappable (the colorbar)
    '''
    ## Default contourfargs
    if 'extend' not in contourfargs:
        contourfargs['extend'] = 'max'

    ## Check that z includes topography (within margin of 40 metres)
    if topog is not None:
        if np.mean(z[0]+40)<np.mean(topog):
            print("ERROR:",np.mean(z[0]), np.min(z[0]), "(mean,lowest z) is lower than topog", np.mean(topog), np.min(topog))
            print("ERROR:", "Try adding topog to each level of z")
            assert False
    
    if npoints is None:
        npoints = number_of_interp_points(lat,lon,start,end)
        
    # Transect slice: data[z,x] x[z,x], z[z,x]
    # struct: {
    #         'transect': vertical cross section of data 
    #         'x': x axis [Y,X] in metres from start point 
    #         'y': y axis [Y,X] in terms of z_th
    #         'lats': [X] lats along horizontal axis
    #         'lons': [X] lons along horizontal axis
    #     } 
    transect_struct = transect_interp(data,lat,lon,start,end,nx=npoints, z=z)
    slicedata = transect_struct['transect']
    # X axis [0,1,...]
    X = transect_struct['x']
    # heights along transect [x,y]
    Y = transect_struct['y']
    
    # Pull out cross section of topography and height
    if latt is None:
        latt=lat
    if lont is None:
        lont=lon
    
    if ax is not None:
        plt.sca(ax)
    # Note that contourf can work with non-plaid coordinate grids provided both are 2-d
    # Contour inputs: xaxis, yaxis, data, colour gradient 
    if contours is not None and 'levels' not in contourfargs:
        contourfargs['levels']=contours
    cmappable=plt.contourf(X,Y,slicedata,**contourfargs)
    
    if colorbar:
        # defaults if not set
        #orientation 	vertical or horizontal
        #fraction 	0.15; fraction of original axes to use for colorbar
        #pad 	0.05 if vertical, 0.15 if horizontal; fraction of original axes between colorbar and new image axes
        #shrink 	1.0; fraction by which to multiply the size of the colorbar
        #aspect 	20; ratio of long to short dimensions
        if 'pad' not in cbar_args:
            cbar_args['pad']=0.01
        if 'aspect' not in cbar_args:
            cbar_args['aspect']=30
        if 'shrink' not in cbar_args:
            cbar_args['shrink'] = 0.85
        if 'fraction' not in cbar_args:
            cbar_args['fraction'] = .075
        plt.colorbar(**cbar_args)
    
    # Add contour lines
    if lines is not None:
        plt.contour(X,Y,slicedata,lines,colors='k')
    
    zbottom = np.tile(np.min(Y),reps=npoints) # xcoords
    xbottom = X[0,:]
    # make sure land is obvious
    if topog is not None:
        slicetopog_struct = transect_interp(topog,latt,lont,start,end,nx=npoints)
        slicetopog = slicetopog_struct['transect']
        # Plot gray fill unless we have sensible heat
        if (sh is not None):
            if (np.max(sh) < 1):
                plt.fill_between(xbottom, slicetopog, zbottom, 
                                 interpolate=True, facecolor='darkgrey',
                                 zorder=2)
            else:
                # 0-50k W/m2 colormap on log scale
                cmap=plt.get_cmap('plasma')
                normalize = matplotlib.colors.SymLogNorm(vmin=0, vmax=10000, linthresh=100, base=10.0)
                
                sh_trans_struct = transect_interp(sh, lat, lon, start, end, nx=npoints)
                shslice = sh_trans_struct['transect']
                # colour a sequence of polygons with the value coming from sh
                for i in range(len(shslice) - 1):
                    if shslice[i]<100:
                        color='darkgrey'
                    else:
                        color=cmap(normalize(shslice[i]))
                    plt.fill_between([xbottom[i], xbottom[i+1]], 
                                     [slicetopog[i],slicetopog[i+1]],
                                     [zbottom[i], zbottom[i+1]], 
                                     color=color,
                                     zorder=2) # put on top of most things
    
    if ztop is not None:
        plt.ylim(0,ztop)
    
    plt.xticks([])
    plt.xlabel('')
    plt.title(title)

    return slicedata, X, Y, cmappable

def plot_transect_s(s, z, lat, lon, start, end, npoints=100, 
               topog=None, sh=None, latt=None, lont=None, ztop=4000,
               title="Wind speed (m/s)", ax=None, colorbar=True,
               contours=np.arange(0,25,2.5),
               lines=np.arange(0,25,2.5), 
               cbar_args={},
               **contourfargs):
    '''
    Draw wind speed cross section
        s is 3d wind speed
        z(3d), lat(1d), lon(1d) is height (m), lats and lons
        start, end are [lat0,lon0], [lat1,lon1]
        contours will be filled colours
        lines will be where to draw black lines
    '''
    ## default cmap
    if 'cmap' not in contourfargs:
        contourfargs['cmap']=_cmaps_['windspeed']
    
    # wind speed
    s[np.isnan(s)] = -5000 # There is one row or column of s that is np.NaN, one of the edges I think
    
    # call transect using some defaults for potential temperature
    return plot_transect(s,z,lat,lon,start,end,npoints=npoints,
                    topog=topog, sh=sh, latt=latt, lont=lont, ztop=ztop,
                    title=title, ax=ax, colorbar=colorbar,
                    contours=contours,lines=lines,
                    cbar_args=cbar_args,
                    **contourfargs)

def plot_transect_theta(theta, z, lat, lon, start, end, npoints=None, 
                   topog=None, sh=None, latt=None, lont=None, ztop=4000,
                   title="$T_{\\theta}$ (K)", ax=None, colorbar=True,
                   contours = np.arange(280,350,1),
                   lines = np.union1d(np.arange(280,301,2), np.arange(310,351,10)),
                   cbar_args={},
                   **contourfargs):
    '''
    Draw theta cross section
        theta is 3d potential temperature
        z(3d), lat(1d), lon(1d) is height (m), lats and lons
        start, end are [lat0,lon0], [lat1,lon1]
        contours will be filled colours
        lines will be where to draw black lines
    '''
    if 'cmap' not in contourfargs:
        contourfargs['cmap'] = _cmaps_['th']
    if 'norm' not in contourfargs:
        contourfargs['norm'] = matplotlib.colors.SymLogNorm(300,base=np.e)
    if 'format' not in cbar_args:
        cbar_args['format']= ScalarFormatter()
    # call transect using some defaults for potential temperature
    return plot_transect(theta,z,lat,lon,start,end,npoints=npoints,
                    topog=topog, sh=sh, latt=latt, lont=lont, ztop=ztop,
                    title=title, ax=ax, colorbar=colorbar,
                    contours=contours, lines=lines,
                    cbar_args=cbar_args,
                    **contourfargs)

def plot_transect_w(w, z, lat, lon, start, end, npoints=None, 
               topog=None, sh=None, latt=None, lont=None, ztop=5000,
               title="Vertical motion (m/s)", ax=None, colorbar=True, 
               contours=np.union1d(np.union1d(2.0**np.arange(-2,6),-1*(2.0**np.arange(-2,6))),np.array([0])),
               lines=np.array([0]),
               cbar_args={},
               **contourfargs):
    '''
    Draw theta cross section
        w is 3d vertical motion
        z(3d), lat(1d), lon(1d) is height (m), lats and lons
        start, end are [lat0,lon0], [lat1,lon1]
        contours will be filled colours
        lines will be where to draw black lines
    '''
    if 'cmap' not in contourfargs:
        contourfargs['cmap'] = _cmaps_['verticalvelocity']
    if 'norm' not in contourfargs:
        contourfargs['norm'] = matplotlib.colors.SymLogNorm(0.25,base=2.0)
    if 'format' not in cbar_args:
        cbar_args['format'] = ScalarFormatter()
        
    # call transect using some defaults for vertical velocity w
    return plot_transect(w, z,lat,lon,start,end,npoints=npoints,
                    topog=topog, sh=sh, latt=latt, lont=lont, ztop=ztop,
                    title=title, ax=ax, colorbar=colorbar,
                    contours=contours, lines=lines,
                    cbar_args=cbar_args,
                    **contourfargs)

def transect_qc(qc, z, lat, lon, start, end, npoints=None, 
               topog=None, sh=None, latt=None, lont=None, ztop=4000,
               title="Water and ice (g/kg air)", ax=None, colorbar=True,
               contours=np.arange(0.0,0.4,0.01),
               lines=np.array([constants.cloud_threshold]),
               cbar_args={},
               **contourfargs):
    '''
    Draw theta cross section
        qc is 3d vertical motion
        z(3d), lat(1d), lon(1d) is height (m), lats and lons
        start, end are [lat0,lon0], [lat1,lon1]
        contours will be filled colours
        lines will be where to draw black lines
    '''
    # defaults for contourfargs
    if 'cmap' not in contourfargs:
        contourfargs['cmap'] = _cmaps_['qc']
    if 'norm' not in contourfargs:
        contourfargs['norm'] = matplotlib.colors.SymLogNorm(0.02, base=2.0)
    if 'format' not in cbar_args:
        cbar_args['format'] = ScalarFormatter()
    # call transect using some defaults for vertical velocity w
    return plot_transect(qc, z,lat,lon,start,end,npoints=npoints,
                    topog=topog, sh=sh, latt=latt, lont=lont, ztop=ztop,
                    title=title, ax=ax, colorbar=colorbar,
                    contours=contours, lines=lines,
                    cbar_args=cbar_args,
                    **contourfargs)

def add_contours(data, z, lat, lon, start, end, npoints=None, 
                 ztop=None, lines=None, 
                 **contourargs):
        '''
        Draw cross section
        ARGUMENTS:
            data is 3d [levs,lats,lons]
            z (3d): height [lev,lats,lons]
            lat(1d), lon(1d): data coordinates
            start, end: [lat0,lon0], [lat1,lon1]
            lines (list): add black lines to contourf
            cbar_args = dict of options for colorbar, drawn if colorbar==True
        return slicedata, slicex, slicez, cmappable (the colorbar)
        '''
        
        # assume adding contours to existing plot
        if ztop is None:
            ztop = plt.gca().get_ylim()[1]
            
            
        if npoints is None:
            npoints = number_of_interp_points(lat,lon,start,end)
        
        # default contour args
        if 'colors' not in contourargs:
            contourargs['colors']='k'
        
        # calculate transect of metric
        transect_struct = transect_interp(data,lat,lon,start,end,nx=npoints, z=z)
        slicedata = transect_struct['transect']
        # X axis [0,1,...]
        X = transect_struct['x']
        # heights along transect [x,y]
        Y = transect_struct['y']
        
        # Note that contourf can work with non-plaid coordinate grids provided both are 2-d
        # Contour inputs: xaxis, yaxis, data, colour gradient 
        
        # Add contour lines
        if lines is not None:
            plt.contour(X,Y,slicedata,lines,**contourargs)
        
        # reset ztop
        plt.ylim(0,ztop)
        
        return transect_struct