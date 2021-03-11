#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 15:37:41 2021
    Vertical cross sections PLUS top down contextual map
@author: jesse
"""

import matplotlib
#matplotlib.use('Agg')

# plotting stuff
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter,LinearLocator, LogFormatter
import matplotlib.patheffects as PathEffects
import numpy as np
import warnings
from datetime import datetime,timedelta
#from scipy import interpolate
#import cartopy.crs as ccrs

# local modules
from utilities import plotting, utils, fio, constants

###
## GLOBALS
###
_sn_ = 'cross_sections'


firefront_centres = {
    'KI_run1':{
        ## where to centre the transect and UTC time
        'latlontimes':[
            ## Start near poly ignition front
            [-35.805,136.73,datetime(2020,1,2,21,00)], 

            ## End near southern coastline around lunch time ()
            [-36.02,136.79,datetime(2020,1,3,3,00)], 
            ],
        },
}
## Duplicates
#_emberstorm_centres_['waroona_run3_1p0']=_emberstorm_centres_['waroona_run3']


def interp_centres(latlontimes, outtimes, 
                   dx=0.1,
                   dy=0,
                   ):
    """
        return list of latlons, interpolated over input times
        ARGUMENTS:
            latlontimes: (lat,lon,datetime)
            outtimes: datetimes over which to stretch the latlontime sequence
            dx: start end points will be centres -+ this lon
            dy: start end points will be centres -+ this lat
    """
    
    lats = [lat for lat,_,_ in latlontimes]
    lons = [lon for _,lon,_ in latlontimes]
    intimes = [time for _,_,time in latlontimes]
    
    # X is hours since 2015
    # interpolate lats and lons onto new list of datetimes
    in_X = [(dt - datetime(2015,1,1)).total_seconds()/3600.0 for dt in intimes]
    out_X = [(dt - datetime(2015,1,1)).total_seconds()/3600.0 for dt in outtimes]
    out_lats = np.interp(out_X,in_X, lats, left=lats[0], right=lats[-1])
    out_lons = np.interp(out_X,in_X, lons, left=lons[0], right=lons[-1])
    #out_latlons = [ (lat, lon) for lat,lon in zip(out_lats, out_lons) ]
    
    out_transects = [[[lat-dy,lon-dx],[lat+dy,lon+dx]] for lat,lon in zip(out_lats,out_lons)]
    
    return out_transects

###
## METHODS
###

def add_vertical_contours(w,lat,lon,
                          wmap_height=300, wmap_levels=[1,3],
                          annotate=True, 
                          xy0=[.7,-.02], 
                          xy1=None):
    '''
    ARGS:
        w [lat,lon] : map of vertical motion
        lat,lon : degrees
        wmap_height : w map altitude
        annotate {True | False} : add annotation?
        xy0: annotate text added at xy0 and xy0 + [0,-.06]
    '''
    
    with warnings.catch_warnings():
        # ignore warning when there are no contours to plot:
        warnings.simplefilter('ignore')
        # downward motion in blueish
        plt.contour(lon, lat, -1*w, levels=wmap_levels, 
                    linestyles=['dashed','solid'],
                    colors=('aquamarine',))
        # upward motion in pink
        plt.contour(lon, lat, w, levels=wmap_levels, 
                    linestyles=['dashed','solid'],
                    colors=('pink',))
    if xy1 is None:
        xy1=[xy0[0], xy0[1]-.03]
    # plt.annotate(
    #     'vertical motion at %dm altitude'%wmap_height, 
    #     xy=xy0, 
    #     xycoords='axes fraction', 
    #     fontsize=9
    #     )
    # plt.annotate(
    #     'dashed is %.1fm/s, solid is %.1fm/s'%(wmap_levels[0],wmap_levels[1]), 
    #     xy=xy1, 
    #     xycoords='axes fraction', 
    #     fontsize=9
    #     )


def transect_winds(u,v,w,z,
                  lats,lons,
                  transect,
                  ztop=5000,
                  wind_contourargs={},
                  ):
    """
    Plot transect wind streamplot: uses utils.transect_winds to get winds along transect plane
    
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
    npoints=utils.number_of_interp_points(lats,lons,start,end)
    
    # vertical wind speed along transect
    transect_w_struct = utils.transect(w,lats,lons,start,end,nx=npoints,z_th=z)
    transect_w = transect_w_struct['transect']
    
    # transect direction left to right wind speed
    transect_winds_struct = utils.transect_winds(u,v,lats,lons,start,end,nx=npoints,z=z)
    transect_s = transect_winds_struct['transect_wind']
    slicex = transect_winds_struct['x']
    slicez = transect_winds_struct['y']
    
    retdict['label'] = transect_winds_struct['label']
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
    
    # Streamplot
    plotting.streamplot_regridded(slicex,slicez,transect_s,transect_w,
                                  density=(1,1), 
                                  color='darkslategrey',
                                  zorder=1,
                                  #linewidth=np.hypot(sliceu,slicew), # too hard to see what's going on
                                  minlength=0.4, # longer minimum stream length (axis coords: ?)
                                  arrowsize=1.5, # arrow size multiplier
                                  )
    plt.xlim(np.min(slicex),np.max(slicex))
    plt.ylim(np.min(slicez),ztop)
    
    return retdict

def topdown_view(extent,
                 fig=None, 
                 subplot_row_col_n=None, 
                 ax=None,
                 lats=None,lons=None, ff=None, sh=None, 
                 u10=None, v10=None, 
                 wmap=None, wmap_height=None,
                 topog=None,
                 tiffname=None,
                 #annotate=False, 
                 showlatlons=True,
                 sh_kwargs={},
                 ):
    """
    Top down view of model run
    ARGUMENTS:
        ax: plotting axis, if this is provided then no backdrop is drawn (assume axis already has backdrop)
            In this case just winds/fire/etc will be overplotted
        lats/lons are 1D arrays, required if adding other stuff
        ff [lats,lons] : firefront array
        sh [lats,lons] : sensible heat flux
        u10 [lats,lons] : 10m altitude longitudinal winds
        v10 [lats,lons] : 10m altitude latitudinal winds
        topog [lats,lons] : surface altitude - can use this instead of tiff
        topog_contours : list of values to contour topography at - default 50m
        annotate: True if annotations are desired for winds and heat flux
        showlatlons: True if latlons should be added to border
    RETURNS:
        fig, ax
    """
    # if we already have an axis, assume the backdrop is provided
    if ax is None:
        if fig is None:
            xsize = 12
            ysize = 12
            if extent is not None:
                # try to guess a good size for aspect ratio
                width = extent[1]-extent[0]
                height = extent[3]-extent[2]
                if width > (1.5*height):
                    xsize=16
                if width > (2*height):
                    xsize=20
                    ysize=10
                if width < (0.75 * height):
                    ysize=16
            fig=plt.figure(figsize=(xsize,ysize))
        # first create map from tiff file unless topography passed in
        if tiffname is not None:
            fig, ax = plotting.map_tiff_qgis(
                fname=tiffname, 
                extent=extent,
                fig=fig,
                subplot_row_col_n=subplot_row_col_n,
                show_grid=True,
                aspect='equal',
                )
        elif topog is not None:
            if subplot_row_col_n is not None:
                prow,pcol,pnum=subplot_row_col_n
                ax = plt.subplot(prow,pcol,pnum)
            plotting.map_topography(extent,topog,lats,lons,
                                    cbar=False,title="")
            ax=plt.gca()
            #ax.set_aspect('equal')
            
            ## 
            plotting.map_add_locations_extent(extent)
            
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    
    if ff is not None:
        ffcolor='red' if (sh is None) else 'darkgrey'
        # add firefront
        plotting.map_fire(ff,lats,lons,colors=ffcolor)
    if sh is not None:
        # add hot spots for heat flux
        # default kwargs for sh plot
        if 'alpha' not in sh_kwargs:
            sh_kwargs['alpha']=0.6
        if 'cbar_kwargs' not in sh_kwargs:
            sh_kwargs['cbar_kwargs'] = {'label':"Wm$^{-2}$"}
        cs_sh, cb_sh = plotting.map_sensibleheat(sh,lats,lons,
                                                 colorbar=False,
                                                 **sh_kwargs)
        if cs_sh is not None:
            # if annotate:
            #     plt.annotate(text="max heat flux = %6.1e W/m2"%np.max(sh),
            #                  xy=[0,1.06],
            #                  xycoords='axes fraction', 
            #                  fontsize=12)
            # colorbar without mucking the axes
            cbar_ax = fig.add_axes([0.75, 0.9, 0.2, 0.01]) # X Y Width Height
            cb_sh = fig.colorbar(cs_sh, cax=cbar_ax, pad=0,orientation='horizontal',
                         #format=LogFormatter(),
                         ticks=[100,1000,10000,100000],
                         )
            
            #cb_sh.ax.set_xticks([100,1000,10000,100000]) 
            cb_sh.ax.set_xticklabels(['10$^2$','10$^3$','10$^4$','10$^5$'])
            
            plt.sca(ax) # reset current axis
            
        
    if u10 is not None:
        # winds, assume v10 is also not None
        s10 = np.hypot(u10,v10)
        speedmax=20 # what speed for thickest wind streams
        lwmax_winds=5 # how thick can the wind streams become
        lw10 = utils.wind_speed_to_linewidth(s10, lwmax=lwmax_winds, speedmax=speedmax)
        # higher density if using topography instead of OSM
        density=(0.6,0.5) if topog is None else (0.75,0.7)
        plt.streamplot(lons,lats,u10,v10, 
                       linewidth=lw10, 
                       color='k',
                       density=density,
                       arrowsize=2.0, # arrow size multiplier
                       )
        # if annotate:
        #     plt.annotate("10m wind linewidth increases up to %dms$^{-1}$"%(speedmax),
        #                  xy=[0,1.12], 
        #                  xycoords="axes fraction", 
        #                  fontsize=12)
        #     plotting.annotate_max_winds(s10, text="10m wind max = %5.1f m/s",
        #                                 xytext=[0,1.025])
    
    if wmap is not None:
        add_vertical_contours(wmap,lats,lons,
                              wmap_height=wmap_height,
                              wmap_levels=[1,3],
                              annotate=True,
                              xy0=[0.73,1.07])
        
    # set limits back to latlon limits
    ax.set_ylim(ylims[0],ylims[1])
    ax.set_xlim(xlims[0],xlims[1])
    # 115.8, 116.1, -32.92,-32.82
    if showlatlons:
        #xticks=np.arange(115.8,116.11,0.05)
        #plt.xticks(xticks,xticks)
        #yticks=np.arange(-32.92,-32.805,0.03)
        #plt.yticks(yticks,yticks)
        ax.xaxis.set_major_locator(LinearLocator(numticks=5))
        ax.yaxis.set_major_locator(LinearLocator(numticks=5))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    
    return fig, ax


def map_and_transects(mr, 
                      latlontimes=None,
                      dx=.3,
                      dy=.2,
                      extent=None,
                      hours=None,
                      topography=True,
                      wmap_height=300,
                      ztop=5000,
                      temperature=False,
                      HSkip=None
                      ):
    """
    show map and transects of temperature and winds
    ARGS:
        # REQUIRED:
            mr: model run we are loading 
        # OPTIONAL:
            latlontimes: [[lat,lon,datetime],...] # for transect centre
            dx: (default 0.2) transect will be centre -+ this longitude
            dy: (default 0) transects will be centre -+ this latitude
            extent: [West,East,South,North] ## can set extent to look down on manually
            hours: [integers]
            extent: [W,E,S,N]
            topography: True|False # set true to use topog for topdown view
            wmap_height: 300m # what height for topdown vertical motion contours?
            ztop: 5000, how high to do transect?
    """
    # generally inner domains are on the order of 2 degrees by 2 degrees
    # we can look at subset if we haven't zoomed in anywhere
    if (extent is None) and (HSkip is None) and ((dx+dy) > .2):
        # exploratory outputs already subset
        if 'exploratory' not in mr:
            HSkip=3

    # read topog
    topog = fio.read_topog(mr,extent=extent,HSkip=HSkip)
    lat = topog.coord('latitude').points
    lon = topog.coord('longitude').points
    topogd=topog.data if topography else None
    
    # set extent to whole space if extent is not specified
    if extent is None:
        extent = [lon[0],lon[-1],lat[0],lat[-1]]
    
    # Read model run
    simname=mr.split('_')[0]
    umdtimes = fio.hours_available(mr)
    dtoffset = fio.sim_info[simname]['UTC_offset']
    
    # hours input can be datetimes or integers
    if hours is not None:
        if not isinstance(hours[0],datetime):
            umdtimes=umdtimes[hours]
        else:
            umdtimes = hours
    
    
    # Some plot stuff
    vertmotion_contours = np.union1d(
        np.union1d(
            2.0**np.arange(-2,6),
            -1*(2.0**np.arange(-2,6))
            ),
        np.array([0]) ) / 4.0
    
    # read one model file at a time
    for umdtime in umdtimes:
        # read cube list
        cubelist = fio.read_model_run(mr, 
                                      hours=[umdtime],
                                      extent=extent,
                                      HSkip=HSkip,
                                      )
        # add temperature, height, destaggered wind cubes
        utils.extra_cubes(cubelist,
                          add_theta=True,
                          add_z=True,
                          add_winds=True,)
        theta, = cubelist.extract('potential_temperature')
        dtimes = utils.dates_from_iris(theta)
        
        # Set up list of cross section end points
        if latlontimes is None:
            latlontimes=firefront_centres[mr]['latlontimes']
        transect_list=interp_centres(latlontimes,
                                     dtimes,
                                     dx=dx,
                                     dy=dy,
                                     )
        
        # read fire
        # TODO: read this out of loop, find time index in loop
        ff, sh, u10, v10 = fio.read_fire(model_run=mr, 
                                         dtimes=dtimes, 
                                         extent=extent,
                                         HSkip=HSkip,
                                         filenames=['firefront','sensible_heat',
                                                    '10m_uwind','10m_vwind'],
                                         )
        
        # pull out bits we want
        uvw = cubelist.extract(['u','v','upward_air_velocity'])
        zcube, = cubelist.extract(['z_th'])
        
        # extra vert map at ~ 300m altitude
        
        levh = utils.height_from_iris(uvw[2])
        levhind = np.sum(levh<wmap_height)
        
        
        # for each time slice pull out potential temp, winds
        for i,dtime in enumerate(dtimes):
            for transecti, transect in enumerate(transect_list):
                
                #utcstamp = dtime.s)trftime("%b %d %H:%M (UTC)")
                ltstamp = (dtime+timedelta(hours=dtoffset)).strftime("%H:%M (LT)")
                # winds
                u,v,w = uvw[0][i].data, uvw[1][i].data, uvw[2][i].data
                s = np.hypot(u,v)
                z = zcube[i].data
                T = theta[i].data
                # fire
                ffi,shi,u10i,v10i=None,None,None,None
                if ff is not None:
                    ffi = ff[i].data 
                if sh is not None:
                    shi = sh[i].data
                if u10 is not None:
                    u10i = u10[i].data
                    v10i = v10[i].data
                #vertical motion at roughly 300m altitude
                wmap=w[levhind]
                
                start,end=transect
                
                ### First plot, topography
                fig,ax1 = topdown_view(extent=extent,
                                       subplot_row_col_n=[3,1,1], 
                                       lats=lat, 
                                       lons=lon, 
                                       topog=topogd,
                                       ff=ffi, 
                                       sh=shi, 
                                       u10=u10i, 
                                       v10=v10i,
                                       wmap=wmap, 
                                       wmap_height=wmap_height, 
                                       )
                
                ## Add transect line
                # start to end x=[lon0,lon1], y=[lat0, lat1]
                plt.plot([start[1],end[1]],[start[0],end[0], ], '--k', 
                         linewidth=2, 
                         #marker='>', markersize=7, markerfacecolor='white'
                         )
                
                ## Subplot 2, transect of potential temp
                # how many horizontal points to interpolate to
                npoints = utils.number_of_interp_points(lat,lon,start,end)
                
                ax2=plt.subplot(3,1,2)
                if temperature:
                    plotting.transect_theta(T, z, lat, lon, start, end,
                                                npoints=npoints,
                                                topog=topogd, 
                                                sh=shi,
                                                ztop=ztop,
                                                contours=np.arange(290,320),
                                                lines=None, 
                                                levels=np.arange(290,321),
                                                cmap='gist_rainbow_r',
                                                )
                    #thetaslice,xslice,zslice=trets
                else:
                    plotting.transect_s(s, z, lat, lon, start, end,
                                                npoints=npoints,
                                                topog=topogd, 
                                                sh=shi,
                                                ztop=ztop,
                                                lines=None, 
                                                cmap='Blues',
                                                )
                                                
                
                ## Add wind streams to theta contour
                retdict = transect_winds(u, v, w, z, lat, lon, transect, 
                                       ztop=ztop,
                                       )
                
                #retdict['label'] = transect_winds_struct['label']
                #retdict['s'] = transect_s
                #retdict['w'] = transect_w
                #retdict['x'] = slicex
                #retdict['y'] = slicez
                #retdict['u'] = transect_winds_struct['transect_u']
                #retdict['v'] = transect_winds_struct['transect_v']
                
                ## Finally show winds on transect
                ax3=plt.subplot(3,1,3)
                
                plotting.transect_w(w, z, lat, lon, start, end, 
                                    npoints=npoints, 
                                    topog=topogd, 
                                    sh=shi, 
                                    ztop=ztop,
                                    title="Vertical motion (m/s)", 
                                    ax=ax3, 
                                    #colorbar=True, 
                                    contours=vertmotion_contours,
                                    lines=np.array([0]),
                                    #cbar_args={},
                                    )
                # Save figure into folder with numeric identifier
                stitle="%s %s"%(mr,ltstamp)
                plt.suptitle(stitle)
                distance=utils.distance_between_points(transect[0],transect[1])
                plt.xlabel("%.3f,%.3f -> %.3f,%.3f (= %.1fkm)"%(transect[0][0],
                                                                transect[0][1],
                                                                transect[1][0],
                                                                transect[1][1], 
                                                                distance/1e3))
                #model_run, plot_name, plot_time, plt, extent_name=None,
                fio.save_fig(mr, "map_and_transect_winds", dtime, 
                             plt=plt)
            
def multiple_transects(mr,
                       hours=None,
                       extent=None,
                       start=None,
                       end=None,
                       dx=None,
                       dy=None,
                       ):
    """
    4 rows 2 columns: 
        first row: top down winds at 10m and 2500m? and transect lines
        2nd, 3rd, 4th rows: transect of heat/winds, transect of vert motion
    ARGUMENTS:
        mr: model run name
        hours: optional run for subset of model hours
        extent: subset extent
        start,end: [lat,lon] of start, end points for transect
            default is middle of plot, left to right 80% extent coverage
        dx,dy: how far to stagger transect lines
            default is no stagger for lons, 20% of extent for lats
    """
    
    # read topog
    cube_topog = fio.read_topog(mr,extent=extent)
    lats = cube_topog.coord('latitude').points
    lons = cube_topog.coord('longitude').points
    topog = cube_topog.data
    
    # set extent to whole space if extent is not specified
    if extent is None:
        extent = [lons[0],lons[-1],lats[0],lats[-1]]
    
    # Read model run
    simname=mr.split('_')[0]
    umdtimes = fio.hours_available(mr)
    dtoffset = fio.sim_info[simname]['UTC_offset']
    
    # hours input can be datetimes or integers
    if hours is not None:
        if not isinstance(hours[0],datetime):
            umdtimes=umdtimes[hours]
        else:
            umdtimes = hours
    
    # Default start/end/dx/dy
    if start is None or end is None:
        start = np.mean(lats), np.mean(lons)-0.4*(lons[-1]-lons[0])
        end = np.mean(lats), np.mean(lons)+0.4*(lons[-1]-lons[0])
    if dy is None and dx is None:
        dy=0.3 * (lats[-1]-lats[0])
        dx=0.0
    if dx is None:
        dx = 0 
    if dy is None:
        dy = 0
    
    # 3 transects, shifted by dy and dx
    transects = [ [start[0]-dy,start[1]-dx], [end[0]-dy,end[1]-dx],
                 [start, end],
                 [start[0]+dy,start[1]+dx], [end[0]+dy,end[1]+dx],
                 ]
    
    ## Loop over hours
    for umdtime in umdtimes:
        # read cube list
        cubelist = fio.read_model_run(mr, 
                                      hours=[umdtime],
                                      extent=extent,
                                      )
        # add temperature, height, destaggered wind cubes
        utils.extra_cubes(cubelist,
                          add_theta=True,
                          add_z=True,
                          add_winds=True,)
        theta, = cubelist.extract('potential_temperature')
        dtimes = utils.dates_from_iris(theta)
        
        # read fire front, sens heat, 10m winds
        ff,sh,u10,v10 = fio.read_fire(model_run=mr,
                                      dtimes=dtimes, 
                                      extent=extent,
                                      filenames=['firefront','sensible_heat',
                                                 '10m_uwind','10m_vwind'],
                                      )
        ## loop over time steps
        for ti,dtime in enumerate(dtimes):
            ### FIRST ROW: 10m HWINDS AND VWINDS, AND STREAMS
            
            
            ### NEXT 3 ROWS: Transects
            
            for transect in transects:
                
                ## SAVE FIGURE
                fio.save_fig(mr,"multiple_transects",dtime,subdir="full_extent",plt=plt)

def flux_plot(SH, lat, lon,FF=None, map_tiff_args={}):
    """
    flux over qgis map
    plotting.map_tiff_qgis is called using args from map_tiff_args
    with defaults being:
        fname="waroonaf_osm.tiff"
        extent=[based on lats,lons]
        locnames=['yarloop','waroona']
    returns fig,ax from map_tiff_qgis
    """
    # area to zoom in on for yarloop
    dlat,dlon = lat[1]-lat[0], lon[1]-lon[0]
    extent = [lon[0]-dlon,lon[-1]+dlon,lat[0]-dlat,lat[-1]+dlat]
    
    ## DEFAULTS FOR MAP TIFF
    if 'fname' not in map_tiff_args:
        map_tiff_args['fname']='waroonaf_osm.tiff'
    if 'extent' not in map_tiff_args:
        map_tiff_args['extent']=extent
    if 'locnames' not in map_tiff_args:
        map_tiff_args['locnames']=['yarloop','waroona']
    
    mtq = plotting.map_tiff_qgis(**map_tiff_args)
    
    # add sensible heat overlay
    cs,cbar = plotting.map_sensibleheat(SH,
                                        lat,lon,
                                        colorbar=True,
                                        alpha=0.8)
    return mtq

def flux_plot_hour(mr='waroona_run3', extent=None, hour=12, 
                   map_tiff_args={}):
    """
    run flux_plot over an hour of model output
    can add map_tiff_qgis arguments in map_tiff_args dict
    Default tiff to use is the waroonaf_osm.tiff
    """
    
    modelhours = fio.run_info[mr]['filedates']
    dtime=modelhours[hour]
    which = "first" if hour < 24 else "second"
    # if not defined, pull first or second tiff for backdrop
    osm="_osm" if hour > 14 else ""
    #if "fname" not in map_tiff_args:
    map_tiff_args['fname']="waroona_%s%s.tiff"%(which,osm)
    
    
    # Read front and SH
    ftimes = [dtime + timedelta(minutes=mins) for mins in np.arange(0,60.1,5)]
    FF, SH = fio.read_fire(model_run=mr, 
                           dtimes=ftimes, 
                           extent=extent, 
                           firefront=True, 
                           sensibleheat=True,
                           )
    lats,lons = FF.coord('latitude').points, FF.coord('longitude').points
    if extent is None:
        extent=[lons[0],lons[1],lats[0],lats[1]]
    
    for i,dt in enumerate(ftimes):
        lt = dt+timedelta(hours=8)
        flux_plot(SH[i].data, lats, lons, 
                  FF=FF[i].data, 
                  map_tiff_args=map_tiff_args)
        plt.title(lt.strftime("Heat flux on the %-dth at %H:%M (LT)"))
        plt.tight_layout()
        fio.save_fig(model_run=mr, script_name=_sn_, pname=dt, plt=plt, 
                     subdir=which+"/flux")
    
        

if __name__ == '__main__':
    latlontimes=firefront_centres["KI_run1"]['latlontimes']
    
    map_and_transects('KI_run1_exploratory', 
            latlontimes=latlontimes,
            hours=np.arange(6,14),
            )
