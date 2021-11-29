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
from matplotlib.ticker import FormatStrFormatter,LinearLocator, LogFormatter, ScalarFormatter
import matplotlib.patheffects as PathEffects
import numpy as np
import warnings
from datetime import datetime,timedelta
#from scipy import interpolate
#import cartopy.crs as ccrs

# local modules
from utilities import plotting, utils, constants
from utilities import fio_iris as fio

###
## GLOBALS
###
_sn_ = 'transects'


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
    Add contours to show vertical motion at one altitude on a top down map
    wmap_levels give contours in m/s
    light blue for downwards, pink for upwards 
    
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
                  npoints=None,
                  streamplot=False,
                  n_arrows=20,
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
    if npoints is None:
        npoints=utils.number_of_interp_points(lats,lons,start,end)
    
    # vertical wind speed along transect
    transect_w_struct = utils.transect(w,lats,lons,start,end,nx=npoints,z=z)
    transect_w = transect_w_struct['transect']
    
    # transect direction left to right wind speed
    transect_winds_struct = utils.transect_winds(u,v,lats,lons,start,end,nx=npoints,z=z)
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
    # Streamplot
    if streamplot:
        print("INFO: streamplotting transect winds: ")
        print("    : xdistance=%.2fm, zheight=%.2fm, SCALING VERT MOTION BY factor of %.6f"%(transect_winds_struct['xdistance'][-1],ztop,Yscale))
        plotting.streamplot_regridded(slicex,slicez,transect_s,transect_w*Yscale,
                                      density=(.5,.5), 
                                      color='darkslategrey',
                                      zorder=1,
                                      #linewidth=np.hypot(sliceu,slicew), # too hard to see what's going on
                                      minlength=0.2, # longer minimum stream length (axis coords: ?)
                                      arrowsize=1.5, # arrow size multiplier
                                      )
    else:
        #print("INFO: quiver plotting transect winds: ")
        #print("    : xdistance=%.2fm, zheight=%.2fm, SCALING VERT MOTION BY factor of %.6f"%(transect_winds_struct['xdistance'][-1],ztop,Yscale))
        #plotting.quiverwinds(slicez,slicex,transect_s,transect_w*Yscale,
        #                     n_arrows=20,
        #                     add_quiver_key=False,
        #                     alpha=0.5,
        #                     )
        plotting.quiverwinds(slicez,slicex,transect_s,transect_w,
                             n_arrows=n_arrows,
                             add_quiver_key=False,
                             alpha=0.5,
                             )
    plt.xlim(np.nanmin(slicex),np.nanmax(slicex))
    plt.ylim(np.nanmin(slicez),ztop)
    
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
                 sh_colorbar=True,
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
    
    xlims = None
    ylims = None
    
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
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
    elif topog is not None:
        if subplot_row_col_n is not None:
            prow,pcol,pnum=subplot_row_col_n
            ax = plt.subplot(prow,pcol,pnum)
        plotting.map_topography(topog,lats,lons,
                                cbar=False,title="")
        ax=plt.gca()
        #ax.set_aspect('equal')
        
        ## 
        plotting.map_add_locations_extent(extent)
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
    
    if ff is not None:
        ffcolor='red' if (sh is None) else 'k'
        # add firefront
        plotting.map_fire(ff,lats,lons,colors=ffcolor,linestyles='dashed')
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
        if (cs_sh is not None) and sh_colorbar:
            # if annotate:
            #     plt.annotate(text="max heat flux = %6.1e W/m2"%np.max(sh),
            #                  xy=[0,1.06],
            #                  xycoords='axes fraction', 
            #                  fontsize=12)
            # colorbar without mucking the axes
            cbar_ax = fig.add_axes([0.75, 0.91, 0.2, 0.01]) # X Y Width Height
            cb_sh = fig.colorbar(cs_sh, cax=cbar_ax, pad=0,orientation='horizontal',
                         #format=LogFormatter(),
                         ticks=[100,1000,10000,100000],
                         )
            cb_sh.ax.set_title("W/m$^2$")
            #cb_sh.ax.set_xticks([100,1000,10000,100000]) 
            cb_sh.ax.set_xticklabels(['10$^2$','10$^3$','10$^4$','10$^5$'])
            
            plt.sca(ax) # reset current axis

    if u10 is not None:
        # winds, assume v10 is also not None
        s10 = np.hypot(u10,v10)
        # float point error issue means streamplot fails
        #print("DEBUG: lons:",lons) 
        # higher density if using topography instead of OSM
        density=(0.6,0.5) if topog is None else (0.75,0.7)
        
        # streamplot requires regular grid
        if np.all(np.diff(lons) == lons[1]-lons[0]) and np.all(np.diff(lats) == lats[1]-lats[0]):
            speedmax=20 # what speed for thickest wind streams
            lwmax_winds=5 # how thick can the wind streams become
            lw10 = utils.wind_speed_to_linewidth(s10, lwmax=lwmax_winds, speedmax=speedmax)
            plt.streamplot(lons,lats,u10,v10, 
                       linewidth=lw10, 
                       color='k',
                       density=density,
                       arrowsize=2.0, # arrow size multiplier
                       )
        else:
            plotting.streamplot_regridded(lons,lats,u10,v10,
                    color='k', 
                    density=density, 
                    arrowsize=2.0,
                    )

        #else:
        #    xskip=int(np.max([len(lons)//25-1,1]))
        #    yskip=int(np.max([len(lats)//25-1,1]))
        #    plt.quiver(lons[::xskip],lats[::yskip],u10[::yskip,::xskip],v10[::yskip,::xskip],
        #            pivot='mid')

        # if annotate:
        #     plt.annotate("10m wind linewidth increases up to %dms$^{-1}$"%(speedmax),
        #                  xy=[0,1.12], 
        #                  xycoords="axes fraction", 
        #                  fontsize=12)
        #     plotting.annotate_max_winds(s10, text="10m wind max = %5.1f m/s",
        #                                 xytext=[0,1.025])
        
        # set limits back to latlon limits
        if xlims is not None:
            ax.set_ylim(ylims[0],ylims[1])
            ax.set_xlim(xlims[0],xlims[1])
    
    if wmap is not None:
        add_vertical_contours(wmap,lats,lons,
                              wmap_height=wmap_height,
                              wmap_levels=[1,3],
                              annotate=True,
                              xy0=[0.73,1.07])
        
    
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

def topdown_view_only(mr, 
                      extent=None,
                      subdir=None,
                      hours=None,
                      topography=True,
                      wmap_height=300,
                      HSkip=None,
                      ):
    """
    show map of 10m winds over topography
    ARGS:
        # REQUIRED:
            mr: model run we are loading 
        # OPTIONAL:
            extent: [West,East,South,North] ## can set extent to look down on manually
            hours: [integers]
            topography: True|False # set true to use topog for topdown view
            wmap_height: 300m # what height for topdown vertical motion contours?
            ztop: 5000, how high to do transect?
    """
    
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
    
    
    # read one model file at a time
    for umdtime in umdtimes:
        # read cube list
        cubelist = fio.read_model_run(mr, 
                                      hours=[umdtime],
                                      extent=extent,
                                      HSkip=HSkip,
                                      )
        
        wcube, = cubelist.extract(['upward_air_velocity'])
        dtimes = utils.dates_from_iris(wcube)
        
        # read fire
        # TODO: read this out of loop, find time index in loop
        ff, sh, u10, v10 = fio.read_fire(model_run=mr, 
                                         dtimes=dtimes, 
                                         extent=extent,
                                         HSkip=HSkip,
                                         filenames=['firefront','sensible_heat',
                                                    '10m_uwind','10m_vwind'],
                                         )
                
        # extra vert map at ~ 300m altitude
        levh = utils.height_from_iris(wcube)
        levhind = np.sum(levh<wmap_height)
            
        # add fire if available
        ffhr,shhr,u10hr,v10hr = None,None,None,None
        if ff is None and ("1p0" in mr or "3p0" in mr):
            print("INFO: Adding fire at higher res to topdown view")
            ffhr,shhr,u10hr,v10hr=fio.read_fire(
                    model_run=mr[:-4], 
                    dtimes=dtimes, 
                    extent=extent,
                    HSkip=HSkip,
                    filenames=['firefront','sensible_heat',
                        '10m_uwind','10m_vwind'],
                    )
            latshr = ffhr.coord('latitude').points
            lonshr = ffhr.coord('longitude').points

        
        # for each time slice pull out potential temp, winds
        for i,dtime in enumerate(dtimes):
            #utcstamp = dtime.s)trftime("%b %d %H:%M (UTC)")
            ltstamp = (dtime+timedelta(hours=dtoffset)).strftime("%H:%M (LT)")
            
            ## fire and winds for this time step
            ffi,shi,v10i,u10i=None,None,None,None
            if ff is not None:
                ffi = ff[i].data 
                shi = sh[i].data
                u10i = u10[i].data
                v10i = v10[i].data
            else:
                ## if no fire data, still want surface HWinds
                xwind,ywind = cubelist.extract(['x_wind','y_wind'])
                u,v = utils.destagger_wind_cubes([xwind[i,0],ywind[i,0]])
                u10i = u.data
                v10i = v.data

            
            #vertical motion at roughly 300m altitude
            wmap=wcube[i,levhind].data
            
            ### First plot, topography
            fig,ax = topdown_view(extent=extent,
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
            plt.title(mr + " 10m horizontal winds at " + ltstamp)
            
            # add high res info if needed
            if ffhr is not None:
                plotting.map_fire(ffhr[i].data,latshr,lonshr)

            #model_run, plot_name, plot_time, plt, subdir=None,
            fio.save_fig(mr, "wind10m_on_topography", dtime,
                    plt=plt,
                    subdir=subdir,
                    )


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
    dtoffset = utils.local_time_offset_from_lats_lons([extent[2],extent[3]],[extent[0],extent[1]])
    
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
                plt.plot([start[1],end[1]],[start[0],end[0], ], '-k', 
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
                       extent=None,
                       subdir=None,
                       hours=None,
                       ztop=None,
                       start=None,
                       end=None,
                       dx=None,
                       dy=None,
                       SouthNorth=False,
                       HSkip=None,
                       ):
    """
    4 rows 2 columns: 
        first row: top down winds at 10m and 2500m? and transect lines
        2nd, 3rd, 4th rows: transect of heat/winds, transect of vert motion
    ARGUMENTS:
        mr: model run name
        hours: optional run for subset of model hours
        extent: subset extent
        ztop: transect height in metres
        start,end: [lat,lon] of start, end points for transect
            default is middle of plot, left to right 80% extent coverage
        dx,dy: how far to stagger transect lines
            default is no stagger for lons, 20% of extent for lats
    """
    # method takes way too long for compute node if running at full resolution without any subsetting
    if (extent is None) and (HSkip is None) and ("exploratory" not in mr):
        HSkip=2
    # some default ztops
    if ztop is None:
        if 'badja' in mr:
            ztop=10000
        elif 'KI' in mr:
            ztop = 4000
        else:
            ztop = 8000
        
    # theta contours/color levels
    theta_min,theta_max=290,316
    theta_contours = np.arange(theta_min,theta_max),
    theta_levels = np.arange(theta_min,theta_max+1)
    theta_cmap = 'gist_rainbow_r'
    theta_norm = matplotlib.colors.Normalize(vmin=theta_min, vmax=theta_max) 
    # create a scalarmappable from the colormap
    theta_sm = matplotlib.cm.ScalarMappable(cmap=theta_cmap, norm=theta_norm)
    # hwind contours/color levels
    hwind_min,hwind_max = 0,25
    hwind_contours = np.arange(hwind_min,hwind_max,2.5)
    hwind_cmap = "Blues"
    hwind_norm = matplotlib.colors.Normalize(vmin=hwind_min, vmax=hwind_max) 
    # create a scalarmappable from the colormap
    hwind_sm = matplotlib.cm.ScalarMappable(cmap=hwind_cmap, norm=hwind_norm)
    
    # read topog
    cube_topog = fio.read_topog(mr,extent=extent, HSkip=HSkip)
    lats = cube_topog.coord('latitude').points
    lons = cube_topog.coord('longitude').points
    topog = cube_topog.data
    
    # set extent to whole space if extent is not specified
    if extent is None:
        extent = [lons[0],lons[-1],lats[0],lats[-1]]
    
    # Read model run
    umdtimes = fio.hours_available(mr)
    dtoffset = utils.local_time_offset_from_lats_lons(lats,lons)
    
    # hours input can be datetimes or integers
    if hours is not None:
        if not isinstance(hours[0],datetime):
            umdtimes=umdtimes[hours]
        else:
            umdtimes = hours
    if not hasattr(umdtimes,"__iter__"):
        umdtimes = [umdtimes]
    
    ## Default start/end/dx/dy is east-west transects
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

    ## Different defaults if SouthNorth is set to True
    if SouthNorth:
        start=np.mean(lats)-0.4*(lats[-1]-lats[0]), np.mean(lons)
        end=np.mean(lats)+0.4*(lats[-1]-lats[0]), np.mean(lons)
        dx=0.3*(lons[-1]-lons[0])
        dy=0.0
        if subdir is None:
            subdir="SouthNorth"
        else:
            subdir=subdir+"_SouthNorth"


    # 3 transects, shifted by dy and dx # order will be top left to bottom right
    transects = [ [[start[0]+dy,start[1]-dx], [end[0]+dy,end[1]-dx]],
                 [start, end],
                 [[start[0]-dy,start[1]+dx], [end[0]-dy,end[1]+dx]],
                 ]
    
    ## Loop over hours
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
        u,v,w = cubelist.extract(['u','v','upward_air_velocity'])
        zcube, = cubelist.extract(['z_th'])
        dtimes = utils.dates_from_iris(theta)
        
        # read fire front, sens heat, 10m winds
        ff,sh,u10,v10 = fio.read_fire(model_run=mr,
                                      dtimes=dtimes, 
                                      extent=extent,
                                      filenames=['firefront','sensible_heat',
                                                 '10m_uwind','10m_vwind'],
                                      HSkip=HSkip,
                                      )
        ## loop over time steps
        for ti,dtime in enumerate(dtimes):
            LT = dtime+timedelta(hours=dtoffset)
            LTstr = LT.strftime("%H%M (UTC+"+"%.2f)"%dtoffset)
            
            # Get time step data from cubes
            ffi=ff[ti].data
            shi=sh[ti].data
            u10i=u10[ti].data
            v10i=v10[ti].data
            ui=u[ti].data
            vi=v[ti].data
            si=np.hypot(ui,vi) # horizontal wind speed
            wi=w[ti].data
            zi=zcube[ti].data
            Ti=theta[ti].data
            
            ## Set up figure
            fig = plt.figure(figsize=(8,11))
            
            ### FIRST ROW: 10m HWINDS AND VWINDS, AND STREAMS
            ax1 = plt.subplot(4,1,1)
            
            topdown_view(extent,
                         fig=fig,
                         ax=ax1,
                         lats=lats,lons=lons,
                         topog=topog,
                         ff=ffi,
                         sh=shi,
                         u10=u10i,
                         v10=v10i,
                         sh_colorbar=False,
                         )
            
            # topdown view should be equal aspect
            ax1.set_aspect("equal")
            
            # interpolate to this many points along transect (or detect automatically)
            default_interp_points=utils.number_of_interp_points(lats,lons,transects[0][0],transects[0][1])
            npoints=np.min([60, default_interp_points])
            #print("DEBUG: how many interp points?", npoints," default would be ", default_interp_points)
            
            
            ### NEXT 3 ROWS: Transects
            for trani,transect in enumerate(transects):
                # add transect to topdown map
                ax1.plot([transect[0][1],transect[1][1]],[transect[0][0],transect[1][0]],'--k',linewidth=2)
                

                ## LEFT PANEL: show H wind speed and wind streams
                axleft = plt.subplot(4,2,3+trani*2)
                
                plotting.transect_s(si, zi, lats, lons, 
                                    transect[0], 
                                    transect[1],
                                    npoints=npoints,
                                    topog=topog, 
                                    sh=shi,
                                    ztop=ztop,
                                    lines=None, 
                                    contours=hwind_contours,
                                    cmap=hwind_cmap,
                                    colorbar=False,
                                    )
                
                wind_transect_struct = transect_winds(ui, vi, wi, zi, lats, lons, 
                                                      transect,
                                                      ztop=ztop,
                                                      npoints=npoints,
                                                      #topog=topog,
                                                      )
                
                TW = wind_transect_struct['w']
                Xvals = wind_transect_struct['x'][0,:]
                Yvals = wind_transect_struct['y'] # 2d array of altitudes for cross section
                label= wind_transect_struct['xlabel']
                plt.xticks([Xvals[0],Xvals[-1]],
                           [label[0],label[-1]],
                           rotation=10)
                if trani==0:
                    plt.title("Winds (m/s)")
                else:
                    plt.title("")
                axleft.set_ylim(np.nanmin(Yvals),ztop)
                
                ## RIGHT PANEL: T and Vert motion
                axright=plt.subplot(4,2,4+trani*2)
                
                TT,TX,TY = plotting.transect_theta(Ti, zi, lats, lons, 
                                                start=transect[0], 
                                                end=transect[1],
                                                npoints=npoints,
                                                topog=topog, 
                                                sh=shi,
                                                ztop=ztop,
                                                contours=theta_contours,
                                                lines=None, 
                                                levels=theta_levels,
                                                cmap=theta_cmap,
                                                colorbar=False,
                                                )
                
                ## Add vert motion contours
                XRet = utils.transect(wi, lats, lons, 
                                      transect[0], transect[1], 
                                      nx=npoints, 
                                      z=zi)
                
                xdistance=XRet['xdistance'][-1]
                
                label=XRet['xlabel']
                Xvals=XRet['x'][0,:]
                VM_contours = [-5,-4,-3,-2,-1,-.5,.5,1,2,3,4,5]
                VM_colours = ['cyan']*6+['pink']*6
                
                plt.contour(XRet['x'],XRet['y'],XRet['transect'], 
                            VM_contours, # contour lines
                            colors=VM_colours, # contour line colours
                            )
                plt.ylim(np.nanmin(XRet['y']),ztop)
                if trani==0:
                    plt.title("T$_{Potential}$ and Vert motion")
                    plt.xlabel("%.2f (km)"%(xdistance/1000.0),labelpad=-10)
                else:
                    plt.title("")
                
                label= wind_transect_struct['xlabel']
                
                plt.xticks([Xvals[0],Xvals[-1]],
                           [label[0],label[-1]],
                           rotation=10)
                
                axright.set_ylim(np.nanmin(XRet['y']),ztop)
            ## SAVE FIGURE
            #print("DEBUG: LTstr",LTstr)
            # add space in specific area, then add Hwinds colorbar
            cbar_ax1 = fig.add_axes([0.06, 0.74, 0.01, 0.2]) # X Y Width Height
            cbar1 = fig.colorbar(hwind_sm, 
                                 cax=cbar_ax1, 
                                 format=ScalarFormatter(), 
                                 pad=0)
            
            # Add Tpot colorbar
            cbar_ax2 = fig.add_axes([0.92, 0.74, 0.01, 0.2]) #XYWH
            cbar2 = fig.colorbar(theta_sm, cax=cbar_ax2, 
                                 format=ScalarFormatter(),
                                 pad=0,
                                 )
            plt.suptitle(mr + " wind transects " + LTstr,
                         fontsize=22)
            fio.save_fig(mr,"multiple_transects",dtime,
                         subdir=subdir,
                         plt=plt,
                         )
    
def multiple_transects_vertmotion(mr,
                        extent=None,
                        subdir=None,
                        hours=None,
                        ztop=3000,
                        start=None,
                        end=None,
                        dx=None,
                        dy=None,
                        SouthNorth=False,
                        HSkip=None,
                        ):
    """
    4 rows 2 columns: 
        first row: top down winds at 10m and 2500m? and transect lines
        2nd, 3rd, 4th rows: transect of heat/winds, transect of vert motion
    ARGUMENTS:
        mr: model run name
        hours: optional run for subset of model hours
        extent: subset extent
        ztop: transect height in metres
        start,end: [lat,lon] of start, end points for transect
            default is middle of plot, left to right 80% extent coverage
        dx,dy: how far to stagger transect lines
            default is no stagger for lons, 20% of extent for lats
    """
    # method takes way too long for compute node if running at full resolution without any subsetting
    if (extent is None) and (HSkip is None) and ("exploratory" not in mr):
        HSkip=2
        
    # theta contours/color levels
    #theta_min,theta_max=290,316
    #theta_contours = np.arange(theta_min,theta_max),
    theta_levels = np.arange(288,400,2)
    
    w_cmap = plotting._cmaps_['verticalvelocity']
    w_norm = matplotlib.colors.SymLogNorm(0.25,base=2.0, vmin=-16,vmax=16,)
    w_contours = np.union1d(np.union1d(2.0**np.arange(-2,5),-1*(2.0**np.arange(-2,5))),np.array([0]))
    w_ticks = [-16,-4, 0, 4, 16] 
    #w_sm = matplotlib.cm.ScalarMappable(cmap=w_cmap, norm=w_norm)
    
    # read topog
    cube_topog = fio.read_topog(mr,extent=extent, HSkip=HSkip)
    lats = cube_topog.coord('latitude').points
    lons = cube_topog.coord('longitude').points
    topog = cube_topog.data
    
    # set extent to whole space if extent is not specified
    if extent is None:
        extent = [lons[0],lons[-1],lats[0],lats[-1]]
    
    # Read model run
    umdtimes = fio.hours_available(mr)
    dtoffset = utils.local_time_offset_from_lats_lons(lats,lons)
    
    # hours input can be datetimes or integers
    if hours is not None:
        if not isinstance(hours[0],datetime):
            umdtimes=umdtimes[hours]
        else:
            umdtimes = hours
    if not hasattr(umdtimes,"__iter__"):
        umdtimes = [umdtimes]
    
    ## Default start/end/dx/dy is east-west transects
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

    ## Different defaults if SouthNorth is set to True
    if SouthNorth:
        start=np.mean(lats)-0.4*(lats[-1]-lats[0]), np.mean(lons)
        end=np.mean(lats)+0.4*(lats[-1]-lats[0]), np.mean(lons)
        dx=0.3*(lons[-1]-lons[0])
        dy=0.0
        if subdir is None:
            subdir="SouthNorth"
        else:
            subdir=subdir+"_SouthNorth"


    # 3 transects, shifted by dy and dx # order will be top left to bottom right
    transects = [ [[start[0]+dy,start[1]-dx], [end[0]+dy,end[1]-dx]],
                  [start, end],
                  [[start[0]-dy,start[1]+dx], [end[0]-dy,end[1]+dx]],
                  ]
    
    ## Loop over hours
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
        u,v,w = cubelist.extract(['u','v','upward_air_velocity'])
        zcube, = cubelist.extract(['z_th'])
        dtimes = utils.dates_from_iris(theta)
        
        # read fire front, sens heat, 10m winds
        ff,sh,u10,v10 = fio.read_fire(model_run=mr,
                                      dtimes=dtimes, 
                                      extent=extent,
                                      filenames=['firefront','sensible_heat',
                                                  '10m_uwind','10m_vwind'],
                                      HSkip=HSkip,
                                      )
        ## loop over time steps
        for ti, dtime in enumerate(dtimes):
            LT = dtime+timedelta(hours=dtoffset)
            LTstr = LT.strftime("%H%M (UTC+"+"%.2f)"%dtoffset)
            
            # Get time step data from cubes
            ffi=ff[ti].data
            shi=sh[ti].data
            u10i=u10[ti].data
            v10i=v10[ti].data
            ui = u[ti].data
            vi = v[ti].data
            wi= w[ti].data
            zi=zcube[ti].data
            Ti=theta[ti].data
            
            ## Set up figure
            fig = plt.figure(figsize=(8,11))
            
            ### FIRST ROW: 10m HWINDS AND VWINDS, AND STREAMS
            ax1 = plt.subplot(4,1,1)
            
            topdown_view(extent,
                          fig=fig,
                          ax=ax1,
                          lats=lats,lons=lons,
                          topog=topog,
                          ff=ffi,
                          sh=shi,
                          u10=u10i,
                          v10=v10i,
                          sh_colorbar=False,
                          )
            
            # topdown view should be equal aspect
            ax1.set_aspect("equal")
            
            # interpolate to this many points along transect (or detect automatically)
            default_interp_points=utils.number_of_interp_points(lats,lons,transects[0][0],transects[0][1])
            # For visual clarity, more than 60 points is overkill
            npoints=np.min([60, default_interp_points])
            #print("DEBUG: how many interp points?", npoints," default would be ", default_interp_points)
            
            ### NEXT 3 ROWS: Transects
            for trani,[starti, endi] in enumerate(transects):
                # add transect to topdown map
                ax1.plot([starti[1],endi[1]],[starti[0],endi[0]],'--k',linewidth=2)

                ## LEFT PANEL: show H wind speed and wind streams
                row_axis = plt.subplot(4,1, 2+trani) # row 2,3,4
                #print("DEBUG: ")
                #[print(type(thing),np.shape(thing)) for thing in [wi,zi,lats,lons,starti,endi,topog,shi]]
                _,_,_,mappable = plotting.transect_w(wi,zi,lats,lons,
                                    start=starti,
                                    end=endi,
                                    ztop=ztop,
                                    npoints=npoints,
                                    topog=topog,
                                    lines=None,
                                    sh=shi,
                                    colorbar=False,
                                    cmap = w_cmap,
                                    norm = w_norm,
                                    contours = w_contours,
                                    )
                theta_transect_struct = utils.transect(Ti,lats,lons,
                                                      start=starti,
                                                      end=endi,
                                                      nx=npoints,
                                                      z=zi)
                theta_transect = theta_transect_struct['transect']
                
                Xvals = theta_transect_struct['x'] # nrows repeated distance from left axis
                Yvals = theta_transect_struct['y'] # 2d array of altitudes for cross section
                label= theta_transect_struct['xlabel']
                
                # add theta_transect contours
                theta_contours = plt.contour(Xvals,Yvals, theta_transect, 
                            levels=theta_levels,
                            alpha=0.3,
                            )
                plt.clabel(theta_contours, theta_contours.levels[::5])
                
                # # quiverwinds
                # # need transect windspeed and directions
                # w_transect_struct = utils.transect(wi,lats,lons,
                #                                    start=starti,
                #                                    end=endi,
                #                                    nx=npoints,
                #                                    z=zi)
                # transect_w = w_transect_struct['transect']
                # transect_winds_struct = utils.transect_winds(ui,vi,lats,lons,
                #                                              start=starti,
                #                                              end=endi,
                #                                              nx=npoints,
                #                                              z=zi)
                # transect_s = transect_winds_struct['transect_wind']
                # slicex = transect_winds_struct['x']
                # slicez = transect_winds_struct['y']
                
                # plotting.quiverwinds(slicez,slicex,transect_s,transect_w,
                #              n_arrows=25,
                #              add_quiver_key=(trani==0),
                #              alpha=0.35,
                #              )
                
                plt.xticks([Xvals[0,0],Xvals[-3,0]],
                            [label[0],label[-3]],
                            rotation=0)
                plt.title("")
                
                row_axis.set_ylim(np.nanmin(Yvals),ztop)
                
                
            ## SAVE FIGURE
            # add space in specific area, then add vert motion colorbar
            cbar_ax1 = fig.add_axes([0.06, 0.74, 0.01, 0.2]) # X Y Width Height
            cbar1 = fig.colorbar(mappable, 
                                 cax=cbar_ax1, 
                                 ticks=w_ticks, 
                                 pad=0)
            cbar1.set_ticks(w_ticks)
            cbar1.ax.set_yticklabels(w_ticks,rotation=23)

            plt.suptitle(mr + "\n vertical motion " + LTstr,
                          fontsize=20)
            fio.save_fig(mr,"multiple_transects_vertmotion",dtime,
                          subdir=subdir,
                          plt=plt,
                          )    
    
def multiple_transects_SN(*args,**kwargs):
    """
        run multiple transects method with southnorth flag
    """
    kwargs['SouthNorth']=True
    return multiple_transects(*args,**kwargs)

def multiple_transects_vertmotion_SN(*args,**kwargs):
    """
        run multiple transects method with southnorth flag
    """
    kwargs['SouthNorth']=True
    return multiple_transects_vertmotion(*args,**kwargs)

def plot_special_transects(mr,start,end,ztop=5000,
                           hours=None,
                           name=None,
                           n_arrows=20):
    """
        plot transect with full resolution
        loop every 30 minutes
    """
    
    # some defaults
    if name is None:
        name="%.2f,%.2f-%.2f,%.2f"%(start[0],start[1],end[0],end[1])
    West = np.min([start[1],end[1]])-0.01
    East = np.max([start[1],end[1]])+0.01
    South = np.min([start[0],end[0]])-0.01
    North = np.max([start[0],end[0]])+0.01
    extent = [West,East,South,North]
    
    # hwind contours/color levels
    hwind_min,hwind_max = 0,25
    hwind_contours = np.arange(hwind_min,hwind_max,1)
    hwind_cmap = "Blues"
    hwind_norm = matplotlib.colors.Normalize(vmin=hwind_min, vmax=hwind_max) 
    # create a scalarmappable from the colormap
    hwind_sm = matplotlib.cm.ScalarMappable(cmap=hwind_cmap, norm=hwind_norm)
    
    # read topog
    cube_topog = fio.read_topog(mr,extent=extent)
    lats = cube_topog.coord('latitude').points
    lons = cube_topog.coord('longitude').points
    topog = cube_topog.data
    
    # Read model run
    umdtimes = fio.hours_available(mr)
    dtoffset = utils.local_time_offset_from_lats_lons(lats,lons)
    
    # hours input can be datetimes or integers
    if hours is not None:
        if not isinstance(hours[0],datetime):
            umdtimes=umdtimes[hours]
        else:
            umdtimes = hours
    if not hasattr(umdtimes,"__iter__"):
        umdtimes = [umdtimes]

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
        u,v,w = cubelist.extract(['u','v','upward_air_velocity'])
        zcube, = cubelist.extract(['z_th'])
        dtimes = utils.dates_from_iris(theta)
        
        # read fire front, sens heat, 10m winds
        ff,sh = fio.read_fire(model_run=mr,
                              dtimes=dtimes, 
                              extent=extent,
                              filenames=['firefront','sensible_heat',],
                              )
        ## loop over time steps
        for ti,dtime in enumerate(dtimes):
            LT = dtime+timedelta(hours=dtoffset)
            LTstr = LT.strftime("%H%M (UTC+"+"%.2f)"%dtoffset)
            
            ## Get time step data from cubes
            #ffi=ff[ti].data
            shi=sh[ti].data
            #u10i=u10[ti].data
            #v10i=v10[ti].data
            ui=u[ti].data
            vi=v[ti].data
            si=np.hypot(ui,vi) # horizontal wind speed
            wi=w[ti].data
            zi=zcube[ti].data
            Ti=theta[ti].data
            
            npoints=utils.number_of_interp_points(lats,lons,start,end,factor=1.0)
            
            ## show H wind speed and wind quivers
            plotting.transect_s(si, zi, lats, lons, 
                                start, 
                                end,
                                npoints=npoints,
                                topog=topog, 
                                sh=shi,
                                ztop=ztop,
                                lines=None, 
                                contours=hwind_contours,
                                cmap=hwind_cmap,
                                colorbar=True,
                                )
            
            ## Add quiver
            wind_transect_struct = transect_winds(ui, vi, wi, zi, lats, lons, 
                                                  [start,end],
                                                  ztop=ztop,
                                                  npoints=npoints,
                                                  n_arrows=n_arrows,
                                                  )
            plt.title("")    
            #TW = wind_transect_struct['w']
            Xvals = wind_transect_struct['x'][0,:]
            Yvals = wind_transect_struct['y'] # 2d array of altitudes for cross section
            label= wind_transect_struct['xlabel']
            plt.xticks([Xvals[0],Xvals[-1]],
                       [label[0],label[-1]],
                       rotation=10)
            plt.gca().set_ylim(np.min(Yvals),ztop)
                
            ## SAVE FIGURE
            #print("DEBUG: LTstr",LTstr)
            # add space in specific area, then add Hwinds colorbar
            plt.suptitle(mr + "\n" + LTstr,
                         fontsize=15)
            fio.save_fig(mr,"special_transects",dtime,
                         subdir=name,
                         plt=plt,
                         )
    

if __name__ == '__main__':
    latlontimes=firefront_centres["KI_run1"]['latlontimes']
    # keep track of used zooms
    KI_zoom = [136.5,137.5,-36.1,-35.6]
    KI_zoom_name = "zoom1"
    KI_zoom2 = [136.5887,136.9122,-36.047,-35.7371]
    KI_zoom2_name = "zoom2"
    badja_zoom=[149.4,150.0, -36.4, -35.99]
    badja_zoom_name="zoom1"
    belowra_zoom=[149.61, 149.8092, -36.2535, -36.0658]
    belowra_zoom_name="Belowra"
    
    KI_jetrun_zoom=[136.6,136.9,-36.08,-35.79]
    KI_jetrun_name="KI_earlyjet"
    # settings for plots
    mr='badja_run2_exploratory'
    zoom=badja_zoom #belowra_zoom
    subdir=badja_zoom_name #belowra_zoom_name
    
    if False: # look at green valley for dragana
        mr="green_valley_run2_fix"
        start,end=[[-35.9,147.5],[-36.34,148.1]]
        extent=[147.35,148.25,-36.5,-35.7]
        hours=[datetime(2019,12,30,8)]
        for method in [multiple_transects_vertmotion,multiple_transects,]:
            method(mr, extent=extent, ztop=9000,
                    start=start,end=end,
                    hours=hours,
                    dx=.01,
                    dy=.01,
                    )

    # special transects
    if True:
        ztop=5000
        special_transects=[
                [[-36.3262,149.6610],[-36.3030,149.9515]],# yowrie - wandella
                [[-36.3071,149.6708],[-36.4152,149.9842]],# yowrie - cobargo
                [[-36.25,149.63],[-36.35,149.78]], # shear
                [[-36.33+.08,149.85-.1],[-36.33-.08,149.85+.1]], # wandella valley
                ]
        st_names = ['yowrie_wandella',
                'yowrie_coburg',
                'shear',
                'wandella_valley',]

        for mr in ['badja_run3','badja_UC1']:
            for i in [0]:
                start,end=special_transects[i]
                name=st_names[i]
                plot_special_transects(mr,start,end,ztop=ztop,name=name)
    
    if False:
        mr='KI_run2'
        # best return shear example at 0020LT
        LLjet_example = [[-35.725,136.7],[-36.08,136.7]]
        LLjet_example_ztop = 3000
        
        start,end=LLjet_example
        ztop=LLjet_example_ztop
        plot_special_transects(mr,start,end,ztop=ztop,
                name="lowlevel_jet",
                n_arrows=10,)
    
    if False:
        multiple_transects_vertmotion("KI_run2_exploratory",ztop=5000,)
    
    ## Special wandella transects
    if False:
        wandella_zoom=[149.5843,  149.88, -36.376, -36.223]
        wandella_zoom_name="Wandella"
        wandella_transect = [[-36.250,149.67],[-36.35,149.82]]
        dx=0.04
        dy=0.0
        if False:
            multiple_transects("badja_run3",
                           extent=wandella_zoom,
                           subdir="Wandella_diagonal", 
                           ztop=8000, 
                           start=wandella_transect[0],
                           end=wandella_transect[1], 
                           dx=dx, 
                           dy=dy,
                           )
        multiple_transects_vertmotion("badja_run3",
                           extent=wandella_zoom,
                           subdir="Wandella_diagonal", 
                           ztop=8000, 
                           start=wandella_transect[0],
                           end=wandella_transect[1], 
                           dx=dx, 
                           dy=dy,
                           )
    
    ## Multiple transects 
    if False:
        ## north_south transects
        #multiple_transects_SN(mr,
        #        extent=zoom, 
        #        subdir=subdir,
        #        )
        ## 
        multiple_transects(mr,
                extent=zoom,
                subdir=subdir,
                )
        
    
    ## TOPDOWN 10m WINDS ONLY
    if False:
        topdown_view_only(mr,extent=zoom,subdir=subdir)
    
    ## MAP WITH DEFINED TRANSECT
    if False:
        map_and_transects('KI_run1_exploratory', 
            latlontimes=latlontimes,
            hours=np.arange(6,14),
            )
