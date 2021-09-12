# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 13:03:46 2021
    Simplish transects using xarray data and no streams
@author: jgreensl
"""

import matplotlib
matplotlib.use('Agg')

# plotting stuff
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, LinearLocator#, LogFormatter
#import matplotlib.patheffects as PathEffects
import numpy as np
import warnings

#from datetime import datetime,timedelta
from pandas import Timedelta
#from scipy import interpolate
#import cartopy.crs as ccrs

# local modules
from utilities import plotting, utils, constants, fio

###
## GLOBALS
###
# [lat0,lon0],[lat1,lon1] transects of note
KI_transects = {
        "southerly":[[0,0],[0,0]],
        "middle":[[0,0],[0,0]],
        }

###
## METHODS
###


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

def plot_wind_transect(
        DS,
        ztop=5000,
        start=None,
        end=None,
        ):
    """
    """
    print("DEBUG: plot_wind_transect is TODO")

def model_run_transect_winds(
        mr,
        hours=None,
        extent=None,
        ztop=5000,
        start=None,
        end=None,
        subdir=None,
        ):
    """
    2 rows 2 columns: 
        first row: top down winds at 10m and 2500m? and transect lines
        plot left row2:
            transect of horizontal wind(contourf) and vert motion(contour), 
        plot right row2:
            direction (colormesh) and T (contour)
    ARGUMENTS:
        mr: model run name
        hours: optional run for subset of model hours
        extent: subset extent
        ztop: transect height in metres
        start,end: [lat,lon] of start, end points for transect
            default is middle of plot, left to right 80% extent coverage
    """
    # Defaults
    if hours is None:
        hours=range(24)
    
    # read topog
    topog=fio.model_run_topography(mr)
    if extent is not None:
        topog = fio.extract_extent(topog,extent)
    lats=topog.latitude.data
    lons=topog.longitude.data
    houroffset=utils.local_time_offset_from_lats_lons(lats,lons)
    # transect will be middle left to right
    if start is None or end is None:
        start = np.mean(lats), np.mean(lons)-0.35*(lons[-1]-lons[0])
        end = np.mean(lats), np.mean(lons)+0.35*(lons[-1]-lons[0])
    
    # read fire model output
    DS_fire=fio.read_model_run_fire(mr)
    
    for hour in hours:
        # read hour of model data
        DS=fio.read_model_run_hour(mr,hour=hour)
        if extent is not None:
            DS = fio.extract_extent(DS,extent)
        
        times=DS.time.data # np.datetime64 array
        #print("DEBUG: transects: ",DS)
        
        # loop over timesteps
        for ti,time_utc in enumerate(times):
            # create square figure
            plt.figure(figsize=[13,13])
            plt.subplot(2,1,1)
            plt.contourf(lons,lats,topog,cmap="terrain")
            
            # add fire
            DS_fire_timeslice = DS_fire.loc[dict(time=time_utc)]
            #print("DEBUG: transect: fire timeslice:", DS_fire_timeslice)
            FF = DS_fire_timeslice['firefront'].data
            plotting.map_fire(FF.T,lats,lons)
            U10 = DS_fire_timeslice['UWIND_2'].data
            V10 = DS_fire_timeslice['VWIND_2'].data
            #print("DEBUG: U10:",type(U10),np.shape(U10))
            plt.streamplot(lons,lats,
                           U10.T, V10.T,
                           density=[.7,.7],
                           color='k',
                           arrowsize=1.5,
                           )
            # get local time
            time_lt = utils.local_time_from_time_lats_lons(time_utc,lats,lons)
            
            time_str=time_lt.strftime("%Y%m%d %H%M")+"(UTC+%.2f)"%houroffset
            
            plt.subplot(2,1,2)
            DS_timeslice=DS.loc[dict(time=time_utc)]
            #print("DEBUG: transect: timeslice DS:", DS_timeslice)
            
            
            
            plot_wind_transect(
                    DS_timeslice,
                    ztop=ztop, 
                    start=start,
                    end=end,
                    )
            plt.suptitle(time_str+ "wind transect")
            fio.save_fig(mr,"new_wind_transect",time_utc,plt,)
    
#        # read fire front, sens heat, 10m winds
#        ff,sh,u10,v10 = fio.read_fire(model_run=mr,
#                                      dtimes=dtimes, 
#                                      extent=extent,
#                                      filenames=['firefront','sensible_heat',
#                                                 '10m_uwind','10m_vwind'],
#                                      )
        

if __name__ == '__main__':
    mr="KI_run1_exploratory"
    model_run_transect_winds(mr)
