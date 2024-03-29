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
from matplotlib.ticker import FormatStrFormatter,LinearLocator, ScalarFormatter#, LogFormatter
#import matplotlib.patheffects as PathEffects
import numpy as np
import warnings
from datetime import datetime,timedelta
#from scipy import interpolate
#import cartopy.crs as ccrs

# local modules
from utilities import plotting, utils #, constants
from utilities import fio_iris, fio
from utilities import transect_utils
###
## GLOBALS
###
_sn_ = 'transects'

###
## METHODS
###


def basic_transects(mr,start,end,ztop=5000,
                    hours=None,
                    name=None,
                    n_arrows=20,
                    T_lines=np.arange(280,350,2),):
    """
        plot transect with full resolution
            horizontal wind speeds + pot. temp + wind quivers
            vertical motion + pot. temp + wind quivers
            todo: rotation + pot. temp + wind quivers
            todo: vorticity + vert motion
            maybe: pot temp + vert motion
        ARGS:
            name: transect name needed as first folder in subdir
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
    cube_topog = fio_iris.read_topog(mr,extent=extent)
    lats = cube_topog.coord('latitude').points
    lons = cube_topog.coord('longitude').points
    topog = cube_topog.data
    
    # Make a little plot showing where transect is located
    wider_extent = [West-.25,East+.25,South-.2,North+.2]
    wider_topog = fio_iris.read_topog(mr,extent=wider_extent)
    wider_lats = wider_topog.coord('latitude').points
    wider_lons = wider_topog.coord('longitude').points
    plotting.map_contourf(wider_topog.data,wider_lats,wider_lons,title=name)
    plotting.map_add_locations_extent(wider_extent)
    # add transect [x0,x1],[y0,y1]
    plt.plot([start[1],end[1]], [start[0],end[0]],'r--',linewidth=3)
    fio.save_fig(mr, _sn_, name, plt, subdir=name)

    # Read model run
    um_times = fio.hours_available(mr)
    dtoffset = utils.local_time_offset_from_lats_lons(lats,lons)
    
    # hours input can be datetimes or integers
    if hours is not None:
        if not isinstance(hours[0],datetime):
            um_times=um_times[hours]
        else:
            um_times = hours
    if not hasattr(um_times,"__iter__"):
        um_times = [um_times]

    ## Loop over hours
    for um_time in um_times:
        # read cube list
        cubelist = fio_iris.read_model_run(mr, 
                                      hours=[um_time],
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
        #print("DEBUG:", cubelist)
        # Calculate rotation
        rotation = utils.rotation(
            u.data,
            v.data,
            w.data,
            zcube.data, # altitude in m
            lats,lons,
            )
        #print("DEBUG: rotation shape,nanmin,nanmean,nanmax",rotation.shape,np.nanmin(rotation),np.nanmean(rotation),np.nanmax(rotation),)

        # read fire front, sens heat, 10m winds
        ff,sh = fio_iris.read_fire(model_run=mr,
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
            
            # rotation
            roti=rotation[ti]

            # Vorticity in 3d?
            
            npoints=transect_utils.number_of_interp_points(lats,lons,start,end,factor=1.0)
            
            ### PLOT 1: Horizontal wind speed + pot.temp + quiver
            ### PLOT 2: Vertical winds + pot-temp + quiver
            ### PLOT 3: Rotation + pot-temp + quiver
            for ploti, subdir in enumerate(["HWind","VWind", "Rotation"]):
                if ploti == 0:
                    ## show H wind speed
                    transect_utils.plot_transect_s(
                        si, zi, lats, lons, 
                        start, end,
                        npoints=npoints,
                        topog=topog, 
                        sh=shi,
                        ztop=ztop,
                        lines=None, 
                        levels=hwind_contours,
                        cmap=hwind_cmap,
                        colorbar=True,
                        )
                elif ploti == 1:
                    ## show V wind speed
                    transect_utils.plot_transect_w(
                        wi, zi, lats, lons, 
                        start, end,
                        npoints=npoints,
                        topog=topog, 
                        sh=shi,
                        ztop=ztop,
                        lines=None,
                        colorbar=True,
                        )
                elif ploti == 2:
                    ## show rotation
                    _,_,_,CS = transect_utils.plot_transect(
                        roti, zi, lats, lons,
                        start, end,
                        npoints=npoints,
                        topog=topog,
                        sh=shi,
                        ztop=ztop,
                        lines=None,
                        colorbar=False, # will add colorbar after, with fixed limits
                        cmap="seismic",
                        levels=np.arange(-60,60.1,3) * 1e-6,
                        extend="both",
                    )
                    # reset over and under colours
                    cmap = CS.get_cmap().copy()
                    cmap.set_under('cyan')
                    cmap.set_over('fuchsia')
                    CS.set_cmap(cmap)
                    plt.colorbar(pad=.01,aspect=30,shrink=0.85,fraction=.075)

                ## Add pot temp contour
                transect_utils.add_contours(
                    Ti, zi, lats, lons, start, end, 
                    npoints=npoints, 
                    ztop=ztop, 
                    lines=T_lines,
                    alpha=0.5,
                    )
                
                ## Add quiver
                wind_transect_struct = transect_utils.plot_transect_winds(
                    ui, vi, wi, zi, lats, lons, 
                    [start,end],
                    ztop=ztop,
                    npoints=npoints,
                    n_arrows=n_arrows,
                    )
                
                plt.title("")
                
                Xvals = wind_transect_struct['x'][0,:]
                Yvals = wind_transect_struct['y'] # 2d array of altitudes for cross section
                label = wind_transect_struct['xlabel']
                plt.xticks([Xvals[0],Xvals[-1]],
                           [label[0],label[-1]],
                           rotation=10)
                plt.gca().set_ylim(np.min(Yvals),ztop)
                
                ## SAVE FIGURE
                # add space in specific area, then add Hwinds colorbar
                plt.suptitle(mr + "\n" + LTstr,
                             fontsize=15)
                fio.save_fig(mr,"transects",dtime,
                             subdir=name+'/'+subdir,
                             plt=plt,
                             )
            
            ### PLOT 3: Vorticity + vmotion contours
            # transect_utils.plot_transect_w(
            #         wi, zi, lats, lons, 
            #         start, end,
            #         npoints=npoints,
            #         topog=topog, 
            #         sh=shi,
            #         ztop=ztop,
            #         lines=None,
            #         colorbar=True,
            #         )
                
            # ## Add vmotion contour
            # transect_utils.add_contours(
            #     wi, zi, lats, lons, start, end, 
            #     npoints=npoints, 
            #     ztop=ztop, 
            #     lines=np.arange(-25,25.1),
            #     cmap='seismic',)
            
            # ## Add quiver
            # wind_transect_struct = transect_utils.plot_transect_winds(
            #     ui, vi, wi, zi, lats, lons, 
            #     [start,end],
            #     ztop=ztop,
            #     npoints=npoints,
            #     n_arrows=n_arrows,
            #     )
            
            # plt.title("")
            
            # Xvals = wind_transect_struct['x'][0,:]
            # Yvals = wind_transect_struct['y'] # 2d array of altitudes for cross section
            # label = wind_transect_struct['xlabel']
            # plt.xticks([Xvals[0],Xvals[-1]],
            #            [label[0],label[-1]],
            #            rotation=10)
            # plt.gca().set_ylim(np.min(Yvals),ztop)
            
            # ## SAVE FIGURE
            # # add space in specific area, then add Hwinds colorbar
            # plt.suptitle(mr + "\n" + LTstr,
            #              fontsize=15)
            # fio.save_fig(mr,"transects",dtime,
            #              subdir=subdir,
            #              plt=plt,
            #              )
            

def KI_transect():
    mr = "KI_run2"
    name = "LLjet_example"
    hours = np.arange(2,18)
    transect = transect_utils.mr_transects[mr][name] #[-35.725,136.7, -36.08,136.7],    
    
    start,end = [transect[0],transect[1]], [transect[2],transect[3]]
    ztop = 2500
    n_arrows=5

    hwind_levels=np.arange(5,25.1,1)

    West = np.min([start[1],end[1]])-0.01
    East = np.max([start[1],end[1]])+0.01
    South = np.min([start[0],end[0]])-0.01
    North = np.max([start[0],end[0]])+0.01
    extent = [West,East,South,North]
    
    # read topog
    cube_topog = fio_iris.read_topog(mr,extent=extent)
    lats = cube_topog.coord('latitude').points
    lons = cube_topog.coord('longitude').points
    topog = cube_topog.data
    
    
    # Read model run subset of hours
    um_times = fio.hours_available(mr)[hours]
    # local time offset
    dtoffset = utils.local_time_offset_from_lats_lons(lats,lons)
    
    ## Loop over hours
    for um_time in um_times:
        # read cube list
        cubelist = fio_iris.read_model_run(mr, 
                                      hours=[um_time],
                                      extent=extent,
                                      )
                                      
        # add temperature, height, destaggered wind cubes
        utils.extra_cubes(cubelist,
                          #add_theta=True,
                          add_z=True,
                          add_winds=True,)
        #theta, = cubelist.extract('potential_temperature')
        u,v,w = cubelist.extract(['u','v','upward_air_velocity'])
        zcube, = cubelist.extract(['z_th'])
        dtimes = utils.dates_from_iris(w)
        
        # read fire front, sens heat
        ff,sh = fio_iris.read_fire(model_run=mr,
                              dtimes=dtimes, 
                              extent=extent,
                              filenames=['firefront','sensible_heat',],
                              )
        
        ## loop over time steps
        for ti,dtime in enumerate(dtimes):
            LT = dtime+timedelta(hours=dtoffset)
            LTstr = LT.strftime("%H%M (UTC+"+"%.2f)"%dtoffset)
            
            ## Get time step data from cubes
            shi=sh[ti].data
            ui=u[ti].data
            vi=v[ti].data
            si=np.hypot(ui,vi) # horizontal wind speed
            wi=w[ti].data
            zi=zcube[ti].data
            
            npoints=transect_utils.number_of_interp_points(lats,lons,start,end,factor=1.0)
            
            ### PLOT 1: Horizontal wind speed + pot.temp + quiver
            ## show H wind speed
            transect_utils.plot_transect_s(
                si, zi, lats, lons, 
                start, end,
                npoints=npoints,
                topog=topog, 
                sh=shi,
                ztop=ztop,
                lines=None, 
                levels=hwind_levels,
                #cmap=hwind_cmap,
                colorbar=True,
                )
                
            ## Add quiver
            wind_transect_struct = transect_utils.plot_transect_winds(
                ui, vi, wi, zi, lats, lons, 
                [start,end],
                ztop=ztop,
                npoints=npoints,
                n_arrows=n_arrows,
                )
                
            plt.title("")
                
            Xvals = wind_transect_struct['x'][0,:]
            Yvals = wind_transect_struct['y'] # 2d array of altitudes for cross section
            label = wind_transect_struct['xlabel']
            plt.xticks(
                [Xvals[0],Xvals[-1]],
                [label[0],label[-1]],
                rotation=10,
                )
            plt.gca().set_ylim(np.min(Yvals),ztop)
            plt.title(mr + "\n" + LTstr, fontsize=15)
            fio.save_fig(mr,"transects",dtime,
                             subdir=name+'/special',
                             plt=plt,
                             )


if __name__ == '__main__':
    
    # test method
    #def basic_transects(mr,start,end,ztop=5000,
    #                hours=None,
    #                name=None,
    #                n_arrows=20,
    #                T_lines=np.arange(280,350,2),):
    if False:
        KI_transect()
    
    if False:
        # all badja_am1 transects
        mr="badja_am1"
        transects=transect_utils.mr_transects[mr]
        for name,transect in transects.items():
            lat0,lon0,lat1,lon1 = transect
            start,end = [[lat0,lon0],[lat1,lon1]]
            basic_transects(mr,start,end,
                    ztop=3000,
                    hours=np.arange(5,12),
                    name=name,
                    n_arrows=17,
                    )
    if True:
        # special badja transects for FGV stuff
        mr="badja_am1"
        transects = {
            "along_front_2.1":[-36.5,149.80, -36.275,149.82],
            "along_front_2.2":[-36.5,149.82, -36.275,149.84],
            "along_front_2.3":[-36.5,149.84, -36.275,149.86],
            # "FGV_path_1":[-36.28,149.7,-36.355,149.95],
            # "FGV_path_2":[-36.30,149.7,-36.375,149.95],
            # "FGV_path_3":[-36.32,149.7,-36.395,149.95],
        }
        for name,transect in transects.items():
            lat0,lon0,lat1,lon1 = transect
            start,end = [[lat0,lon0],[lat1,lon1]]
            basic_transects(mr,start,end,
                    ztop=4000,
                    hours=np.arange(6,10),
                    name=name,
                    n_arrows=17,
                    )
