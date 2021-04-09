#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 16:28:18 2019
    Show fire spread and intensity over time
@author: jesse
"""


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patheffects, colors, image, ticker
import numpy as np
# for legend creation:
from matplotlib.lines import Line2D
from datetime import datetime,timedelta

# continuous colourmap
import cmocean

import iris

from cartopy import crs as ccrs

from utilities import plotting, utils, constants
from utilities import fio_iris as fio

###
## GLOBALS
###
_sn_ = 'winds'



def wind_and_heat_flux_looped(mr,
        extent=None,
        tiffname=None,
        HSkip=None,
        subdir=None,
        ):
    """
    Figure with 2 panels: 10m wind speed, and wind direction overlaid on heat flux
    wind direction can be put onto a tiff if tiffname is set
    """
    #dtime = datetime(2017,2,12,7,30) # 1830 LT
    simname=mr.split('_')[0]
    LToffset= timedelta(hours=fio.sim_info[simname]['UTC_offset'])

    # Read 10m winds and firefront:
    ff,fflux,u10,v10=fio.read_fire(mr,
            extent=extent,
            HSkip=HSkip,
            filenames=['firefront','sensible_heat','10m_uwind','10m_vwind',],
            )
    # just look at every 10 mins:
    ff=ff[::10]
    fflux=fflux[::10]
    u10=u10[::10]
    v10=v10[::10]

    # topography
    topog=fio.read_topog(mr,
            extent=extent,
            HSkip=HSkip,
            )


    lat,lon = u10.coord('latitude').points,u10.coord('longitude').points
    dtimes = utils.dates_from_iris(u10)
    #print(dtimes)
    # set up plotting
    # windspeed colormap scaling
    vmin,vmax=0,20 # m/s
    hcontours=np.linspace(vmin,vmax,11)
    hnorm=colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    # heatflux colorbar scaling (norm and levels)
    index_min,index_max=2,5
    fflux_min,fflux_max=10**index_min,10**index_max
    fflux_norm = colors.LogNorm(vmin=fflux_min,vmax=fflux_max)
    fflux_levels = np.logspace(index_min,index_max,20)

    # loop over time:
    for ii,utc in enumerate(dtimes):
        LT = utc + LToffset
        ffi=ff[ii]
        u10i=u10[ii]
        v10i=v10[ii]
        ffluxi=fflux[ii]
        
        fig=plt.figure(figsize=(14,16))
        ax1=fig.add_subplot(2,1,1)

        ######### FIRST FIGURE: windspeed ############
        ##############################################

        s10 = utils.wind_speed_from_uv_cubes(u10i,v10i)
        # wind speed contourf
        csh = plt.contourf(lon, lat, s10.data, hcontours, 
                cmap="Blues",
                extend='max',
                )
        plt.colorbar(ticklocation=ticker.MaxNLocator(5),pad=0)
    
        #plotting.map_add_locations_extent(extentname, hide_text=False)
        plt.title("10m Wind Speed at %s"%LT.strftime("%H%M (LT)"))
    
        ######## SECOND FIGURE: heat and wdir ########
        ##############################################
    
        ## This figure will be on a sirivan tiff
        # helper function to set up tiff image backdrop
        if tiffname is not None:
            _,ax2 = plotting.map_tiff_qgis(
                    fname=tiffname,
                    extent=extent,
                    fig=fig,
                    subplot_row_col_n=[2,1,2],
                    )
        else:
            ax2=plt.subplot(2,1,2)
            print("DEBUG: topog, lat, lon",topog.shape, np.shape(lat), np.shape(lon))
            plotting.map_topography(topog.data,lat,lon)


        # add contourfs showing heat flux
        cs=plt.contourf(lon,lat,ffluxi.data+1, fflux_levels,
            cmap='hot',norm=fflux_norm, 
            vmin=fflux_min, vmax=fflux_max,
            alpha=0.8,
            extend='max',
            )
        # NOTES: extend is required in this case or else values under vmin hide many of the contourfs
        #      : I added 1 to the ffluxti so that we don't need to worry about the 0 value in log scaling
        plt.colorbar(
                ticklocation=ticker.MaxNLocator(5),
                pad=0,
                ticks=[1e2,1e3,1e4,1e5],
                )
            
        ##Streamplot the horizontal winds
        ## This shifts the subplot grid axes slightly!! 
        # this is seemingly to show the arrows in full where they extend outside the boundary
        #streamLW=utils.wind_speed_to_linewidth(s10.data,speedmax=vmax)
        ax2.streamplot(lon,lat,u10i.data,v10i.data, 
                color='k',
                #linewidth=streamLW,
                minlength=0.5,
                density=(0.7, 0.6),
                )
        # set limits back to latlon limits
        ax2.set_ylim(lat[0],lat[-1])  
        ax2.set_xlim(lon[0],lon[-1])
    
        # remove x and y ticks
        #plt.xticks([],[])
        #plt.yticks([],[])
        ax2.set_title('Heat flux (Wm$^{-2}$)')
    
        #plotting.map_add_locations_extent(extentname, hide_text=True)
        fio.save_fig(mr, "wind10_and_heat", utc,
                plt=plt, 
                subdir=subdir,
                )

def rotation_looped(mr, extent=None, dtimes=None, HSkip=None, subdir=None,):
    """
    """
    simname=mr.split('_')[0]
    H0 = fio.sim_info[simname]['filedates'][0]
    if dtimes is None:
        dtimes=[H0 + timedelta(minutes=x*30) for x in range(24*2)]
    LToffset= timedelta(hours=fio.sim_info[simname]['UTC_offset'])

    # Read 10m winds and firefront:
    ff,u10,v10 = fio.read_fire(mr,
        dtimes=dtimes,
        extent=extent,
        HSkip=HSkip,
        filenames=['firefront','10m_uwind','10m_vwind'],
        )
    #print(u10)
    lat,lon = u10.coord('latitude').points,u10.coord('longitude').points
    
    # set up plotting
    cmap='gist_rainbow' # wind dir cmap
    #cmap = cmocean.cm.phase # not quite dense enough

    # most wind coming from southwest: southwest for  map
    dirmin,dirmax=0,360 # most wind coming from south west,
    norm=colors.Normalize(vmin=dirmin,vmax=dirmax)
    tickform=ticker.ScalarFormatter()
    levs = np.arange(dirmin,dirmax+1)
    
    wd = utils.wind_dir_from_uv(u10.data,v10.data)
    proj=ccrs.PlateCarree()
    #print(u10.coord('latitude').coord_system)
    #print(u10.coord('latitude').coord_system())
    #transform=u10.coord('latitude').coord_system.as_cartopy_projection()

    for ii,dtime in enumerate(dtimes):
        dtimeLT = dtime+LToffset
        # remove time dim:
        ffi = ff[ii]
        u10i = u10[ii]
        v10i = v10[ii]
        wdi = wd[ii]

        ax = plt.axes(projection=proj)
        #cs =  plt.contourf(lon,lat,wdi,levs,
        #        cmap=cmap,
        #        norm=norm,
        #        #extend='both',
        #        )
        cs = plt.pcolormesh(lon,lat,wdi,
                cmap=cmap,
                norm=norm,
                shading='auto', # pcolormesh takes corners of grid squares
                #transform=transform,
                )
        plt.colorbar(ticklocation=tickform,
                pad=0,
                ticks=np.arange(0,361,30),
                )

        # add firefront
        plotting.map_fire(ffi.data, lat, lon, 
                alpha=.8, 
                linestyles='--',
                linewidths=1,
                )

        ax=plt.gca()
        ax.set_title('10m wind direction at %s (LT)'%dtimeLT.strftime("%H:%M"))
        #xticks=[149.85,149.9,149.95,150.0]
        #ax.set_xticks(xticks)
        #ax.set_xticklabels([str(x) for x in xticks],rotation=30)
        #plotting.map_add_locations_extent('sirivan_pcb',nice=True)
        #plt.plot(casLL[1],casLL[0],color='grey',linewidth=0,
        #        marker='o',)

        fio.save_fig(mr, "wind_dir_10m", dtime, subdir=subdir, plt=plt)

if __name__=='__main__':
    
    # keep track of used zooms
    badja_zoom=[149.4,150.0, -36.4, -35.99]
    KI_zoom= [136.52,137.24, -36.07,-35.6] 
    KI_zoom_name='zoom1'
    KI_tiffname='KI.tiff' # 'badja.tiff'
    badja_zoom_name="zoom1"
    badja_tiffname=None

    # settings for plots
    mr='KI_run3'
    extent=None# badja_zoom
    subdir=None# badja_zoom_name
    tiffname=None# badja_tiffname

    ### Run the stuff
    
    # CHECK LOWER LEVEL ROTATION
    if True:
        rotation_looped(mr,
                extent=extent,
                subdir=subdir,
                )

    # Look at winds and heat flux
    if True:
        wind_and_heat_flux_looped(mr, 
                extent=extent,
                HSkip=2,
                tiffname=tiffname,
                subdir=subdir,
                )
    
