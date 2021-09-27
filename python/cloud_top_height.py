# -*- coding: utf-8 -*-
"""
Created on Thu May  6 14:47:04 2021
    calculate and display cloud top heights
    based on water and ice mixing ratios > .01 g/kg
    
    badga_run3 at ~50% extent reduction
        CPU Time Used: 00:10:53
        Memory Used: 14.57GB
        Walltime Used: 00:12:51

@author: jgreensl
"""

# plotting stuff
import matplotlib.pyplot as plt
from matplotlib import ticker
#from matplotlib import colors
import matplotlib
#import matplotlib.patheffects as PathEffects
import numpy as np

# local modules
from utilities import plotting, utils, fio, constants

##
## METHODS
###


def plot_cloud_top_height(DA_cloud,zth, DA_ff=None,DA_sh=None,DA_topog=None,thresh=constants.cloud_threshold):
    """
    thresh is 0.01 g/kg_air of water/ice content
    """
    # first find latitude and longitude
    lats=DA_cloud.latitude.values
    lons=DA_cloud.longitude.values
    cmap="gist_ncar"
    cbar_levels=np.arange(0,18000,50)


    # cloud top height calculation
    zth_masked = np.copy(zth)
    # set height without clouds to zero
    zth_masked[DA_cloud.values<thresh] = 0.0
    # max height in each column should be cloud top
    cth = np.max(zth_masked,axis=0)
    # Create figure and subplots

    rv = plt.contourf(lons,lats,cth,cbar_levels,
            extend="max",
            cmap=cmap,
            )
    
    if DA_ff is not None:
        plotting.map_fire(DA_ff.values,lats,lons, 
                          colors=['grey'], 
                          linestyles=['--'])
    
    if DA_sh is not None:
        sh=DA_sh.values.T
        plotting.map_sensibleheat(sh,
                                  DA_sh.lat.values,
                                  DA_sh.lon.values,
                                  colorbar=False,
                                  #zorder=3,
                                  )
    
    # topography
    if DA_topog is not None:
        topog_con = (2, 100, 200, 300, 400, 800)
        if len(lats) == DA_topog.shape[1]:
            topog=DA_topog.values.T
        else:
            topog = DA_topog.values
        plt.contour(lons,lats,topog,np.array(topog_con),colors='k')
    
    return rv


def cloud_top_height(mr, extent=None, subdir=None):
    """
    ARGS:
        mr: model run name
        extent: [W,E,S,N] in degrees
        subdir: name to save zoomed into
        coastline: set to positive number to add coastline contour in metres
    """
    # Read firefront, heatflux (W/m2), U and V winds    
    DS_fire = fio.read_model_run_fire(mr)
    DA_topog = fio.model_run_topography(mr)
    
    if extent is not None:
        DS_fire = fio.extract_extent(DS_fire,extent)
        DA_topog = fio.extract_extent(DA_topog,extent)
        if subdir is None:
            subdir=str(extent)
    
    lats=DS_fire.lat.data
    lons=DS_fire.lon.data
    if extent is None:
        extent=[lons[0],lons[-1],lats[0],lats[-1]]
        
    # loop over timesteps
    hours=fio.hours_available(mr)
    for hi,hour_utc in enumerate(hours):
        
        ## slice time
        DS_atmos = fio.read_model_run_hour(mr,hour=hi)
        times_utc = DS_atmos.time.values
        
        # read x and y winds
        DA_water = DS_atmos['cld_water'] # kg/kg air
        DA_ice = DS_atmos['cld_ice'] # kg/kg air
        #print("DEBUG: water, ice",DA_water,DA_ice)
        DA_cloud = (DA_water+DA_ice)*1000.0
        DA_cloud['units'] = "g kg-1" # kg kg-1 to g/kg
        #print("DEBUG: cloud1",DA_cloud.shape, np.max(DA_cloud.values), np.mean(DA_cloud.values))
        DA_p = DS_atmos['pressure']
        DA_mslp = DS_atmos['mslp']#'air_pressure_at_sea_level']
        # subset to extent
        if extent is not None:
            DA_cloud = fio.extract_extent(DA_cloud, extent)
            DA_p = fio.extract_extent(DA_p, extent)
            DA_mslp = fio.extract_extent(DA_mslp, extent)
        zth = utils.zth_calc(DA_mslp.values, DA_p.values)
        #print("DEBUG: zth" ,zth.shape, type(zth), np.max(zth), np.mean(zth))
        
        for ti, time_utc in enumerate(times_utc):
            DS_fire_slice = DS_fire.sel(time=time_utc)
            DA_ff  = DS_fire_slice['firefront']
            DA_sh  = DS_fire_slice['SHEAT_2']
            
            ## get local time
            time_lt = utils.local_time_from_time_lats_lons(time_utc,lats,lons)
            houroffset = utils.local_time_offset_from_lats_lons(lats,lons)
            time_str=time_lt.strftime("%dT%H:%M")+"(UTC+%.2f)"%houroffset
                            
            cs_rv = plot_cloud_top_height(DA_cloud[ti], zth[ti], 
                    DA_topog=DA_topog,
                    )

            plotting.map_add_locations_extent(extent,hide_text=True)
            #if coastline>0 and np.min(DA_topog.values)<coastline:
            #    plt.contour(lons,lats,DA_topog.values, np.array([coastline]),colors='k')
            #    plt.gca().set_aspect("equal")
            #    #plt.gca().set(xticks=[],yticks=[],aspect="equal")
            plt.gca().set_aspect("equal")
            plt.title(mr+" "+time_str)

            cbar = plt.colorbar(cs_rv, 
            #                    cax=cbar_ax, 
            #                    ticks=vort_ticks, 
                                pad=0,
            #                    orientation='horizontal',
                                )
            #cbar.ax.set_xticklabels(vort_ticks_str,rotation=20)
            # save figure
            fio.save_fig(mr,"cloud_top_height", time_utc, plt, subdir=subdir)

if __name__ == '__main__':

    # keep track of used zooms
    KI_zoom = [136.5,137.5,-36.1,-35.6]
    KI_zoom_name = "zoom1"
    KI_zoom2 = [136.5887,136.9122,-36.047,-35.7371]
    KI_zoom2_name = "zoom2"
    badja_zoom=[149.4,150.0, -36.4, -35.99]
    badja_zoom_name="zoom1"
    
    if True:
        mr = "badja_run3"
        zoom=badja_zoom
        zoom_name=badja_zoom_name
        cloud_top_height(mr,
                        extent=zoom,
                        subdir=zoom_name,
                        )

