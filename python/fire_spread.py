# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 16:51:15 2021
    Grey area where fire already burnt, 
    heat flux and 10m winds overlaid
    Run on badja zoom1:   
        CPU Time Used: 04:15:40  
        Memory Used: 7.41GB    
        Walltime Used: 04:15:54  

@author: jgreensl
"""

import matplotlib
#matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import dask # for a warning blocker

#import xarray as xr
#import cartopy.crs as ccrs
#from matplotlib import colors
#from datetime import datetime, timedelta
from utilities import fio, utils, plotting, constants


def isochrone_comparison(mrs, extent=None, subdir=None, minute_increment=240):
    """
    minute_increment is for dashed lines
    """
    # colours from first 80% of gnuplot
    colors = plt.cm.gnuplot(np.linspace(0,0.8,num=len(mrs)))
    
    DS_fires = {mr:fio.read_model_run_fire(mr) for mr in mrs}
    DA_topog = fio.model_run_topography(mrs[0])
    
    if extent is not None:
        for mr in mrs:
            latsi=DS_fires[mr].lat.data
            lonsi=DS_fires[mr].lon.data
            print("DEBUG: %s before spatial subset:"%mr)
            print(DS_fires[mr].dims)
            print(np.min(lonsi),np.max(latsi),np.min(latsi),np.max(latsi))
            #print(DS_fires[mr].coords)
            DS_fires[mr] = fio.extract_extent(DS_fires[mr],extent)
            print("DEBUG: %s after spatial subset:"%mr)
            print(DS_fires[mr].dims)
            latsi=DS_fires[mr].lat.data
            lonsi=DS_fires[mr].lon.data
            print(np.min(lonsi),np.max(lonsi),np.min(latsi),np.max(latsi))
            
        DA_topog = fio.extract_extent(DA_topog,extent)
    
    DA_FFs = {mr:DS_fires[mr]['firefront'] for mr in mrs}
    if extent is None:
        extent=[lons[0],lons[-1],lats[0],lats[-1]]
    
    times=DS_fires[mrs[0]].time.data
    lats0 = DS_fires[mrs[0]].lat.data
    lons0 = DS_fires[mrs[0]].lon.data
    # title from local time at fire ignition
    time_lt = utils.local_time_from_time_lats_lons(times,lats0,lons0)
    
    ## Plot starts here, we put isochrones onto topography
    plotting.map_topography(DA_topog.values,lats0,lons0,cbar=True)
    
    # loop over timesteps after fire starts
    # This suppresses a warning when reading large arrays
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        hasfire=np.min(DA_FFs[mrs[0]].values,axis=(1,2)) < 0

        for ti,time_utc in enumerate(times[hasfire][::minute_increment]): # every N minutes once fire starts
        
            for mr, color in zip(mrs,colors):
            
                DA_FF = DA_FFs[mr]
                lats=DA_FF.lat.data
                lons=DA_FF.lon.data
                # another chunk warning
                FF = DA_FF.sel(time=time_utc).data.T
                #print("DEBUG: FF, lats, lons",FF.shape, lats.shape, lons.shape)
                plotting.map_fire(FF,lats,lons, 
                              linewidths=1,
                              colors=[color],
                              linestyles='--',
                              )

        # Finally do last timestep
        for mr, color in zip(mrs,colors):
            DA_FF = DA_FFs[mr]
            lats=DA_FF.lat.data
            lons=DA_FF.lon.data
            ## slice time
            FF = DA_FF.sel(time=times[-1]).data.T
            plotting.map_fire(FF,lats,lons, 
                              linewidths=2,
                              colors=[color],
                              )
    
    # Add legend
    lines = [matplotlib.lines.Line2D([0], [0], color=c, linewidth=2, linestyle='-') for c in colors]
    mrs_names=[mrs_name.split("_")[-1] for mrs_name in mrs]
    plt.legend(lines, mrs_names, 
            bbox_to_anchor=(-.02, 0.9), # put legend top left (above figure)
            loc='lower left', # for bbox connection
            handlelength=1, # default is 2, line length in legend is too long
            ncol=2, # two columns
            )
    
    plotting.map_add_locations_extent(extent,hide_text=False)
    
    title=np.array(time_lt)[hasfire][0].strftime("Isochrones from %H%M")
    plt.title(title)
        
    plt.gca().set_aspect("equal")

    # save figure
    if subdir is None:
        subdir = str(mrs)
    fio.save_fig_to_path("../figures/isochrone_comparison/"+subdir+".png", plt)

def isochrones(mr, extent=None, subdir=None, labels=False):
    """
    """
    DS_fire = fio.read_model_run_fire(mr)
    DA_topog = fio.model_run_topography(mr)
    
    if extent is not None:
        DS_fire = fio.extract_extent(DS_fire,extent)
        DA_topog = fio.extract_extent(DA_topog,extent)
        if subdir is None:
            subdir=str(extent)
    
    DA_FF = DS_fire['firefront']
    lats=DS_fire.lat.data
    lons=DS_fire.lon.data
    if extent is None:
        extent=[lons[0],lons[-1],lats[0],lats[-1]]
    
    times=DS_fire.time.data
    # title from local time at fire ignition
    time_lt = utils.local_time_from_time_lats_lons(times,lats,lons)
    
    ## Plot starts here, we put isochrones onto topography
    plotting.map_topography(DA_topog.values,lats,lons)
    
    # loop over timesteps after fire starts
    hasfire=np.min(DA_FF.values,axis=(1,2)) < 0
    for ti,time_utc in enumerate(times[hasfire][::60]):
        
        ## slice time
        FF = DA_FF.sel(time=time_utc).data.T
        
        #ffcontour = 
        plotting.map_fire(FF,lats,lons, linewidths=0.5+((ti%4)==0))
        
        #if labels and ((ti%4) == 0):
        #    lt_stamp0=utils.local_time_from_time_lats_lons(time_utc,lats,lons)
        #    lt_stamp=lt_stamp0.strftime("%H%M")
        #    plt.clabel(ffcontour, fmt=lt_stamp, colors='k', inline=True)
                
                
    plotting.map_add_locations_extent(extent,hide_text=False)
    
    title=np.array(time_lt)[hasfire][0].strftime("Fire isochrones (first at %H%M)")
    plt.title(title)
        
    plt.gca().set_aspect("equal")

    # save figure
    fio.save_fig(mr,"isochrones", mr, plt, subdir=subdir)
    
def plot_fire_speed(DA_fs, DA_ff, DA_u, DA_v, **contourfargs):
    """
    Note from Harvey:
        The speed will be zero before the fire start. 
        And its minimum will be 0.001 (I think) after the fire start. 
        I believe the issue is not harmful, but worth fixing.
    ARGS (xr dataarrays):
        firespeed[[t,]lat,lon],
        firefront[lat,lon],
        u[lat,lon],
        v[lat,lon],
        plt.contourf kwargs can be passed in
    """
    lats=DA_u.lat.values
    lons=DA_u.lon.values
    
    # default contourfargs
    if 'cmap' not in contourfargs:
        contourfargs['cmap'] = "gist_stern_r"
    if 'extend' not in contourfargs:
        contourfargs['extend'] = 'both'

    # take maximum if there is time dim
    if len(DA_fs.shape)==3:
        DA_fs = DA_fs.max(dim='time')

    if len(lats) == DA_u.shape[1]:
        u = DA_u.values.T
        v = DA_v.values.T
        fs= DA_fs.values.T
        ff= DA_ff.values.T
    else:
        u = DA_u.values
        v = DA_v.values
        fs= DA_fs.values
        ff= DA_ff.values
    
    # burnt area
    ## Do the filled contour plot
    fspeed_levels=np.linspace(0,1,21)
    plt.contourf(lons, lats, fs,
                 fspeed_levels, # color levels
                 **contourfargs,
                 )
                
    plt.colorbar()

    plotting.map_fire(ff,lats,lons)
    
    plotting.quiverwinds(lats,lons,u,v,
            n_arrows=20,
            )
    

def plot_fire_spread(DA_sh, DA_ff, DA_u, DA_v):
    """
    """
    if hasattr(DA_u,"longitude"):
        lats=DA_u.latitude
        lons=DA_u.longitude
    elif hasattr(DA_u,"lon"):
        lats=DA_u.lat.values
        lons=DA_u.lon.values
    else:
        print(DA_u)
        print("ERROR: COULDN'T FIND L(ong/at)ITUDE")
        
    if len(lats) == DA_u.shape[1]:
        u = DA_u.values.T
        v = DA_v.values.T
        ff= DA_ff.values.T
        sh= DA_sh.values.T
    else:
        u = DA_u.values
        v = DA_v.values
        ff= DA_ff.values
        sh= DA_sh.values
    #ws = utils.wind_speed(u,v)
    #print(np.min(ff),np.max(ff))
    # min is -0.0316...
    # max is .0084
    
    # burnt area
    if np.min(ff)<-0.02:
        #    contourfargs['norm']=col.LogNorm()
        burn_levels=[-.07,-.0001,0]
        ## Do the filled contour plot
        plt.contourf(lons, lats, ff,
                     burn_levels, # color levels
                     #vmax=-.01,
                     cmap='Greys_r',
                     #alpha=0.8,
                     )
        
    #plt.imshow(lons,lats,masked, interpolation='none')
    # heat flux
    if np.max(sh)>100:
        plotting.map_sensibleheat(sh,lats,lons,colorbar=True)
    
    # quiver
    plotting.quiverwinds(lats,lons,u,v,)
    
    

def fire_spread(mr, extent=None, subdir=None, coastline=2, timesteps=5, firespeed=True):
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
    times=DS_fire.time.data
    
    houroffset=utils.local_time_offset_from_lats_lons(lats,lons)
    # loop over timesteps
    # interesting times don't begin before 3 hours in any run
    if "exploratory" in mr:
        ind_interest = np.arange(0,len(times),10)
        times_of_interest=times[::10]
    elif "AF" in mr:
        ind_interest = np.arange(0,len(times),timesteps)
        times_of_interest=[times[i] for i in ind_interest]
    else:
        ind_interest = np.union1d([0,29,59,89,119,149,],np.arange(179,24*60-1,timesteps))
        times_of_interest=[times[i] for i in ind_interest]
        
    for ti,time_utc in enumerate(times_of_interest):
        
        ## slice time
        DS_fire_slice = DS_fire.sel(time=time_utc)
        DA_u10 = DS_fire_slice['UWIND_2'] # m/s east west
        DA_v10 = DS_fire_slice['VWIND_2'] # m/s south north
        DA_ff  = DS_fire_slice['firefront'] # na
        DA_sh  = DS_fire_slice['SHEAT_2'] # W/m2
        DA_fs  = DS_fire_slice['fire_speed'] # m/s
        
        ## also look at fire speed up until current time
        DS_fire_slice_upto = DS_fire.isel(time=np.arange(ind_interest[ti]+1))
        DA_fs_upto = DS_fire_slice_upto['fire_speed'] # m/s has time dim
        
        ## get local time
        time_lt = utils.local_time_from_time_lats_lons(time_utc,lats,lons)
        #time_str=time_lt.strftime("%Y%m%d %H%M")+"(UTC+%.2f)"%houroffset
        time_str=time_lt.strftime("%d %H%M")+"(UTC+%.2f)"%houroffset
        title = mr+"\n"+time_str
        ## FIRST FIGURE: fire spread
        
        plot_fire_spread(DA_sh, DA_ff, DA_u10, DA_v10)
        plotting.map_add_locations_extent(extent,hide_text=True, fontsizes=13)
        plt.title(title)
        
        if coastline>0 and np.min(DA_topog.values)<coastline:
            plt.contour(lons,lats,DA_topog.values, np.array([coastline]),
                        colors='k')
        plt.gca().set_aspect("equal")

        # save figure
        fio.save_fig(mr,"fire_spread", time_utc, plt, subdir=subdir)

        ## SECOND FIGURE: fire speed
        if firespeed:
            plot_fire_speed(DA_fs, DA_ff, DA_u10, DA_v10)
            plotting.map_add_locations_extent(extent, hide_text=True)
            plt.title(title)
            
            if coastline>0 and np.min(DA_topog.values)<coastline:
                plt.contour(lons,lats,DA_topog.values, np.array([coastline]),
                            colors='k')
            plt.gca().set_aspect("equal")

            # save figure
            fio.save_fig(mr,"fire_speed", time_utc, plt, subdir=subdir)

            ## Third FIGURE: fire speed maximum
            plot_fire_speed(DA_fs_upto, DA_ff, DA_u10, DA_v10)
            plotting.map_add_locations_extent(extent, hide_text=False)
            plt.title(title)
            
            if coastline>0 and np.min(DA_topog.values)<coastline:
                plt.contour(lons,lats,DA_topog.values, np.array([coastline]),
                            colors='k')
            plt.gca().set_aspect("equal")

            # save figure
            fio.save_fig(mr,"fire_speed_max", time_utc, plt, subdir=subdir)

def suitecall(mr, extent=None, subdir=None):
    """ fire spread plot and isochrones? """
    isochrones(mr, extent=extent, subdir=subdir)
    fire_spread(mr, extent=extent, subdir=subdir)


if __name__ == '__main__':
    # keep track of used zooms
    KI_zoom_name = "zoom1"
    KI_zoom = constants.extents['KI'][KI_zoom_name]
    KI_zoom2_name = "zoom2"
    KI_zoom2 = constants.extents['KI'][KI_zoom2_name]
    badja_zoom_name="zoom1"
    badja_zoom=constants.extents['badja'][badja_zoom_name]
    sirivan_extent = [149.2, 150.4, -32.4, -31.6]
    sirivanz_extent = [149.4, 150.19, -32.2, -31.8]
    harvey_test_runs=["AF_coupled","AF_uncoupled","AF_coupled-spark","AF_uncoupled-spark"]

    # Check am1 vs am2
    if False:
        extent = constants.extents['badja_am']['zoom2']
        isochrone_comparison(['badja_am1','badja_am3','badja_am4','badja_am5'],
                extent=extent,
                minute_increment=60,
                subdir="badja_high_res_runs",
                )
    # check harveys runs in more detail    
    if False:
        for mr in harvey_test_runs:
            fire_spread(mr=mr, extent=sirivanz_extent, subdir="sirivanz",
                    timesteps=12, firespeed=False)

    if True: 
        mr='wooroloo_run1'
        extent=constants.extents['wooroloo']['zoom1']
        subdir='zoom1'

        isochrones(mr=mr, extent=extent, subdir=subdir)
        fire_spread(mr=mr,
                extent=extent,
                subdir=subdir,
                timesteps=10,
                firespeed=False)

    if False:
        #mrs=["badja_run1","badja_run2","badja_run3","badja_run4"]
        mrs=["badja_run3","badja_UC1"]
        extent=badja_zoom
        subdir="uncoupled" #badja_zoom_name
        #mrs=['KI_run1','KI_run2','KI_run3']
        #extent=KI_zoom
        #subdir=KI_zoom_name

        isochrone_comparison(mrs,extent=extent,subdir=subdir)
        
    
    if False:
        #mrs=["badja_run1","badja_run2","badja_run3","badja_run4"]
        mrs=["badja_LRC1","badja_UC1"]
        extent=None # badja_zoom
        subdir=None # badja_zoom_name
        #mrs=['KI_run1','KI_run2','KI_run3']
        #extent=KI_zoom
        #subdir=KI_zoom_name
        for mr in mrs:
            isochrones(mr, extent=extent, subdir=subdir)
    
    #mr = 'badja_run3'
    #zoom = badja_zoom
    #zoom_name=badja_zoom_name
    #fire_spread(mr,zoom,zoom_name)
