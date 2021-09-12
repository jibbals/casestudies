# -*- coding: utf-8 -*-
"""
Created on Thu May  6 14:47:04 2021

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
from utilities import plotting, utils, fio

##
## METHODS
###


def plot_vorticity(DA_u,DA_v,DA_ff=None,DA_sh=None,DA_topog=None,
                   **quiverwinds_args):
    """
    """
    # first find latitude and longitude
    if hasattr(DA_u,"longitude"):
        lats=DA_u.latitude.values
        lons=DA_u.longitude.values
    elif hasattr(DA_u,"lon"):
        lats=DA_u.lat.values
        lons=DA_u.lon.values
    else:
        print(DA_u)
        print("ERROR: COULDN'T FIND L(ong/at)ITUDE")
        
    if len(lats) == DA_u.shape[1]:
        u = DA_u.values.T
        v = DA_v.values.T
    else:
        u = DA_u.values
        v = DA_v.values
    
    # defaults for quivers
    if "n_arrows" not in quiverwinds_args:
        quiverwinds_args['n_arrows'] = 15
    if "alpha" not in quiverwinds_args:
        quiverwinds_args['alpha'] = .5
    
    # Relative vorticity)
    rel_vor = utils.dragana_vorticity(lats,lons,u,v)
    
    ## color map something
    avo_con = 1e-4*2**np.arange(0,9)
    avo_con = np.union1d(avo_con,-avo_con)
    avo_con = np.union1d(np.array([0]),avo_con)
    avo_map = plt.cm.RdBu_r
    if matplotlib.__version__ > '3.3.0':
        avo_norm = matplotlib.colors.SymLogNorm(2e-4,base=10)
    else:
        avo_norm = matplotlib.colors.SymLogNorm(2e-4,)#base=10)
    
    # Modify colormap to have white for certain range
    n = 90
    x = 0.5
    lower = avo_map(np.linspace(0, x, n))
    white = avo_map(np.ones(100)*0.5)
    upper = avo_map(np.linspace(1-x, 1, n))
    colors = np.vstack((lower, white, upper))
    tmap = matplotlib.colors.LinearSegmentedColormap.from_list('terrain_map_white', colors)

    # Create figure and subplots

    # Vorticity
    #fig = plt.figure(1,figsize=(12,7))
    #ax = plt.subplot(1,1,1,aspect='equal') # One subplot
    rv = plt.contourf(lons,lats,rel_vor,avo_con,
                      cmap=tmap,
                      norm=avo_norm,
                      extend='both',
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
    
    # Add winds to plot
    plotting.quiverwinds(lats,lons,u,v,
                         **quiverwinds_args)

    # topography
    if DA_topog is not None:
        topog_con = (2,200,400,800)
        if len(lats) == DA_topog.shape[1]:
            topog=DA_topog.values.T
        else:
            topog = DA_topog.values
        plt.contour(lons,lats,topog,topog_con)
    
    return rv

def vorticity_10m(
        mr,
        extent=None,
        subdir=None,
        timeskip=10,
        ):
    """
    
    ARGUMENTS:
        mr: model run name
        hours: optional run for subset of model hours
        extent: subset extent
        subdir: savefolder in case of specific extent
    """
    
    
    # read fire model output
    DS_fire=fio.read_model_run_fire(mr, extent=extent)
    
    lats = DS_fire.lat.values
    lons = DS_fire.lon.values
    times = DS_fire.time.data[::timeskip] # np.datetime64 array
    
    DA_u10 = DS_fire['UWIND_2'][::timeskip]
    DA_v10 = DS_fire['VWIND_2'][::timeskip]
    DA_ff = DS_fire['firefront'][::timeskip]
    
        
    houroffset = utils.local_time_offset_from_lats_lons(lats,lons)
    localtimes = utils.local_time_from_time_lats_lons(times,lats,lons)
    # loop over timesteps
    for ti,time_utc in enumerate(times):
            
        ## get local time
        time_lt=localtimes[ti]
        time_str=time_lt.strftime("%dT%H:%M")+"(UTC+%.2f)"%houroffset
                        
        ## FIRST FIGURE: 10m WIND DIR:
        plt.figure()
        
        rv=plot_vorticity(DA_u10[ti], DA_v10[ti], DA_ff[ti], )#topog)
        # Add colorbar to the plot
        plt.colorbar(rv) 
        # add title, set aspect
        plt.title(mr + "\nvorticity (10m) " + time_str)
        plt.gca().set_aspect("equal")
            
        # save figure
        fio.save_fig(mr,"vorticity_10m", time_utc, plt, subdir=subdir)
        
        

def vorticity(mr, extent=None, subdir=None, levels=[3,10,20,30,40, 50,60,70,90], coastline=2):
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
        
    #times=DS_fire.time.data
    #localtimes = utils.local_time_from_time_lats_lons(times,lats,lons)
    #houroffset=utils.local_time_offset_from_lats_lons(lats,lons)
    
    # loop over timesteps
    hours=fio.hours_available(mr)
    for hi,hour_utc in enumerate(hours):
        
        ## slice time
        DS_atmos = fio.read_model_run_hour(mr,hour=hi)
        times_utc = DS_atmos.time.values
        
        # read x and y winds
        DA_x = DS_atmos['wnd_ucmp']
        DA_y = DS_atmos['wnd_vcmp']
        # destagger x and y winds
        DA_u,DA_v = utils.destagger_winds_DA(DA_x,DA_y)
        # subset to extent
        if extent is not None:
            DA_u = fio.extract_extent(DA_u, extent)
            DA_v = fio.extract_extent(DA_v, extent)
        
        for ti, time_utc in enumerate(times_utc):
            DS_fire_slice = DS_fire.sel(time=time_utc)
            DA_ff  = DS_fire_slice['firefront']
            DA_sh  = DS_fire_slice['SHEAT_2']
            
            ## get local time
            time_lt = utils.local_time_from_time_lats_lons(time_utc,lats,lons)
            houroffset = utils.local_time_offset_from_lats_lons(lats,lons)
            time_str=time_lt.strftime("%dT%H:%M")+"(UTC+%.2f)"%houroffset
                            
            ## FIRST FIGURE: 10m WIND DIR:
            fig = plt.figure(
                figsize=[11,11],
                )
            #fig_grid = fig.add_gridspec(3, 3, wspace=0, hspace=0)
            for li,level in enumerate(levels):
                plt.subplot(3,3,1+li, aspect="equal")
                if li == 0:
                    in_sh = DA_sh
                    in_ff = None
                else:
                    in_sh = None
                    in_ff = DA_ff
                cs_rv = plot_vorticity(DA_u[ti,li],DA_v[ti,li],
                        DA_sh=in_sh,
                        DA_ff=in_ff,
                        add_quiver_key=(li==0),
                        n_arrows=10,
                        )
                # add model level height average to each subplot
                plt.text(0.01,0.01, # bottom left using axes coords
                        "%.2f m"%DS_atmos.level_height[level].values,
                        fontsize=9,
                        transform=plt.gca().transAxes,# bottom left using axes coords
                        zorder=10,
                        )
                plotting.map_add_locations_extent(extent,hide_text=True)
                if coastline>0 and np.min(DA_topog.values)<coastline:
                    plt.contour(lons,lats,DA_topog.values, np.array([coastline]),
                                colors='k')
                #plt.gca().set_aspect("equal")
                plt.gca().set(xticks=[],yticks=[],aspect="equal")
    
            plt.suptitle(mr+" "+time_str)

            plt.subplots_adjust(
                    wspace = 0.0,  # the amount of width reserved for space between subplots,
                    hspace = 0.0,
                    )
            #plt.tight_layout()
            
            # add space in specific area, then add vert wind colourbar
            cbar_ax = fig.add_axes([0.05, 0.975, 0.25, 0.015]) # X Y Width Height
            vort_ticks=[-1.28e-2, -0.16e-2, 0, 0.16e-2, 1.28e-2]
            vort_ticks_str=["-1.28e-2", "-0.16e-2", "0", "0.16e-2", "1.28e-2"]
            cbar = fig.colorbar(cs_rv, 
                                cax=cbar_ax, 
                                ticks=vort_ticks, 
                                pad=0,
                                orientation='horizontal',
                                )
            cbar.ax.set_xticklabels(vort_ticks_str,rotation=20)
            # save figure
            fio.save_fig(mr,"vorticity", time_utc, plt, subdir=subdir)

if __name__ == '__main__':

    # keep track of used zooms
    KI_zoom_name = "zoom1"
    KI_zoom = constants.extents['KI'][KI_zoom_name]
    KI_zoom2_name = "zoom2"
    KI_zoom2 = constants.extents['KI'][KI_zoom2_name]
    badja_zoom_name="zoom1"
    badja_zoom=constants.extents['badja'][badja_zoom_name]
    
    if True:
        mr = "KI_run2"
        zoom=KI_zoom2
        zoom_name=KI_zoom2_name
        vorticity(mr,
                  extent=zoom,
                  subdir=zoom_name,
                  #levels=[1,2,3,5,8,10,12,14,18], # lower levels for exploratory
                  )

    if False:
        mr="badja_run3"
        zoom=badja_zoom
        zoom_name=badja_zoom_name
        vorticity_10m(mr,zoom,zoom_name)
