#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 15:14:27 2020
    Look at surfaces in 3d
        show surface level, and atmospheric stuff in a set of 3d axes
    may require extra modules to be loaded...
    module load gtk
@author: jesse
"""


import matplotlib
#matplotlib.use('Agg',warn=False)

# plotting stuff
import matplotlib.pyplot as plt
import numpy as np
#import warnings
from datetime import datetime, timedelta

# 3d plotting
# isosurface plotting available in plotly
import plotly.graph_objects as go
from plotly.io import show
import plotly.io as pio
# Make browser the default plotly renderer
pio.renderers.default = "browser"

## Run these if running on local laptop:
import sys
if "g/data" not in sys.prefix:
    # turn on orca (server that takes plotly interactive images and saves them to static png)
    pio.orca.ensure_server()
    # check orca status
    pio.orca.status

# local modules
from utilities import plotting, utils, fio, constants

###
## GLOBALS
###
_sn_ = 'threedee'
_verbose_ = True

## Surface heat colour scale for plotly
shcmax=20000.
shcmin=0
shcolor=[
    [0, 'rgb(255,255,255)'],  # start at white
    [100/shcmax, 'rgb(80,80,80)'], # head to grey for minor heat
    [1000/shcmax, 'rgb(255,0,0)'], # head to red for next milestone
    [10000/shcmax, 'rgb(255,0,255)'], # end at purple
    [1.0, 'rgb(0,0,0)'], # approach black off the scale I guess
    ]

###
## METHODS
###

def verbose(*args):
    if _verbose_:
        print("INFO(verbose):",*args)

def cube_to_xyz(cube,
                ztopind=-1):
    """
    take iris cube [lev, lat, lon]
    pull out the data and reshape it to [lon,lat,lev]
    """
    assert len(cube.shape)==3, "cube is not 3-D"
    data = cube[:ztopind,:,:].data.data
    # data is now a level, lat, lon array... change to lon, lat, lev
    xyz = np.moveaxis(np.moveaxis(data,0,2),0,1)
    return xyz
    

def create_figure(gofigures, 
                  camera_eye=[1.4,-1.6,.35], 
                  aspectratio=[1,1,0.8], 
                  filename=None,
                  **layoutargs):
    """
    show or save a figure
    gofigures: list of go objects (e.g., [go.Surface(...),go.Isosurface(...)])
    view: where the camera viewpoint is located
    aspectratio: the aspect ratio (todo: figure out)
    filename: if not None, try to save figure 
    """
    # make figure
    fig = go.Figure(data=gofigures)
    # place camera
    cx,cy,cz=camera_eye
    ax,ay,az=aspectratio
    # default minimal axis labelling
    if 'scene' not in layoutargs:
        layoutargs['scene'] = dict(aspectratio=dict(x=ax,y=ay,z=az),
                                   camera_eye=dict(x=cx,y=cy,z=cz),
                                   xaxis = dict(nticks=3, title='',),
                                   yaxis = dict(nticks=3, title='',),
                                   zaxis = dict(nticks=3, title='',)
                                   ,)
    fig.update_layout(**layoutargs)
    
    if filename is None:
        pio.show(fig)
    else:
        ## Try to save figure
        fio.make_folder(filename)
        print("INFO: Saving Image: ",filename)
        fig.write_image(filename)

def cloud_system(mr='KI_run1_exploratory', hour=1, 
                theta_height=1000, theta_min=311, theta_max=320,
                vert_motion_height = 2500,
                top_height=8000, send_to_browser=False,
                extent=None,
                HSkip=5,
                ltoffset=8):
    """
    Read an hour of model output, plot it in 3d using plotly
    saves output as .png
    ARGUMENTS:
        hour: which output hour (first is 0, last is 23 or -1)
        theta_height: how high are we looking regarding potential temp?
        theta_min, theta_max: min and max potential temperature to draw isosurface
        vert_motion_height: altitude of vertical motion surface,
        height_top: how high (m) included in plot
        send_to_browser: instead of trying to save figures, send one to the browser (interactive)
    """
    
    #hours=fio.hours_available(mr)
    #dtime=hours[hour]
    
    cubes = fio.read_model_run_hour(mr, 
                               hour=hour, 
                               extent=extent, 
                               #add_theta=True, 
                               #add_topog=True, 
                               #add_winds=True,
                               #HSkip=HSkip,
                               )
    utils.extra_cubes(cubes,add_theta=True,add_winds=True,add_z=True)
    
    th, qc = cubes.extract(['potential_temperature','qc'])
    # datetimes in hour output
    cubetimes = utils.dates_from_iris(th)
    
    ff, = fio.read_fire(mr,
                        dtimes=cubetimes,
                        extent=extent,
                        HSkip=HSkip)
    
    # Get the rest of the desired data
    topog, = cubes.extract(['surface_altitude'])
    d_topog = topog.data.data.T # convert to lon,lat
    # set one pixel to -150 to fix color scale
    d_topog[1,1] = -150
    
    u,v,w = cubes.extract(['u','v','upward_air_velocity'])
    levh  = qc.coord('level_height').points
    topind = np.sum(levh<top_height)
    topind_th = np.sum(levh<theta_height)
    # these are level, lat, lon cubes
    lat,lon = qc.coord('latitude').points, qc.coord('longitude').points
    
    # dimensional mesh
    X,Y,Z = np.meshgrid(lon,lat,levh) 
    ## X Y Z are now [lat, lon, lev] for some reason
    [X,Y,Z] = [ np.moveaxis(arr,0,1) for arr in [X,Y,Z]]
    ## Now they are lon, lat, lev
    ## Cut down to desired level
    [X, Y, Z] = [ arr[:,:,:topind] for arr in [X,Y,Z]]
    
    # topography surface
    topog_layer = go.Surface(
        z=Z[:,:,0],
        x=X[:,:,0],
        y=Y[:,:,0],
        colorscale='earth', # was not reversed on local laptop
        #cmin=-150, # make water blue, near zero green, hills brown
        reversescale=True,
        surfacecolor=d_topog,
        showscale=False, # remove colour bar,
    )
    
    namedlocs=[]
    namedlocs_lats = []
    namedlocs_lons = []
    for (namedloc, (loclat, loclon)) in plotting._latlons_.items():
        #print(namedloc, loclat, loclon)
        if loclon < extent[1] and loclon > extent[0] and loclat < extent[3] and loclat > extent[2]:
            if 'fire' not in namedloc and 'pyrocb' not in namedloc:
                namedlocs.append(namedloc)
                namedlocs_lats.append(loclat)
                namedlocs_lons.append(loclon)
    
    # angle for view is 270 degrees - 90* hour/24
    # or 270 - 90 * (60*hour + 60*hi/n_hi)/(24*60)
    
    for hi, cubetime in enumerate(cubetimes):
        #verbose("Creating surfaces")
        camera_eye = [-2 * np.cos(np.deg2rad(290-120*((60*(hour+hi/len(cubetimes))))/(24.0*60))), 
                      2 * np.sin(np.deg2rad(290-120*((60*(hour+hi/len(cubetimes))))/(24.0*60))),
                      0.35]
        #print("DEBUG:", camera_eye)
        
        # get cloud, theta, vert motion, firefront in terms of lon,lat,lev
        d_qc = cube_to_xyz(qc[hi],ztopind=topind)
        d_th = cube_to_xyz(th[hi],ztopind=topind_th)
        d_w = cube_to_xyz(w[hi],ztopind=topind)
        
        #d_ff = ff[hi].data.data # firefront already in lon,lat shape
        
        # surfaces to be plotted in 3d
        surface_list = [topog_layer]
        
        ## Points for waroona, yarloop
        locations_scatter = go.Scatter3d(
            x=namedlocs_lons,
            y=namedlocs_lats,
            z=[0]*len(namedlocs),
            mode='markers',
            marker=dict(
                size=4,
                color='black',           # array/list of desired values
                #colorscale='Viridis',   # choose a colorscale
                opacity=0.8
                ),
            )
        surface_list.append(locations_scatter)
        
        ## atmospheric heat (theta)
        if np.sum(d_th > theta_min) > 0:
            verbose("adding heat surface")
            theta_surf = go.Isosurface(
                z=Z[:,:,:topind_th].flatten(),
                x=X[:,:,:topind_th].flatten(),
                y=Y[:,:,:topind_th].flatten(),
                value=d_th.flatten(),
                isomin=theta_min,
                isomax=theta_max,
                surface_count=4,
                opacity=0.7,
                colorscale='Hot',
                showscale=False
                )
            surface_list.append(theta_surf)
            
        
        ## volume plot showing vertical motion 
        wmax = 7
        wmin = 3
        vm_ind = np.sum(levh<vert_motion_height)
        up_volume = go.Volume(
            x=X[:,:,:vm_ind].flatten(), 
            y=Y[:,:,:vm_ind].flatten(), 
            z=Z[:,:,:vm_ind].flatten(),
            value=d_w[:,:,:vm_ind].flatten(),
            isomin=wmin,
            isomax=wmax,
            opacity=0.1, # max opacity
            surface_count=15,
            colorscale='Reds', # This was PiYG_r on local laptop
            #reversescale=True,  # should be equivalent to _r
            showscale=False, 
        )
        surface_list.append(up_volume)
        
        down_volume = go.Volume(
            x=X[:,:,:vm_ind].flatten(), 
            y=Y[:,:,:vm_ind].flatten(), 
            z=Z[:,:,:vm_ind].flatten(),
            value=d_w[:,:,:vm_ind].flatten(),
            isomin=-wmax,
            isomax=-wmin,
            opacity=0.1, # max opacity
            surface_count=15,
            colorscale='Blues', # This was PiYG_r on local laptop
            reversescale=True,  # should be equivalent to _r
            showscale=False, 
        )
        surface_list.append(down_volume)
        
        ## Cloud isosurface
        qcmin = 0.1
        qcmax = 1
        qc_volume = go.Volume(
            x=X.flatten(), 
            y=Y.flatten(), 
            z=Z.flatten(),
            value=d_qc.flatten(),
            isomin=qcmin,
            isomax=qcmax,
            opacity=0.1, # max opacity
            surface_count=20,
            showscale=False, 
        )
        surface_list.append(qc_volume)
        
        #
        ### 3d figure:
        #
        
        ## title, lables
        lt = cubetime + timedelta(hours=ltoffset)
        layoutargs = dict(
            title={"text":lt.strftime(model_run+' %d %H:%M (lt)'),
                   "yref": "paper",
                   "y" : 0.775,
                   "x" : 0.3,
                   "yanchor" : "bottom",
                   "xanchor" : "left",},
            xaxis_title="",
            yaxis_title="",
            font=dict(
                    family="Courier New, monospace",
                    size=18,
                    color="#222222"
                ),
            #margin=dict(
            #        #l=60,
            #        #r=10,
            #        #b=0,
            #        t=0,
            #    ),
            )

        figname = None
        if not send_to_browser:
            #figname = cubetime.strftime('figures/threedee/test_%Y%m%d%H%M.png')
            figname = fio.standard_fig_name(mr,_sn_,cubetime)
        create_figure(surface_list, filename=figname, camera_eye=camera_eye, **layoutargs)

if __name__=='__main__':
    KI_extent=None
    
    mr="KI_run1_exploratory"
    extent=KI_extent
    # theta limits for 3d plot
    theta_min = 311
    theta_max = 320
    
    # Cloud images
    if True:
        for hour in [1]:
            cloud_system(mr=mr,
                         hour = hour,
                         extent = extent,
                         HSkip = 2,
                         top_height = 13500,
                         #theta_height=1250,
                         #theta_min=sirivan_theta_min,
                         #theta_max=sirivan_theta_max,
                         send_to_browser=True,)
    
            