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
from utilities import plotting, fio, utils, constants
#from utilities import fio_iris as fio

###
## GLOBALS
###

_layout_args_=dict(
    width = 1000,
    height = 1000,
    scene = dict(aspectratio=dict(x=1,y=1,z=.8),
                 camera_eye=dict(x=.4,y=-.5,z=.35),
                 xaxis = dict(nticks=3, title='',),
                 yaxis = dict(nticks=3, title='',),
                 zaxis = dict(nticks=3, title='',),
                 ),
    font = dict(family="Courier New, monospace",
                size=18,
                color="#222222"
                ),
    )

_topog_defaults_=dict(
    colorscale='earth', # was not reversed on local laptop
    #cmin=-150, # make water blue, near zero green, hills brown
    reversescale=True,
    showscale=False, # remove colour bar,
    )

_wind_weak_surf_defaults_=dict(
    isomin=15, # 5 m/s wind speed min
    isomax=20,
    surface_count=2, # N contours between min and max
    opacity=0.15,
    colorscale='Blues',
    showscale=False,
    )
_wind_strong_surf_defaults_=dict(
    isomin=22, # 5 m/s wind speed min
    isomax=30,
    surface_count=3, # N contours between min and max
    opacity=0.5,
    colorscale='viridis',
    showscale=False,
    )

_upmotion_surf_defaults_=dict(
    isomin=3, # m/s
    isomax=9,
    surface_count=3,  # 3, 6, 9 m/s
    opacity=0.2,
    colorscale='ylorrd',
    showscale=False,
    )
_downmotion_surf_defaults_=dict(
    isomin=-9, # m/s opposite of upmotion
    isomax=-3,
    reversescale=True,
    surface_count=3, 
    opacity=0.2,
    colorscale='gnbu',
    showscale=False,
    )

_vorticity_pos_surf_defaults_=dict(
    isomin=2, # units fpr OWZ?
    isomax=5,
    surface_count=2, 
    opacity=0.6,
    colorscale='hot_r',
    showscale=False,
    )
_vorticity_neg_surf_defaults_=dict(
    isomin=-5, # units of zeta...
    isomax=-2,
    reversescale=True,
    surface_count=2, 
    opacity=0.6,
    colorscale='ylgnbu',
    showscale=False,
    )

_theta_surf_defaults_=dict(
        isomin=318, # Kelvin for showing fire area
        isomax=330, # near surface 
        surface_count=10, #
        opacity=0.3,
        colorscale='Hot',
        showscale=False,
        )

# What will rotation look like in the negative space
_rotation_strong_neg_surf_defaults_=dict(
        isomin=-1e-3, # units s-2
        isomax=-1e-4, 
        surface_count=2, #
        opacity=0.7,
        colorscale='Blues',
        showscale=False,
        )
_rotation_weak_neg_surf_defaults_=dict(
        isomin=-1e-4, # units s-2
        isomax=-5e-5, 
        surface_count=2, #
        opacity=0.15,
        colorscale='Greens',
        showscale=False,
        )
_rotation_weak_pos_surf_defaults_=dict(
        isomin=5e-5, # units s-2
        isomax=1e-4, 
        surface_count=2, #
        opacity=0.15,
        colorscale='YlOrRd',
        showscale=False,
        )
_rotation_strong_pos_surf_defaults_=dict(
        isomin=1e-4, # units s-2
        isomax=1e-3, 
        surface_count=2, #
        opacity=0.7,
        colorscale='hot_r',
        showscale=False,
        )


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

def get_camera_eye(hour, i_h, list_h, start=245, z=0.35):
    """
    hour: integer current hour (normally 0 to 23)
    i_h: integer for what time step are we up to within the hour (normally 0 to 6)
    list_h: list of all available timesteps for this hour (normally 3 or 6)
    start=290 : initial angle for camera eye: 270 is from the south, 180 should be from east..
    z=0.35 : initial altitude for camera eye: .35 is like a 3rd of the axis (I think that's how it works)
    """
    
    angle = np.deg2rad(start-120*(hour+i_h/len(list_h))/(24.0))
    ce = {
        'x':-2 * np.cos(angle), 
        'y':2 * np.sin(angle),
        'z':0.35,
        }
    # ce = {
    #     'x':-2 * np.cos(np.deg2rad(290-120*(hour+i_h/len(list_h))/(24.0))), 
    #     'y':2 * np.sin(np.deg2rad(290-120*(hour+i_h/len(list_h))/(24.0))),
    #     'z':0.35,
    #     }
    return ce


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
    data = cube[:ztopind,:,:].data
    # data is now a level, lat, lon array... change to lon, lat, lev
    xyz = np.moveaxis(np.moveaxis(data,0,2),0,1)
    return xyz

def latlonlev_to_xyz(lat,lon,levh):
    # dimensional mesh
    X,Y,Z = np.meshgrid(lon,lat,levh) 
    ## X Y Z are now [lat, lon, lev] for some reason
    [X,Y,Z] = [ np.moveaxis(arr,0,1) for arr in [X,Y,Z]]
    ## Now they are lon, lat, lev
    return X,Y,Z

def create_figure(gofigures, 
                  #camera_eye=[1.4,-1.6,.35], 
                  camera_eye=None, #[.4,-.5,.35],
                  #aspectratio=[1,1,0.8], 
                  filename=None,
                  **layoutargs):
    """
    show or save a figure
    gofigures: list of go objects (e.g., [go.Surface(...),go.Isosurface(...)])
    view: where the camera viewpoint is located
    aspectratio: the aspect ratio (todo: figure out)
    filename: if not None, try to save figure 
    """
    for k,v in _layout_args_.items():
        if k not in layoutargs:
            layoutargs[k]=v
    if camera_eye is not None:
        if 'scene' in layoutargs:
            layoutargs['scene']['camera_eye']=camera_eye
        else:
            layoutargs['scene']=dict(camera_eye=camera_eye)
    # make figure
    fig = go.Figure(data=gofigures)
    fig.update_layout(**layoutargs)
    
    if filename is None:
        pio.show(fig)
    else:
        ## Try to save figure
        fio.make_folder(filename)
        print("INFO: Saving Image: ",filename)
        fig.write_image(filename)

# TODO: Remove default surf args, just send them in as surf_args
def isosurface_wrapper(lat,lon,levh,data, 
                       topind=-1, method = go.Isosurface,
                       default_surf_args = None, **surf_args):
    
    if default_surf_args is not None:
        for k,v in default_surf_args.items():
            if k not in surf_args:
                surf_args[k]=v
    
    x,y,z = latlonlev_to_xyz(lat,lon,levh)
    if "value" not in surf_args:
        surf_args["value"] = cube_to_xyz(data,ztopind=topind)
    
    val=surf_args['value']
    print("DEBUG: value to be isosurfaced: type, shape, min,max",type(val),np.shape(val),np.nanmin(val),np.nanmax(val))
    verbose("adding surface")
    surf = method(
        x=x[:,:,:topind].flatten(),
        y=y[:,:,:topind].flatten(),
        z=z[:,:,:topind].flatten(),
        #value=data_xyz.flatten(),
        **surf_args
        )
    return surf


def title_in_layout(title):
    layoutargs = dict(title={"text":title, #lt.strftime(mr+' %d %H:%M (lt)'),
                             "yref": "paper",
                             "y" : 0.775,
                             "x" : 0.3,
                             "yanchor" : "bottom",
                             "xanchor" : "left",
                             },
                      )
    return layoutargs

def topog_surface(lat, lon, levh, topog, **surf_args):
    """
    ARGS:
        lat,lon,levh: 1D dims
        topog [lat,lon]: 2D array
        arguments for go.Surface optional
    """
    for k,v in _topog_defaults_.items():
        if k not in surf_args:
            surf_args[k]=v
    
    x,y,z = latlonlev_to_xyz(lat,lon,levh)
    data_xy = topog.T
    verbose("adding topog")
    surf = go.Surface(
        x=x[:,:,0],
        y=y[:,:,0],
        z=z[:,:,0],
        surfacecolor=data_xy,
        **surf_args
        )
    return surf
    
def theta_surface(lat,lon,levh, data, topind=5, **surf_args):
    #print("DEBUG:",type(lat),type(lon),type(levh),type(data),type(topind),type(_theta_surf_defaults_),type(surf_args))
    return isosurface_wrapper(lat,lon,levh,data,topind,
        default_surf_args = _theta_surf_defaults_,
        **surf_args)

def up_down_drafts(lat,lon,levh,data,topind,up_args=None,down_args=None):
    """
    RETURNS: upwards isosurface, downwards isosurface
    """
    if up_args is None:
        up_args = {}
    if down_args is None:
        down_args = {}

    up_surf = isosurface_wrapper(lat,lon,levh,data,topind, 
        default_surf_args=_upmotion_surf_defaults_, **up_args)
    down_surf = isosurface_wrapper(lat,lon,levh, data,topind,
        default_surf_args=_downmotion_surf_defaults_, **down_args)
    return up_surf,down_surf
            
def wind_system(mr='KI_run2_exploratory', hour=2, 
                top_height=5000, 
                send_to_browser=False,
                extent=None,
                #HSkip=5,
                #ltoffset=8
                **iso_surf_args,
                ):
    """
    Read an hour of model output, plot it in 3d using plotly
    saves output as .png
    ARGUMENTS:
        hour: which output hour (first is 0, last is 23 or -1)
        ws_min,ws_max: windspeed min and max vals
        height_top: how high (m) included in plot
        send_to_browser: instead of trying to save figures, send one to the browser (interactive)
    """
    
    #hours=fio.hours_available(mr)
    #dtime=hours[hour]
    
    DS = fio.read_model_run_hour(mr, 
                               hour=hour, 
                               extent=extent, 
                               )
    # TODO: use extra_data_read instead of calculating on the fly. (subset issues)
    #utils.extra_DataArrays(DS,add_winds=True,add_theta=True)
    extras = fio.extra_data_read(mr,hour,extent=extent)

    # Get the rest of the desired data
    topog = DS['surface_altitude']
    d_topog = topog.load().data # lat,lon
    
    w = DS['vertical_wnd']
    #ws = DS['s']
    #th = DS['potential_temperature']
    ws = extras['s']
    th = extras['potential_temperature']
    
    # datetimes in hour output
    DAtimes=w.time.data
    lats=w.latitude.data
    lons=w.longitude.data
    # title from local time at fire ignition
    time_lt = utils.local_time_from_time_lats_lons(DAtimes,lats,lons)
    offset_lt = utils.local_time_offset_from_lats_lons(lats,lons)
    
    if "level_height" in w:
        levh  = w.level_height.load().data
    else:
        levh  = w.level_height_0.load().data
    
    # these are level, lat, lon cubes
    lat,lon = w.latitude.data, w.longitude.data
    
    if extent is None:
        extent=[lon[0],lon[-1],lat[0],lat[-1]]
    
    # index of highest level to plot
    topind = np.sum(levh<top_height)
    
    # topography surface
    topog_layer = topog_surface(lat,lon,levh,d_topog)
    
    for hi, DAtime in enumerate(DAtimes):
        # angle for view is 270 degrees - 120 * hour/24
        # or 270 - 120 * (hour + hi/n_hi)/(24)
        camera_eye = get_camera_eye(hour,hi,DAtimes,start=180)
        
        # surfaces to be plotted in 3d
        surface_list = [topog_layer]
        
        # heat
        theta_surf = theta_surface(lat,lon,levh, th[hi])
        surface_list.append(theta_surf)
        
        ## wind speed (horizontal) layers
        ws_surf1 = isosurface_wrapper(lat,lon,levh, ws[hi], topind, _wind_weak_surf_defaults_, **iso_surf_args)
        ws_surf2 = isosurface_wrapper(lat,lon,levh, ws[hi], topind, _wind_strong_surf_defaults_, **iso_surf_args)
        surface_list.append(ws_surf1)
        surface_list.append(ws_surf2)
        
        ## title for layout args
        title="%s (UTC+%.1f)"%(time_lt[hi].strftime("%dT%H:%M"),offset_lt)
        layoutargs = title_in_layout(title)

        figname = None
        if not send_to_browser:
            #figname = cubetime.strftime('figures/threedee/test_%Y%m%d%H%M.png')
            figname = fio.standard_fig_name(mr,_sn_,DAtime,subdir="windspeed")
        create_figure(surface_list, filename=figname, camera_eye=camera_eye, **layoutargs)

def vertmotion_system(mr, hour=2, 
                top_height=5000, 
                send_to_browser=False,
                extent=None,
                up_args=None,
                down_args=None,
                ):
    """
    Read an hour of model output, plot it in 3d using plotly
    saves output as .png
    ARGUMENTS:
        hour: which output hour (first is 0, last is 23 or -1)
        ws_min,ws_max: windspeed min and max vals
        height_top: how high (m) included in plot
        send_to_browser: instead of trying to save figures, send one to the browser (interactive)
    """
    
    DS = fio.read_model_run_hour(mr, 
                               hour=hour, 
                               extent=extent, 
                               )
    # TODO: use extra_data_read instead of calculating on the fly. (subset issues)
    extras = fio.extra_data_read(mr, hour=hour, extent=extent)
    #utils.extra_DataArrays(DS,add_theta=True)
    
    # Get the rest of the desired data
    topog = DS['surface_altitude']
    d_topog = topog.load().data # lat,lon
    # set one pixel to -150 to fix color scale
    #d_topog[1,1] = -150
    
    #th = DS['potential_temperature']
    th = extras['potential_temperature']
    w = DS['vertical_wnd']
    
    # datetimes in hour output
    DAtimes=w.time.data
    lats=w.latitude.data
    lons=w.longitude.data
    # title from local time at fire ignition
    time_lt = utils.local_time_from_time_lats_lons(DAtimes,lats,lons)
    offset_lt = utils.local_time_offset_from_lats_lons(lats,lons)
    
    # set one pixel to -150 to fix color scale
    #d_topog[1,1] = -150
    
    if "level_height" in w:
        levh  = w.level_height.load().data
    else:
        levh  = w.level_height_0.load().data
    
    topind = np.sum(levh<top_height)
    
    # these are level, lat, lon cubes
    lat,lon = w.latitude.data, w.longitude.data
    
    if extent is None:
        extent=[lon[0],lon[-1],lat[0],lat[-1]]
    
    # topography surface
    topog_layer = topog_surface(lat,lon,levh,d_topog)
    
    
    for hi, DAtime in enumerate(DAtimes):
        # angle for view is 270 degrees - 120 * hour/24
        # or 270 - 120 * (hour + hi/n_hi)/(24)
        camera_eye = get_camera_eye(hour,hi,DAtimes,245)
        
        # surfaces to be plotted in 3d
        surface_list = [topog_layer]
        
        # heat
        theta_surf = theta_surface(lat,lon,levh, th[hi])
        surface_list.append(theta_surf)
        
        ## atmospheric heat (theta)
        up,down = up_down_drafts(lat,lon,levh, w[hi],topind=topind)
        surface_list.append(up)
        surface_list.append(down)
        
        ## title, lables
        title="%s (UTC+%.1f)"%(time_lt[hi].strftime("%dT%H:%M"),offset_lt)
        layoutargs = title_in_layout(title)

        figname = None
        if not send_to_browser:
            figname = fio.standard_fig_name(mr,_sn_,DAtime,subdir="vertmotion")
        create_figure(surface_list, filename=figname, camera_eye=camera_eye, **layoutargs)

def vorticity_system(mr, hour=2, 
                top_height=1200, 
                send_to_browser=False,
                extent=None, #constants.extents['KI']['zoom1'],
                #HSkip=5,
                #ltoffset=8
                **iso_surf_args,
                ):
    """
    Read an hour of model output, plot it in 3d using plotly
    saves output as .png
    ARGUMENTS:
        hour: which output hour (first is 0, last is 23 or -1)
        ws_min,ws_max: windspeed min and max vals
        height_top: how high (m) included in plot
        send_to_browser: instead of trying to save figures, send one to the browser (interactive)
    """
    
    #hours=fio.hours_available(mr)
    #dtime=hours[hour]
    
    DS = fio.read_model_run_hour(mr, 
                               hour=hour, 
                               extent=extent, 
                               )
    #print(DS)
    extras = fio.extra_data_read(mr,hour=hour,extent=extent,)
    #utils.extra_DataArrays(DS,add_winds=True,add_theta=True)
    
    # Get the rest of the desired data
    topog = DS['surface_altitude']
    d_topog = topog.load().data # lat,lon
    # set one pixel to -150 to fix color scale
    #d_topog[1,1] = -150
    
    w=DS['vertical_wnd']
    #th = DS['potential_temperature']
    th = extras['potential_temperature']
    
    # datetimes in hour output
    DAtimes=w.time.data
    lats=w.latitude.data
    lons=w.longitude.data
    #u,v = DS['u'],DS['v']
    u,v = extras['u'],extras['v']
    
    #print(type(u),type(u.data),type(lats))
    #print(u.shape,v.shape,len(lats),len(lons))
    
    # title from local time at fire ignition
    time_lt = utils.local_time_from_time_lats_lons(DAtimes,lats,lons)
    offset_lt = utils.local_time_offset_from_lats_lons(lats,lons)
    
    if "level_height" in w:
        levh  = w.level_height.load().data
    else:
        levh  = w.level_height_0.load().data
    
    topind = np.sum(levh<top_height)
    
    # these are level, lat, lon cubes
    lat,lon = w.latitude.data, w.longitude.data
    
    if extent is None:
        extent=[lon[0],lon[-1],lat[0],lat[-1]]
    
    # topography surface
    topog_layer = topog_surface(lat,lon,levh,d_topog)
    
    
    for hi, DAtime in enumerate(DAtimes):
        # angle for view is 270 degrees - 120 * hour/24
        # or 270 - 120 * (hour + hi/n_hi)/(24)
        camera_eye = get_camera_eye(hour,hi,DAtimes)
        
        vort_3d = np.zeros(th[hi].shape)
        OW = np.zeros(th[hi].shape)
        OWN = np.zeros(th[hi].shape)
        OWZ = np.zeros(th[hi].shape)
        for levi in range(topind):
            vort_3d[levi],OW[levi],OWN[levi],OWZ[levi] = utils.vorticity(u[hi,levi].data,v[hi,levi].data,lats,lons)
            
        print("DEBUG: zeta",np.nanmin(vort_3d),np.nanmax(vort_3d),np.nanmean(vort_3d))
        print("DEBUG: OW",np.nanmin(OW),np.nanmax(OW),np.nanmean(OW))
        print("DEBUG: OWN",np.nanmin(OWN),np.nanmax(OWN),np.nanmean(OWN))
        print("DEBUG: OWZ",np.nanmin(OWZ),np.nanmax(OWZ),np.nanmean(OWZ))
        
        # surfaces to be plotted in 3d
        surface_list = [topog_layer]
        
        # heat
        theta_surf = theta_surface(lat,lon,levh, th[hi])
        surface_list.append(theta_surf)
        
        ## atmospheric heat (theta)
        vort_surf1=isosurface_wrapper(lat,lon,levh, OWZ, topind=topind,
                                      default_surf_args=_vorticity_pos_surf_defaults_)
        vort_surf2=isosurface_wrapper(lat,lon,levh, OWZ, topind=topind,
                                      default_surf_args=_vorticity_neg_surf_defaults_)
        surface_list.append(vort_surf1)
        surface_list.append(vort_surf2)
        
        title="%s (UTC+%.1f)"%(time_lt[hi].strftime("%dT%H:%M"),offset_lt)
        layoutargs = title_in_layout(title)

        figname = None
        if not send_to_browser:
            #figname = cubetime.strftime('figures/threedee/test_%Y%m%d%H%M.png')
            figname = fio.standard_fig_name(mr,_sn_,DAtime,subdir="vorticity")
        create_figure(surface_list, filename=figname, camera_eye=camera_eye, **layoutargs)

# def cloud_system(mr='KI_run1_exploratory', hour=1, 
#                 theta_height=1000, theta_min=311, theta_max=320,
#                 vert_motion_height = 2500,
#                 top_height=8000, send_to_browser=False,
#                 extent=None,
#                 HSkip=5,
#                 ):
#     """
#     Read an hour of model output, plot it in 3d using plotly
#     saves output as .png
#     ARGUMENTS:
#         hour: which output hour (first is 0, last is 23 or -1)
#         theta_height: how high are we looking regarding potential temp?
#         theta_min, theta_max: min and max potential temperature to draw isosurface
#         vert_motion_height: altitude of vertical motion surface,
#         height_top: how high (m) included in plot
#         send_to_browser: instead of trying to save figures, send one to the browser (interactive)
#     """
    
#     #hours=fio.hours_available(mr)
#     #dtime=hours[hour]
    
#     DS = fio.read_model_run_hour(mr,
#                                hour=hour,
#                                extent=extent,
#                                )
    
#     #utils.extra_DataArrays(DS,add_theta=True,add_winds=True,add_z=True)
#     extras = fio.extra_data_read(mr,hour=hour,extent=extent)

#     qc = DS['cld_ice'][...,::HSkip,::HSkip]+DS['cld_water'][...,::HSkip,::HSkip]
#     #th = DS['potential_temperature']
#     th = extras['potential_temperature'][...,::HSkip,::HSkip]
    
#     # datetimes in hour output
#     DAtimes=th.time.data
#     lats=th.latitude.data
#     lons=th.longitude.data
#     # title from local time at fire ignition
#     time_lt = utils.local_time_from_time_lats_lons(DAtimes,lats,lons)
#     offset_lt = utils.local_time_offset_from_lats_lons(lats,lons)
    
#     #DS_ff = fio.read_model_run_fire(mr, extent=extent, )
#     #ff = (DS_ff['firefront'].loc[dict(time=DAtimes)])[...,::HSkip,::HSkip]
#     #sh = DS_ff['SHEAT_2'].loc[dict(time=DAtimes)]
    
#     # Get the rest of the desired data
#     topog = DS['surface_altitude']
#     d_topog = topog.load().data[...,::HSkip,::HSkip]
#     # set one pixel to -150 to fix color scale
#     d_topog[1,1] = -150
    
#     #u = extras['u'][...,::HSkip,::HSkip]
#     #v = extras['v'][...,::HSkip,::HSkip]
#     w = DS['vertical_wnd'][...,::HSkip,::HSkip]
#     if "level_height" in w:
#         levh  = w.level_height.load().data
#     else:
#         levh  = w.level_height_0.load().data
    
#     topind = np.sum(levh<top_height)
#     topind_th = np.sum(levh<theta_height)
    
#     # these are level, lat, lon cubes
#     lat,lon = qc.latitude.data, qc.longitude.data
    
#     if extent is None:
#         extent=[lon[0],lon[-1],lat[0],lat[-1]]
#     # dimensional mesh
    
#     # X,Y,Z = np.meshgrid(lon,lat,levh) 
#     # ## X Y Z are now [lat, lon, lev] for some reason
#     # [X,Y,Z] = [ np.moveaxis(arr,0,1) for arr in [X,Y,Z]]
#     # ## Now they are lon, lat, lev
#     # ## Cut down to desired level
#     # [X, Y, Z] = [ arr[:,:,:topind] for arr in [X,Y,Z]]
    
#     # topography surface
#     topog_layer = topog_surface(lat,lon,levh,d_topog)
    
#     namedlocs=[]
#     namedlocs_lats = []
#     namedlocs_lons = []
#     for (namedloc, (loclat, loclon)) in plotting._latlons_.items():
#         #print(namedloc, loclat, loclon)
#         if loclon < extent[1] and loclon > extent[0] and loclat < extent[3] and loclat > extent[2]:
#             if 'fire' not in namedloc and 'pyrocb' not in namedloc:
#                 namedlocs.append(namedloc)
#                 namedlocs_lats.append(loclat)
#                 namedlocs_lons.append(loclon)
    
    
#     for hi, DAtime in enumerate(DAtimes):
#         #verbose("Creating surfaces")
#         camera_eye = get_camera_eye(hour,hi,DAtimes)
        
#         # get cloud, theta, vert motion, firefront in terms of lon,lat,lev
#         #d_qc = cube_to_xyz(qc[hi],ztopind=topind)
#         #d_th = cube_to_xyz(th[hi],ztopind=topind_th)
#         #d_w = cube_to_xyz(w[hi],ztopind=topind)
        
#         # surfaces to be plotted in 3d
#         surface_list = [topog_layer]
        
#         ## Points for waroona, yarloop
#         # locations_scatter = go.Scatter3d(
#         #     x=namedlocs_lons,
#         #     y=namedlocs_lats,
#         #     z=[np.nanmin(levh)]*len(namedlocs),
#         #     mode='markers',
#         #     marker=dict(
#         #         size=4,
#         #         color='black',           # array/list of desired values
#         #         #colorscale='Viridis',   # choose a colorscale
#         #         opacity=0.8
#         #         ),
#         #     )
#         # surface_list.append(locations_scatter)
        
#         ## atmospheric heat (theta)
#         verbose("adding heat surface")
#         print("DEBUG: heat shape, topind",th.shape, topind_th)
#         print("DEBUG: heat[hi,:topind_th] 5ns",np.nanpercentile(th[hi,:topind_th],[0,25,50,75,100]))
#         #print("DEbuG: levh:",levh) # about 0-5km in badja_am1_exploratory
#         fire_front=theta_surface(lat,lon,levh,th[hi],)
#         surface_list.append(fire_front)
#         # theta_surf = isosurface_wrapper(lat,lon,levh, th[hi].load(), topind=topind_th,
#         #                                 #default_surf_args=_vorticity_neg_surf_defaults_)
#         #                                 isomin=theta_min,
#         #                                 isomax=theta_max,
#         #                                 surface_count=4,
#         #                                 opacity=0.5,
#         #                                 colorscale='Hot',
#         #                                 showscale=False,)
#         # # theta_surf = go.Isosurface(
#         # #     z=Z[:,:,:topind_th].flatten(),
#         # #     x=X[:,:,:topind_th].flatten(),
#         # #     y=Y[:,:,:topind_th].flatten(),
#         # #     value=d_th.flatten(),
#         # #     isomin=theta_min,
#         # #     isomax=theta_max,
#         # #     surface_count=4,
#         # #     opacity=0.7,
#         # #     colorscale='Hot',
#         # #     showscale=False
#         # #     )
#         # surface_list.append(theta_surf)
            
#         ## volume plot showing vertical motion 
#         vm_ind = np.sum(levh<vert_motion_height)
#         print("DEBUG: w shape, topind",w.shape, vm_ind)
#         print("DEBUG: w 5ns",np.nanpercentile(w[hi,:vm_ind],[0,25,50,75,100]))
#         up_surf,down_surf = up_down_drafts(lat,lon,levh,w[hi],topind=vm_ind)
#         surface_list.append(up_surf)
#         surface_list.append(down_surf)
#         ## Cloud isosurface
#         qcmin = 0.1
#         qcmax = 1
#         print("DEBUG: qc shape, topind", qc.shape, topind)
#         print("DEBUG: qc 5ns",np.nanpercentile(qc[hi,:topind],[0,25,50,75,100]))
#         # qc_volume = isosurface_wrapper(lat,lon,levh,qc[hi].load(),topind, method=go.Volume,
#         #     isomin=qcmin,
#         #     isomax=qcmax,
#         #     opacity=0.09, # max opacity
#         #     surface_count=10,
#         #     showscale=False, 
#         # )
#         # # qc_volume = go.Volume(
#         # #     x=X.flatten(), 
#         # #     y=Y.flatten(), 
#         # #     z=Z.flatten(),
#         # #     value=d_qc.flatten(),
#         # #     isomin=qcmin,
#         # #     isomax=qcmax,
#         # #     opacity=0.1, # max opacity
#         # #     surface_count=20,
#         # #     showscale=False, 
#         # # )
#         # surface_list.append(qc_volume)
        
#         ## title, lables
#         title="%s %s (UTC+%.1f)"%(mr, time_lt[hi].strftime("%dT%H:%M"),offset_lt)
#         layoutargs = title_in_layout(title)

#         figname = None
#         if not send_to_browser:
#             #figname = cubetime.strftime('figures/threedee/test_%Y%m%d%H%M.png')
#             figname = fio.standard_fig_name(mr,_sn_,DAtime,subdir="clouds")
#         create_figure(surface_list, filename=figname, camera_eye=camera_eye, **layoutargs)

def cloud_system(mr, hour=1, 
                #theta_height=1000, theta_min=311, theta_max=320,
                #vert_motion_height = 2500,
                top_height=10000, send_to_browser=False,
                extent=None,
                #HSkip=5,
                ):
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
    DS,extras,DAtimes,LT,levh,lat,lon,topind = setup_3d(mr,hour,top_height,extent)
    
    d_topog = DS['surface_altitude'].load().data # lat,lon
    # set one pixel to -150 to fix color scale
    #d_topog[1,1] = -150
    
    th = extras['potential_temperature']
    w=DS['vertical_wnd']

    # topography surface
    topog_layer = topog_surface(lat,lon,levh,d_topog)

    for hi, DAtime in enumerate(DAtimes):
        # angle for view is 270 degrees - 120 * hour/24
        # or 270 - 120 * (hour + hi/n_hi)/(24)
        camera_eye = get_camera_eye(hour,hi,DAtimes)
        
        # surfaces to be plotted in 3d
        surface_list = [topog_layer]
        
        # heat
        theta_surf = theta_surface(lat,lon,levh, th[hi])
        surface_list.append(theta_surf)
        
        # Lets add clouds:
        # ...
        
        # temporarily vert motion too
        up,down = up_down_drafts(lat,lon,levh, w[hi],topind=topind)
        surface_list.append(up)
        surface_list.append(down)
        
        ## title, lables
        title="%s (LT)"%(LT[hi].strftime("%dT%H:%M"))
        layoutargs = title_in_layout(title)

        figname = None
        if not send_to_browser:
            figname = fio.standard_fig_name(mr,_sn_,DAtime,subdir="clouds")
        create_figure(surface_list, filename=figname, camera_eye=camera_eye, **layoutargs)



def setup_3d(mr,hour,top_height,extent):
    DS = fio.read_model_run_hour(mr, 
                               hour=hour, 
                               extent=extent, 
                               )
    # need to read from extra_data folder, based on fully resolved model output
    extras = fio.extra_data_read(mr,hour=hour, force_recreate=False, extent=extent)
    #if extent is not None:
    #    extras = fio.extract_extent(extras,extent)
    #utils.extra_DataArrays(DS,add_theta=True)
    
    # Get the dims
    v1 = DS['cld_ice']
    
    # datetimes in hour output
    DAtimes=v1.time.data
    # these are level, lat, lon cubes
    lat,lon = v1.latitude.data, v1.longitude.data
    # title from local time at fire ignition
    time_lt = utils.local_time_from_time_lats_lons(DAtimes,lat,lon)
    #offset_lt = utils.local_time_offset_from_lats_lons(lat,lon)
    
    if "level_height" in v1:
        levh  = v1.level_height.load().data
    else:
        levh  = v1.level_height_0.load().data
    
    topind = np.sum(levh<top_height)
    
    #if extent is None:
    #    extent=[lon[0],lon[-1],lat[0],lat[-1]]
    
    return DS,extras,DAtimes, time_lt,levh,lat,lon,topind

def rotation_system(mr, hour=4, 
                top_height=5000, 
                send_to_browser=False,
                extent=None,
                pos_args={},
                neg_args={},
                ):
    """
    Read an hour of model output, plot it in 3d using plotly
    saves output as .png
    ARGUMENTS:
        mr: model run name (matched to data folder)
        hour: which output hour (first is 0, last is 23 or -1)
        top_height: in metres
        send_to_browser: instead of trying to save figures, send one to the browser (interactive)
        extent: [W,E,S,N]
        pos_args: isosurface arguments for positive rotation (shear/deformation)
        neg_args:  ' ' ' for negative rotation
    """
    
    DS,extras,DAtimes, LT,levh,lat,lon,topind=setup_3d(mr,hour,top_height,extent)
    
    d_topog = DS['surface_altitude'].load().data # lat,lon
    # set one pixel to -150 to fix color scale
    #d_topog[1,1] = -150
    
    th = extras['potential_temperature']
    rot = extras['rotation']
    # topography surface
    topog_layer = topog_surface(lat,lon,levh,d_topog)
    
    for hi, DAtime in enumerate(DAtimes):
        # angle for view is 270 degrees - 120 * hour/24
        # or 270 - 120 * (hour + hi/n_hi)/(24)
        camera_eye = get_camera_eye(hour,hi,DAtimes)
        
        # surfaces to be plotted in 3d
        surface_list = [topog_layer]
        
        # heat
        theta_surf = theta_surface(lat,lon,levh, th[hi])
        surface_list.append(theta_surf)
        
        # Rotation
        #print("DEBUG: rotation 5ns",np.nanpercentile(rot[hi],[0,25,50,75,100]))
        rot_strong_neg_surf = isosurface_wrapper(lat,lon,levh,rot[hi],topind,
                                 default_surf_args=_rotation_strong_neg_surf_defaults_,**neg_args)
        rot_strong_pos_surf = isosurface_wrapper(lat,lon,levh,rot[hi],topind,
                                 default_surf_args=_rotation_strong_pos_surf_defaults_,**pos_args)
        surface_list.append(rot_strong_pos_surf)
        surface_list.append(rot_strong_neg_surf)

        rot_weak_neg_surf = isosurface_wrapper(lat,lon,levh,rot[hi],topind,
                                 default_surf_args=_rotation_weak_neg_surf_defaults_,**neg_args)
        rot_weak_pos_surf = isosurface_wrapper(lat,lon,levh,rot[hi],topind,
                                 default_surf_args=_rotation_weak_pos_surf_defaults_,**pos_args)
        surface_list.append(rot_weak_pos_surf)
        surface_list.append(rot_weak_neg_surf)

        ## title, lables
        title="%s (LT)"%(LT[hi].strftime("%dT%H:%M"))
        layoutargs = title_in_layout(title)

        figname = None
        if not send_to_browser:
            figname = fio.standard_fig_name(mr,_sn_,DAtime,subdir="rotation")
        create_figure(surface_list, filename=figname, camera_eye=camera_eye, **layoutargs)

if __name__=='__main__':
    KI_extents=constants.extents["KI"]
    KI_extent=KI_extents['zoom1']
    badja_extents=constants.extents["badja"]
    #badja_extent=badja_extents['zoom1']
    badja_extent=badja_extents['bigger']

    #mr="badja_run3_exploratory"
    mr = "badja_am1_exploratory"
    extent=badja_extent
    #mr = "KI_run2_exploratory"
    #extent = KI_extent
    
    # 3d plots
    if True:
        for hour in range(3,5):#[2,3,4,5,6,7,8,9,10,11,12]: 
            for funk in [
                #vorticity_system,
                #vertmotion_system,
                #wind_system,
                rotation_system,
                ]:
                    funk(
                        mr=mr, 
                        hour=hour, 
                        extent=extent,
                        send_to_browser=False,
                        top_height=6500,
                    )
    
    # Cloud images
    if False:
        for hour in range(4,5):
            cloud_system(mr=mr,
                         hour = hour,
                         extent = extent,
                         top_height = 10000,
                         send_to_browser=False,)
    
            