#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 08:45:14 2019
    Weather summary looking at winds and clouds
@author: jesse
"""
import matplotlib
matplotlib.use("Agg")

from matplotlib import colors, ticker, patches
from matplotlib import patheffects, gridspec
import matplotlib.dates as mdates

import numpy as np
import matplotlib.pyplot as plt
import iris # minimal fio done here 
import cartopy.crs as ccrs
import warnings
from datetime import datetime, timedelta

from utilities import utils, plotting, fio, constants

## GLOBAL
#Script name
_sn_ = 'weather_summary'
__cloud_thresh__ = constants.cloud_threshold

def plot_weather_summary(U,V,W, height, lat, lon, 
                         extentname=None, 
                         Q=None, FF=None, 
                         topog=None, topog_contours=[4],
                         hwind_limits=None):
    '''
    Show horizontal slices of horizontal and vertical winds averaged between 
    several vertical levels. Also shows clouds (Q) and fires (FF) with contour outline.
    
    INPUTS: 
        U,V,W: wind speed in lon, lat, vert dimension m/s [z,lat,lon]
        height: altitude ASL or AGL m [z,lat,lon] or [z]
        height: level heights for titles (array [ levs ])
        topog: surface altitudes [lat,lon] in metres
        topog_contours: add a contour along these surface altitudes
        hwind_limits: None, or [vmin, vmax]
            horizontal wind map contourf colorbar limits
    '''
    ## Just make sure we are using regular grid
    if not np.all(np.equal(np.diff(lat),lat[1]-lat[0])):
        print("WARNING: lats adjusted in plot_weather_summary")
        lat = np.linspace(lat[0],lat[-1],len(lat))
    if not np.all(np.equal(np.diff(lon),lon[1]-lon[0])):
        print("WARNING: lons adjusted in plot_weather_summary")
        lon = np.linspace(lon[0],lon[-1],len(lon))

    extent = [lon[0],lon[-1],lat[0],lat[-1]]

    ## Use height array (level height averages) to subset data into plots
    # first row is for levels between 100 and 500 metres altitude
    row1 = (100<=height) * (height<500)
    # etc...
    row2 = (500<=height) * (height<1500)
    row3 = (1500<=height) * (height<3000)
    row4 = (3000<=height) * (height<5000)
    # todo row 5
    row5 = (5000<height) * (height<10000)

    # vertical wind colourbar is constant
    wcmap=plotting._cmaps_['verticalvelocity']
    wnorm=colors.SymLogNorm(0.25,base=2.) # linear to +- 0.25, then log scale
    # specific contours for vwind map
    wcontours=np.union1d(np.union1d(2.0**np.arange(-2,6),-1*(2.0**np.arange(-2,6))),np.array([0]))
    # number of contours in hwind map
    nhcontours=16
    hcontours=nhcontours
    hcmap='plasma' #plotting._cmaps_['windspeed']
    speedmax=None
    if hwind_limits is not None:
        hvmin,speedmax = hwind_limits
        hcontours=np.linspace(hvmin,speedmax,nhcontours)
        hnorm=colors.Normalize(vmin=hwind_limits[0],vmax=hwind_limits[1], clip=True)

    fig = plt.figure(figsize=[10,10])
    # for each vertical subset we will look at (one per row)
    for ii,row in enumerate([row1, row2, row3, row4, row5]):
        lax = plt.subplot(5,2,ii*2+1, )
        
        Ui = U[row]
        Vi = V[row]
        Si = np.hypot(Ui,Vi)
        
        # mean of levels (.data may be masked array)
        Ur = np.mean(Ui, axis=0)
        Vr = np.mean(Vi, axis=0)
        Sr = np.mean(Si, axis=0)
        
        # wind speed contourf
        if hwind_limits is None:
            csh = plt.contourf(lon, lat, Sr, hcontours)
            # if we're not using a uniform hwind scale, add colorbars
            plt.colorbar(ticklocation=ticker.MaxNLocator(5),pad=0)
        else:
            csh = lax.contourf(lon, lat, Sr, 
                               levels=hcontours,
                               cmap=hcmap, norm=hnorm)
        
        # Add fire contour if availalbe
        if FF is not None:
            plotting.map_fire(FF, lat, lon)
        # can do underlay of topography contour(s)
        if topog is not None:
            #print("DEBUG:", lon.shape, lat.shape, topog.shape,topog_contours)
            #print("     : lons", lon[:7])
            #print("     : lats", lat[:7])
            #print("     :", type(topog))
            
            lax.contour(lon,lat,topog,topog_contours,
                        colors='k', alpha=0.7, linewidths=1)
        
        plotting.map_add_locations_extent(extent, hide_text=True)
        
        ##Streamplot the horizontal winds
        ## This shifts the subplot grid axes slightly!! 
        # this is seemingly to show the arrows in full where they extend outside the boundary
        streamLW=utils.wind_speed_to_linewidth(Sr,speedmax=speedmax)
        lax.streamplot(lon,lat,Ur,Vr, 
                       color='k',
                       linewidth=streamLW,
                       minlength=0.5,
                       density=(0.5, 0.4))
        # set limits back to latlon limits
        lax.set_ylim(lat[0],lat[-1])  # outliers only
        lax.set_xlim(lon[0],lon[-1])
        
        # remove x and y ticks
        plt.xticks([],[])
        plt.yticks([],[])
        
        # in first row, add title
        if ii==0:
            lax.set_title('horizontal winds (m/s)')
        
        # add ylabel
        plt.ylabel("%.0f - %.0f m"%(height[row][0], height[row][-1]),fontsize=13)
        #lax.yaxis.set_label_position("right")
        
        ## Now plot vertical motion
        plt.subplot(5,2,ii*2+2,)
        # take mean over vertical space
        Wi = W[row]
        Wr = np.mean(Wi,axis=0)
        
        cs = plt.contourf(lon, lat, Wr, wcontours, cmap=wcmap, norm=wnorm)
        plotting.map_add_locations_extent(extent, hide_text=ii>0, nice=False,)
        
        # add cloud hatching
        if Q is not None:
            Qmax = np.max(Q[row], axis=0)
            # add hatches over where Qmax is greater than cloud threshhold
            plt.contourf(lon,lat,Qmax, [0,__cloud_thresh__,100], 
                         colors=['None']*3, hatches=[None,'/','//'],
                         extend='both',)
        
        if FF is not None:
            plotting.map_fire(FF, lat, lon)
        
        if topog is not None:
            plt.contour(lon,lat,topog,topog_contours,
                        colors='k', alpha=0.7, linewidths=1)
            
        plt.xticks([],[])
        plt.yticks([],[])
        if ii==0:
            plt.title('vertical motion (m/s)')
        
        
    # reduce vert gap between subplots
    fig.subplots_adjust(hspace=0.1)
    # add space in specific area, then add vert wind colourbar
    cbar_ax = fig.add_axes([0.905, 0.4, 0.01, 0.2]) # X Y Width Height
    fig.colorbar(cs, cax=cbar_ax, format=ticker.ScalarFormatter(), pad=0)
    # Add horizontal wind colourbar (if uniform)
    if hwind_limits is not None:
        cbar_ax2 = fig.add_axes([0.48, 0.4, 0.01, 0.2]) #XYWH
        fig.colorbar(csh, cax=cbar_ax2, format=ticker.ScalarFormatter(),
                pad=0, extend='max')
        

def weather_summary_model(mr,
                          fdtimes=None,
                          zoom_in=None,
                          subdir=None,
                          HSkip=None,
                          hwind_limits=None):
    '''
    Read model run output hour by hour, running plot_weather_summary on each
    time slice. Can subset to just show fdtimes, and can also zoom to specific 
    lat lon box, which saves into the 'zoomed' subfolder
    '''
    # font sizes etc
    plotting.init_plots()
    
    extentname = mr.split('_')[0]
    extent = constants.extents[extentname]
    if zoom_in is not None:
        extentname=None
        extent = zoom_in
    if fdtimes is None:
        fdtimes = fio.run_info[mr]['filedates']
    FF = None
    
    # read one hour at a time, plot each available time slice
    for fdtime in fdtimes:
        
        # read cubes
        cubes = fio.read_model_run(mr, hours=[fdtime], extent=extent, 
                                   HSkip=HSkip)
        utils.extra_cubes(cubes,add_winds=True) # add u,v,s,wind_direction
        topog = fio.read_topog(mr,
                               extent=extent,
                               HSkip=HSkip)
        u,v= cubes.extract(['u','v'])
        w = cubes.extract('upward_air_velocity')[0]
        qc, = cubes.extract('qc')
        lat = w.coord('latitude').points
        lon = w.coord('longitude').points
        height = utils.height_from_iris(w)
        dtimes = utils.dates_from_iris(u)
        # read fire front
        ff, = fio.read_fire(mr, dtimes, extent=extent, HSkip=HSkip)
        
        # for each time slice create a weather summary plot
        for i,dtime in enumerate(dtimes):
            
            ui, vi = u[i].data, v[i].data
            wi = w[i].data
            qci = qc[i].data
            if ff is not None:
                FF = ff[i].data
            
            plot_weather_summary(ui, vi, wi, height, lat, lon, 
                                 extentname=extentname,
                                 Q = qci, FF=FF,
                                 hwind_limits=hwind_limits,
                                 topog=topog.data,
                                 )
            
            offsethours = fio.run_info[mr]['UTC_offset']
            ltime=dtime+timedelta(hours=offsethours)
            plt.suptitle("%s weather "%mr + ltime.strftime("%Y %b %d %H:%M (LT)"))
            
            if (zoom_in is not None) and (subdir is None): 
                subdir = 'zoomed'
            fio.save_fig(model_run=mr,
                         plot_name="weather_summary", 
                         plot_time=dtime, 
                         plt=plt,
                         extent_name=extentname,)

def weather_series(model_run='waroona_run3', 
                   extent=None,
                   HSkip=None,
                   day2=False,
                   showFP=False,
                   showPFT=False,
                   showwsmax=False,
                   showmap=False,
                   mapname=None,
                   showfirelines=False,
                   showfirelinehour=None,
                   showQC=False,
                   localtime=True,
                   test=False,
                   ):
    """
        time series of surface weather
    """
    
    extentname=model_run.split('_')[0]
    # zoomed extent for analysis
    extentnamez = extentname + 'z'
    if extent is None:
        extent=constants.extents[extentnamez]
    
    qc_thresh = constants.cloud_threshold
    
    # READ EVERYTHING, SUBSET, CALC DESIRED METRICS, PLOT METRICS
    modeltimes = fio.run_info[model_run]['filedates']
    fdtimes = modeltimes[-24:] if day2 else modeltimes[:24]
    if test:
        fdtimes=[modeltimes[0],] # Testing just look at first hour
    cubes = fio.read_model_run(model_run, extent=extent,
                               HSkip=HSkip,
                               hours=fdtimes)
    ctimes = utils.dates_from_iris(cubes[0])
    ## read all the fire data, and lats/lons
    ## 10m u and v winds (already destaggered) are in fire model output
    ff,sh,u10, v10 = fio.read_fire(model_run=model_run, dtimes=ctimes,
                           extent=extent, HSkip=HSkip,
                           firefront=True, sensibleheat=True,
                           day2=day2, day1=(not day2),
                           wind=True)
    
    lons,lats = sh.coord('longitude').points, sh.coord('latitude').points
    assert sh is not None, "Missing sensible heat file"
    # datetimes list, convert to local time if desired
    ftimes = utils.dates_from_iris(sh)
    ftimes_lt = ftimes
    ctimes_lt = ctimes
    offset_hrs = fio.run_info[model_run]['UTC_offset']
    if localtime:
        offset = timedelta(hours=offset_hrs)
        ftimes_lt = np.array([ft + offset for ft in ftimes ])
        ctimes_lt = np.array([ct + offset for ct in ctimes ])
    
    # spatial sum over time of sensible heat gives firepower
    # gives GWatts
    firepower = np.sum(utils.firepower_from_cube(sh), axis=(1,2))
    
    # remove zeros:
    prefire = np.isclose(np.cumsum(firepower), 0)
    firepower[prefire] = np.NaN
    
            
    ## Read PFT
    # calculated using kevin's code, stored as GWatts
    lat,lon = plotting._latlons_["fire_%s_upwind"%extentname]
    if showPFT:
        pft, ptimes, _, _ = fio.read_pft(model_run,lats=lat,lons=lon)
        ptimes_lt = ptimes
        #pft = np.nanmean(pft,axis=(1,2)) # average spatially
        if localtime:
            ptimes_lt = np.array([pt + offset for pt in ptimes ])
    
    ## Read Temperature:
    # just want surface for most of these, lets just take bottom two rows:
    # surf_constraint = iris.Constraint(model_level_number = lambda cell: cell<=2)
    #cubes=fio.read_model_run(model_run, extent=extent, constraints=surf_constraint, 
    #                         add_RH=True)
    
    
    ## get temperature, RH, cloud
    qTqc = cubes.extract(['specific_humidity','air_temperature','qc'])
    print("DEBUG: cubes")
    print(qTqc)
    q,T,qc = qTqc

    index_500m = np.argmin(np.abs(utils.height_from_iris(q)-500))
    assert index_500m > 0, "Index500 didn't work = %d"%index_500m
    clats, clons = q.coord('latitude').points, q.coord('longitude').points
    # compute RH from specific and T in kelvin
    T.convert_units('K')
    # just want surface for Temp and RH
    q = q[:,0,:,:]
    T = T[:,0,:,:].data
    # qc stuff takes heaps of RAM
    #qc_weight[np.isnan(qc_weight)] = 0.0 # return the zeros
    RH = utils.relative_humidity_from_specific(q.data, T)
    RH = np.mean(RH, axis=(1,2))
    T = np.mean(T, axis=(1,2))
    ## get wind speed/ wind dir
    u, v = cubes.extract(['x_wind','y_wind'])
    ## wind speed at 10m is output by the fire model
    u10=u10.data #np.swapaxes(u10.data, 1,2) # time, lon, lat -> time, lat, lon
    v10=v10.data #np.swapaxes(v10.data, 1,2) # also for v10
    u500=u[:,index_500m,:,:]
    v500=v[:,index_500m,:,:]
    
    # DESTAGGER u and v using iris interpolate
    u500 = u500.interpolate([('longitude',v.coord('longitude').points)],
                        iris.analysis.Linear())
    v500 = v500.interpolate([('latitude',u.coord('latitude').points)],
                        iris.analysis.Linear())
    ws10_all = utils.wind_speed_from_uv(u10,v10)
    ws10_q1 = np.quantile(ws10_all, 0.25, axis=(1,2))
    ws10_q3 = np.quantile(ws10_all, 0.75, axis=(1,2))
    ws10_max = np.max(ws10_all, axis=(1,2))
    ws10 = np.mean(ws10_all, axis=(1,2))
    ws500_all = utils.wind_speed_from_uv_cubes(u500,v500).data
    ws500_q1 = np.quantile(ws500_all, 0.25, axis=(1,2))
    ws500_q3 = np.quantile(ws500_all, 0.75, axis=(1,2))
    #ws500_max = np.max(ws500_all, axis=(1,2))
    ws500 = np.mean(ws500_all, axis=(1,2))
    # mean wind direction based on mean u,v
    u10_mean = np.mean(u10,axis=(1,2))
    v10_mean = np.mean(v10,axis=(1,2))
    u500_mean = np.mean(u500.data,axis=(1,2)) 
    v500_mean = np.mean(v500.data,axis=(1,2))
    #wd0 = utils.wind_dir_from_uv(u10_mean,v10_mean)
    
    #######################
    #### PLOTTING PART ####
    #######################
    
    # figure and multiple y axes for time series
    fig = plt.figure(figsize=[12,10])
    dx,dy = extent[1]-extent[0], extent[3]-extent[2]
    extentplus = np.array(extent)+np.array([-0.3*dx,0.3*dx,-0.2*dy,0.2*dy])
    if showmap:
        # map with extent shown
        if mapname is None:
            mapname = "%s.tiff"%extentname
        fig, ax_map = plotting.map_tiff_qgis(fname=mapname, 
                                             fig=fig,
                                             extent=list(extentplus), 
                                             subplot_row_col_n = [3,1,1],
                                             #locnames=[extentname,]
                                             )
        if day2:
            plotting.map_add_locations_extent('waroona_day2',
                                              hide_text=False,
                                              nice=True)
        else:
            plotting.map_add_locations_extent(extentnamez,
                                              hide_text=False,
                                              nice=True)
        # Add rectangle
        botleft = extent[0],extent[2]
        ax_map.add_patch(patches.Rectangle(xy=botleft,
                                           width=dx,
                                           height=dy,
                                           #facecolor=None,
                                           fill=False,
                                           edgecolor='blue',
                                           linewidth=2,
                                           #linestyle='-',
                                           alpha=0.6, 
                                           ))
        
        # add fire outline
        plt.sca(ax_map)
        # Firefront shown at each hour increment:
        if not test and showfirelines:
            for ffhour in fdtimes:
                ffhind = utils.date_index(ffhour, ctimes)
                if ffhind is None:
                    ffhind = 0
                    ffhour = ctimes[0]
                ffline = plotting.map_fire(ff[ffhind].data, lats, lons,alpha=0.4)
                #if ffline is not None:
                #    clbls = plt.clabel(ffline, [0], fmt=ffhour.strftime('%H'), 
                #                       inline=True, colors='wheat')
                #    # padding so label is readable
                #    plt.setp(clbls, path_effects=[patheffects.withStroke(linewidth=3, foreground="k")])
        if showfirelinehour is not None:
            ffsample,= fio.read_fire(model_run=model_run, dtimes=[showfirelinehour],
                           extent=list(extentplus), HSkip=HSkip,
                           firefront=True, 
                           day2=day2, day1=(not day2),)
            ffsample_lats = ffsample.coord('latitude').points
            ffsample_lons = ffsample.coord('longitude').points
            print("DEBUG: ffsample info:")
            print(ffsample)
            ffline = plotting.map_fire(ffsample[0].data, ffsample_lats,ffsample_lons, alpha=0.4)

        ax_map.set_title("Surface area-average over time")
    
    ax_fp = plt.subplot(2+showmap,1,1+showmap) # firepower axis
    color_fp = "orange"
    ax_T = ax_fp.twinx() # Temperature
    color_T = "red"
    if (not showFP) and (not showPFT):
        ax_RH = ax_fp # just use left axis if no firepower
    else:
        ax_RH = ax_fp.twinx() # Rel Humid
        # offsets for extra y axes, and make just the one spine visible
        ax_RH.spines["right"].set_position(("axes", 1.07))
        plotting.make_patch_spines_invisible(ax_RH)
        ax_RH.spines["right"].set_visible(True)
        ax_fp.set_ylabel("GWatts")
        ax_fp.yaxis.label.set_color(color_fp)
    color_RH = "magenta"
    ax_ws = plt.subplot(2+showmap,1,2+showmap, sharex=ax_fp) # Wind speed
    color_ws = 'black'
    color_qc = "darkblue"
    color_wd = "brown"
    # wind speed at 500m on same axis as at 10m
    #ax_ws500 = ax_ws.twinx() # wind speed at 500m
    color_ws500 = 'chocolate'
    
    
    ## Plot firepower, and PFT on one axis
    plt.sca(ax_fp)
    if showPFT:
        line_pft, = plt.plot_date(ptimes_lt, pft, '--', 
                                color=color_fp, label='PFT', alpha=0.6)
    if showFP:
        line_fp, = plt.plot_date(ftimes_lt, firepower, '-',
                                 color=color_fp, label='firepower')
    else:
        ignition=ftimes_lt[~prefire][0]
        plt.annotate("ignition",(mdates.date2num(ignition), 0), xytext=(-60, -5), 
                     textcoords='offset points', arrowprops=dict(arrowstyle='-|>'), 
                     color='red')
    ## plot temperature on right axis
    plt.sca(ax_T)
    line_T, = plt.plot_date(ctimes_lt, T-273.15, '-',
                           color=color_T, label="Temperature",)
    ## plot RH on further right axis
    plt.sca(ax_RH)
    line_RH, = plt.plot_date(ctimes_lt, RH, '-',
                             color=color_RH, label="RH",)

    if showQC:
        qc_sum = qc.collapsed('model_level_number', iris.analysis.SUM)
        qc_frac = np.sum(qc_sum.data > qc_thresh, axis=(1,2))/(len(clats)*len(clons))
        qc_weight = qc_sum.collapsed(['longitude','latitude'], iris.analysis.SUM).data
        #qc_weight[qc_weight <= 0.0001] = np.NaN # take out zeros
        qc_q3 = np.nanpercentile(qc_weight, 75)
        qc_q2 = np.nanpercentile(qc_weight, 50)
        qc_heavy = qc_weight > qc_q3
        qc_mid = (qc_weight > qc_q2) * (qc_weight < qc_q3)
        plt.plot_date(ctimes_lt, qc_frac, 'o',
                      color=color_qc, 
                      fillstyle='none', 
                      mec=color_qc,
                      mew=1,)
        line_qc, = plt.plot_date(ctimes_lt[qc_heavy], qc_frac[qc_heavy], 'o',
                                 color=color_qc,
                                 label="Clouds",
                                 fillstyle='full')
        plt.plot_date(ctimes_lt[qc_mid], qc_frac[qc_mid], 'o',
                      color=color_qc,
                      fillstyle='bottom')
    
    ## Wind speed, wind speed stdev, wind dir quiver
    plt.sca(ax_ws)
    line_ws, = plt.plot_date(ctimes_lt, ws10, 'o',
                             color=color_ws, label="10m wind speed (+IQR)",)
    if showwsmax:
        line_wsmax, = plt.plot_date(ctimes_lt, ws10_max, '^', 
                                    color=color_ws, label="10m max wind speed")
    
    # add quartiles
    plt.fill_between(ctimes_lt, ws10_q3, ws10_q1, alpha=0.6, color='grey')
    if not test:
        # normalize windspeed for unit length quivers
        wdnorm = np.sqrt(u10_mean**2 + v10_mean**2)
        # dont show quiver at every single point
        qskip = len(wdnorm)//24 # just show 24 arrows
        plt.quiver(ctimes_lt[::qskip], ws10_q1[::qskip], 
                   u10_mean[::qskip]/wdnorm[::qskip], 
                   v10_mean[::qskip]/wdnorm[::qskip], 
                   pivot='mid', 
                   alpha=.9, 
                   color=color_ws,
                   headwidth=3, headlength=2, headaxislength=2,
                   width=.004)
    ## Same again but for 500m wind
    #plt.sca(ax_ws500)
    line_ws500, = plt.plot_date(ctimes_lt, ws500, '.',
                                color=color_ws500, 
                                label="500m wind speed (+IQR)",)
    # add quartiles
    plt.fill_between(ctimes_lt, ws500_q3, ws500_q1, alpha=0.4, color=color_ws500)
    
    # normalize windspeed for unit length quivers
    wdnorm500 = np.sqrt(u500_mean**2 + v500_mean**2)
    if not test:
            
        plt.quiver(ctimes_lt[::qskip], ws500_q3[::qskip], 
                   u500_mean[::qskip]/wdnorm500[::qskip], 
                   v500_mean[::qskip]/wdnorm500[::qskip], 
                   pivot='mid', 
                   alpha=.7, 
                   color=color_wd,
                   headwidth=3, headlength=2, headaxislength=2,
                   width=.003)
    # top row:
    
    ax_T.set_ylabel("Temperature (C)")
    ax_T.yaxis.label.set_color(color_T)
    ax_RH.set_ylabel("RH and cloud (frac)")
    ax_RH.yaxis.label.set_color(color_RH)
    
    # bottom row:
    ax_ws.set_ylabel("wind speed (m/s)")
    ax_ws.yaxis.label.set_color(color_ws)
    ax_ws.set_xlabel("UTC + %d"%offset_hrs)
    #ax_ws500.set_ylabel("500m wind speed (m/s)")
    #ax_ws500.yaxis.label.set_color(color_ws500)
    
    ## Plot periphery
    lines = [line_T, line_RH,]
    if showQC:
        lines.append(line_qc)
    if showFP:
        lines.append(line_fp)
    if showPFT:
        lines.append(line_pft)
    labels = [l.get_label() for l in lines]
    print("PRE LLEGEND")
    plt.savefig("test1.png")
    ax_T.legend(lines, labels, loc='center left')
    #plt.suptitle(model_run,y=0.9)
    
    if showwsmax:
        lines2 = [line_ws, line_wsmax, line_ws500]
    else:
        lines2 = [line_ws, line_ws500]
    labels2 = [l.get_label() for l in lines2]
    ax_ws.legend(lines2,labels2, loc='best')
    
    # format x-ticks date times
    ax_fp.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax_fp.xaxis.set_minor_locator(mdates.HourLocator())
    ax_fp.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    
    fig.tight_layout(rect=[0,0,.99,1]) # left, bottom, right, top
    pname = 'timeseries_day2.png' if day2 else 'timeseries.png'
    fio.save_fig(model_run, _sn_, pname, plt=plt)

if __name__=='__main__':
    

    mr='KI_run2'
    hours = fio.hours_available(mr)
    hwind_minmax = [0, 30]
    weather_summary_model(mr,HSkip=None,fdtimes=hours, hwind_limits=hwind_minmax)
    
    print("INFO: weather_summary.py done")

