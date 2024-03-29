#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 14:06:49 2019

  In this script is repeatable generic plotting stuff like adding a scale to a cartopy map

@author: jesse
"""

# Plotting stuff
import numpy as np
import matplotlib
from matplotlib import patheffects, patches
from matplotlib.lines import Line2D # for custom legends
import matplotlib.colors as col
import matplotlib.pyplot as plt
import matplotlib.ticker as tick

# interpolation
from scipy.interpolate import interp2d

# Mapping stuff
import cartopy
import cartopy.crs as ccrs
from cartopy.io import shapereader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.io.img_tiles as cimgt
from owslib.wmts import WebMapTileService
import warnings
# read tiff file
from osgeo import gdal

# local stuff
from utilities import utils, constants

###
### GLOBALS
###
__DATADIR__=constants.__DATADIR__
## Standard colour maps for use between figures
_cmaps_ = {}
#_cmaps_['verticalvelocity'] = 'PiYG_r'  # jeff likes this for vert velocity
_cmaps_['verticalvelocity'] = 'seismic'  # I like blue to red
_cmaps_['windspeed']        = 'YlGnBu' # jeff likes this for wind speeds
_cmaps_['qc']               = 'BuPu' # white through to purple for water+ice content in air
_cmaps_['topog']            = 'terrain'
_cmaps_['th']               = 'plasma' # potential temperature

## useful locations and extents
# Extents: EWSN
# latlons: [lat,lon] 
_extents_               = constants.extents
_latlons_               = constants.latlons 


def init_plots():
    matplotlib.rcParams['font.size'] = 13.0
    matplotlib.rcParams["text.usetex"]      = False     # I forget what this is for, maybe allows latex labels?
    matplotlib.rcParams["legend.numpoints"] = 1         # one point for marker legends
    matplotlib.rcParams["figure.figsize"]   = (9, 7)    # Default figure size
    matplotlib.rcParams["axes.titlesize"]   = 17        # title font size
    matplotlib.rcParams["figure.titlesize"] = 20        # figure suptitle size
    matplotlib.rcParams["axes.labelsize"]   = 14        #
    matplotlib.rcParams["xtick.labelsize"]  = 12        #
    matplotlib.rcParams["ytick.labelsize"]  = 12        #
    matplotlib.rcParams['image.cmap'] = 'plasma'        # Colormap default
    matplotlib.rcParams['axes.formatter.useoffset'] = False # another one I've forgotten the purpose of
    # rcParams["figure.dpi"] 
    #matplotlib.rcParams["figure.dpi"] = 200           # DEFAULT DPI for plot output

def topography_colouring(vmin,vmax,sealevel=0.05, nsea=50, nland=200):
    """
    based on input topography, get colormap levels and norm
    
    RETURN colormap, colorlevels, colornorm
        
    """
    
    colors_undersea = plt.cm.terrain(np.linspace(0,0.17,nsea))
    colors_land = plt.cm.terrain(np.linspace(0.25, 1, nland))
    # combine them and build a new colormap
    colors_terr = np.vstack((colors_undersea, colors_land))
    cut_terrain_map = matplotlib.colors.LinearSegmentedColormap.from_list('cut_terrain', colors_terr)
    
    if vmin < sealevel:
        print("TODO:")
    
    norm = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
    levels=np.linspace(vmin, vmax, num=nland+nsea+1)
    return cut_terrain_map,levels,norm

def cmap_with_white_range(cmap, ncolour=50, nwhite=50, x=0.5):
    """
    ncolour levels, nwhite levels, ncolor levels
    0 to x will be lower colours
    x to 1 will be upper levels
    """
    lower = cmap(np.linspace(0, x, ncolour))
    white = cmap(np.ones(nwhite)*x)
    upper = cmap(np.linspace(1-x, 1, ncolour))
    splitcolors = np.vstack((lower, white, upper))
    tmap = col.LinearSegmentedColormap.from_list('map_split_white', splitcolors)
    return tmap

def wind_dir_color_ring():
    """
    return struct with 'cmap' and 'norm' for building color ring
    """
    dircolorlist=['cyan','darkturquoise',
              'darkblue','mediumblue','blue',
              'limegreen','lime','green',
              'hotpink','deeppink','mediumvioletred',
              'lightskyblue','cyan']
    dircolorbounds=[0,15,45,75,105,135,165,195,225,255,285,315,345,360]
    dirticks=[0,45,90,135,180,225,270,315]

    cmap = col.ListedColormap(dircolorlist)
    norm=col.BoundaryNorm(dircolorbounds, cmap.N)

    n=200 # secants for mesh

    t = np.linspace(0,2*np.pi,n) # angles to show
    r = np.linspace(0.2,.9,2) # radius

    rg, tg = np.meshgrid(r,t)
    # Convert degrees math to degrees clockwise from north
    deg_to_met=(-1*tg.T*180/np.pi+90)%360
    
    retstruct={
        'cmap':cmap,
        'norm':norm,
        't':t,
        'r':r,
        'rg':rg,
        'tg':tg,
        'deg_to_met':deg_to_met,
        'dircolorlist':dircolorlist,
        'dircolorbounds':dircolorbounds,
        'dirticks':dirticks,
        }
    return retstruct

def add_wind_dir_color_ring(fig, color_ring_struct,
                            XYwh=[.7,.15,.08,.08],
                            deglable=False,
                            ):
    """
    ARGS:
        fig: figure to add ring colorbar to (uses add_axes)
        XYwh: [X,Y,width,height] for color ring axes
        color_ring_struct: structure containing:
            'cmap','norm',... from wind_dir_color_ring method
    RETURNS:
        ring_img, ring_ax for if you want to edit the colorbar somehow
    """
    ring_ax=fig.add_axes(XYwh,projection="polar")
    # define colormap normalization for 0 to 2*pi
    t=color_ring_struct['t']
    r=color_ring_struct['r']
    cmap=color_ring_struct['cmap']
    norm=color_ring_struct['norm']
    values=color_ring_struct['deg_to_met']
    
    ring_img = ring_ax.pcolormesh(t,r,
                                  values,
                                  cmap=cmap,
                                  norm=norm,
                                  shading="nearest",
                                  )
    ring_ax.set_yticklabels([])
    ring_ax.set_xticklabels([])
    if deglable:
        # convert 0 to 360 math direction ticks to met direction ticks
        math_ticks=np.deg2rad([0,90,180,270])
        met_ticks=np.array([90,0,270,180])
        ring_ax.set_xticks(math_ticks)
        ring_ax.set_xticklabels(met_ticks)
    return ring_img, ring_ax

def add_legend(ax,colours,labels,styles=None,markers=None, **lineargs):
    """
    create custom legend using a list of colours as input
    ARGS:
        ax: matplotlib axis to add legend to
        colours: ['red','green','cyan',...] whatever
        labels: ['','','',...]
        styles: ['--','-','.',...] whatever or None (assumes all normal lines)
        further arguments sent to Line2D function
    """
    custom_lines=[]
    for i,colour in enumerate(colours):
        if styles is not None:
            lineargs['linestyle'] = styles[i]
        if markers is not None:
            lineargs['marker'] = markers[i]
        custom_lines.append(Line2D([0], [0], color=colour, **lineargs))
    ax.legend(custom_lines, labels)


def annotate_max_winds(winds, upto=None, **annotateargs):
    """
    annotate max wind speed within 2d winds field
    ARGUMENTS:
        winds [y, x] = wind speed in m/s
        y, x = data axes,
        upto: if set, only look at y<=upto
        extra arguments for plt.annotate
            defaults: textcoords='axes fraction',
                      xytext=(0,-0.2), # note xy will be pointing at max windspeed location
                      fontsize=15,
                      s="windspeed max = %5.1f m/s", # Can change text, but needs a %f somewhere
                      arrowprops=dict(...) # set this to None for no arrow
                      
    """
    # Annotate max wind speed
    # only care about lower troposphere
    # 60 levels is about 2800m, 80 levels is about 4700m, 70 levels : 3700m
    if upto is None:
        upto=np.shape(winds)[0]
    # find max windspeed, put into same shape as winds [y,x]
    mloc = np.unravel_index(np.argmax(winds[:upto,:],axis=None),winds[:upto,:].shape)
    
    # Set default annotate args
    if 'textcoords' not in annotateargs:
        annotateargs['textcoords']='axes fraction'
    if 'fontsize' not in annotateargs:
        annotateargs['fontsize']=11
    if 'xytext' not in annotateargs:
        annotateargs['xytext']=(0, 1.03)
    if 'arrowprops' not in annotateargs:
        annotateargs['arrowprops'] = dict(
            arrowstyle='-|>',
            alpha=0.8,
            linestyle=(0,(1,5)),
            color='red',)
    
    # use indices to get how far the point is from top-right
    # 1-topright gives fractional distance from botleft (for xy)
    annotateargs['xycoords'] = 'axes fraction'
    ymax,xmax = np.shape(winds)
    annotateargs['xy'] = [1-(xmax - mloc[1])/xmax,1-(ymax - mloc[0])/ymax]
    
    if 'text' not in annotateargs:
        annotateargs['text'] = "wind max = %5.1f ms$^{-1}$"%(winds[:upto,:][mloc])
    else:
        annotateargs['text'] = annotateargs['text']%(winds[:upto,:][mloc])
    
    plt.annotate(**annotateargs)

def contours_symlog(w,levels=5):
    """
    Jeff's handy log spacer
    with <levels> colours log spaced on each side of 0
    """
    wmn = w.min()
    wmx = w.max()
    w1 = np.ceil(np.log2(np.maximum(wmx,-wmn)))
    wc = 2.0**(w1+np.arange(-1*levels,1))
    wc = np.union1d([0],np.union1d(wc,-wc))
    
    # Then the following code will plot w[k,:,:] with double-ended log scaling, at powers of 2, and a sensible colorbar:
    #ax = plt.subplot(nr,2,2*ir+1,aspect='equal',sharex=ax,sharey=ax)
    #wcon1 = wcon(w[k,:,:])
    #plt.contourf(x*1e-3,y*1e-3,w[k,:,:],wcon1,cmap='RdYlBu_r',norm=col.SymLogNorm(wcon1[-1]/16,base=2.0))
    #plt.colorbar(format=tick.LogFormatterSciNotation(base=2.0,minor_thresholds=(np.inf,np.inf)))
    
    return wc



def map_add_grid(ax, **gridargs):
    """
    add gridlines to some plot axis
    ARGUMENTS:
        ax: plt axis instance
        arguments for ax.gridlines
            defaults are draw_labels=True, color='gray', alpha=0.35, ls='--', lw=2
    """
    if 'draw_labels' not in gridargs:
        gridargs['draw_labels']=True
    if 'linewidth' not in gridargs:
        gridargs['linewidth']=2
    if 'color' not in gridargs:
        gridargs['color']='gray'
    if 'alpha' not in gridargs:
        gridargs['alpha']=0.35
    if 'linestyle' not in gridargs:
        gridargs['linestyle']='--'
    
    gl = ax.grid(gridargs)
    #gl.xlabels_top = False
    #gl.ylabels_left = False
    #gl.xlines = False

def map_add_locations_extent(extent, 
                             hide_text=False,
                             color='grey', 
                             nice=True,
                             dx=.025,
                             dy=.015,
                             **nice_text_args,
                             ):
    '''
    wrapper for map_add_locations that adds all the points for that extent
    dx,dy are distance between text and location (in degrees)
    '''
    names,latlons = utils.locations_from_extent(extent)
    
    # Capitalise all words in name
    Names = [name.title() for name in names]
    if hide_text:
        Names = ['']*len(names)
    
    if nice:
        map_add_nice_text(plt.gca(),latlons=latlons,texts=Names,**nice_text_args)
    else:
        map_add_locations(latlons, texts=Names, color=color, textcolor='k', dx=dx,dy=dy)

def map_add_locations(latlons, texts, proj=None,
                      marker='o', color='grey', markersize=None, 
                      textcolor='k',
                      dx=.025,dy=.015):
    '''
    input is list of names to be added to a map using the lat lons in _latlons_
    '''
    for i,(latlon,name) in enumerate(zip(latlons,texts)):
        y,x=latlon
        
        # dx,dy can be scalar or list
        dxi,dyi = dx,dy
        if isinstance(dx, (list,tuple,np.ndarray)):
            dxi,dyi = dx[i], dy[i]

        # Add marker and text
        pltargs = {
                "color":color,
                "linewidth":0,
                "marker":marker,
                "markersize":markersize,
                }
        txtargs = {
                "color":color,
                "horizontalalignment":"right",
                }
        if proj is not None:
            pltargs['transform']=proj
            txtargs['transform']=proj
        
        plt.plot(x,y, **pltargs)
        plt.text(x+dxi, y+dyi, name, **txtargs)

def map_add_rectangle(LRBT, **rectargs):
    """
        Add rectangle based on passed in [left,right,bottom,top]
        can also pass arguments to patches.Rectangle
    """
    ## Add box around zoomed in area
    xy = [LRBT[0], LRBT[2]]
    width = LRBT[1]-LRBT[0]
    height = LRBT[3]-LRBT[2]
    rectargs['xy']=xy
    rectargs['width']=width
    rectargs['height']=height
    if 'facecolor' not in rectargs:
        rectargs['facecolor']=None
    if 'fill' not in rectargs:
        rectargs['fill']=False
    if 'edgecolor' not in rectargs:
        rectargs['edgecolor']='k'
    if 'linewidth' not in rectargs:
        rectargs['linewidth']=2
    #linestyle='-'
    if 'alpha' not in rectargs:
        rectargs['alpha']=0.7
    #transform=proj
    plt.gca().add_patch(patches.Rectangle(**rectargs))

def map_draw_gridlines(ax, linewidth=1, color='black', alpha=0.5, 
                       linestyle='--', draw_labels=True,
                       gridlines=None):
    gl = ax.gridlines(linewidth=linewidth, color=color, alpha=alpha, 
                      linestyle=linestyle, draw_labels=draw_labels)
    if draw_labels:
        gl.top_labels = gl.right_labels = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
    if gridlines is not None:
        yrange,xrange=gridlines
        gl.xlocator = matplotlib.ticker.FixedLocator(xrange)
        gl.ylocator = matplotlib.ticker.FixedLocator(yrange)

def map_contourf(data, lat,lon, title="",
                 cbargs={}, **contourfargs):
    '''
    Show topography map matching extents
    '''
    cs = plt.contourf(lon,lat,data, **contourfargs)
    cb = None
    
    # set x and y limits to match extent
    xlims = lon[0],lon[-1] # East to West
    ylims = lat[0],lat[-1] # South to North
    plt.ylim(ylims); plt.xlim(xlims)
    if len(cbargs)>0:
        if "label" not in cbargs:
            cbargs['label']=""
        if "pad" not in cbargs:
            cbargs['pad']=0.01
        cb=plt.colorbar(**cbargs)
    plt.title(title)
    ## Turn off the tick values
    plt.xticks([]); plt.yticks([])
    return cs, cb


def map_fire(ff,lats,lons, **contourargs):
    """   
    Plot fire outline onto either an XY grid or geoaxes (eg from map_tiff)
    if mapping latlons onto geoaxes transform needs to be True
    INPUTS:
        ff : 2d array [lats, lons]
        lats,lons : 1darray (degrees)
        
    """
    if ff is None:
        return None
    # unmask if necessary
    if np.ma.is_masked(ff):
        ff = ff.data
    # default fire colour:
    if 'colors' not in contourargs:
        contourargs['colors']='red'
    fflatlon = ff
    fireline = None
    # May need to transpose the fire
    if (ff.shape[0] == len(lons)) and (ff.shape[0] != len(lats)):
        fflatlon = ff.T
        #print("WARNING: fire is in [lon,lat]")
    # only plot if there is fire
    if np.sum(ff<0) > 0:
        fireline = plt.contour(lons,lats,fflatlon,np.array([0]),**contourargs)
    
    return fireline 

def map_sensibleheat(sh, lat, lon,
                     levels = np.logspace(2,5,30),
                     colorbar=True,
                     cbar_kwargs={},
                     **contourfargs):
    """
    ARGUMENTS:
        sh [lat,lon] : sensible heat flux
        levels : color levels to be contourfed (optional)
        colorbar : {True | False} add colourbar?
        cbar_kwargs: dict()
            arguments for plt.colorbar(cs, **cbar_kwargs)
        additional contourf arguments are sent through to contourf
    RETURNS:
        cs, cbar: contour return value, colorbar handle
    """
    flux = sh + 0.01 # get rid of zeros
    if np.nanmax(flux) < 100:
        return None,None
    
    # set defaults
    if 'norm' not in contourfargs:
        contourfargs['norm']=col.LogNorm()
    if 'vmin' not in contourfargs:
        contourfargs['vmin']=100
    if 'cmap' not in contourfargs:
        contourfargs['cmap']='YlOrRd'# was 'gnuplot2'
    #if 'extend' not in contourfargs:
    #    contourfargs['extend']='max'
    
    shlatlon=flux
    if (sh.shape[0] == len(lon)) and (sh.shape[0] != len(lat)):
        shlatlon = sh.T
        print("WARNING: Heat flux is in [lon,lat]")
    
    ## Do the filled contour plot
    cs = plt.contourf(lon, lat, shlatlon,
                      levels, # color levels
                      **contourfargs,
                      )
    cbar=None
    if colorbar:
        # colorbar default arguments
        if 'orientation' not in cbar_kwargs:
            cbar_kwargs['orientation'] = 'vertical'
        if 'shrink' not in cbar_kwargs:
            #fraction by which to multiply the size of the colorbar
            cbar_kwargs['shrink'] = 0.6
        if 'aspect' not in cbar_kwargs:
            # ratio of long to short dimensions
            cbar_kwargs['aspect'] = 30 
        if 'pad' not in cbar_kwargs:
            # space between cbar and fig axis
            cbar_kwargs['pad'] = 0.01
        #if 'extend' not in cbar_kwargs:
        #    cbar_kwargs['extend']='max'
        if 'ticks' not in cbar_kwargs:
            cbar_kwargs['ticks'] = [1e2,1e3,1e4,1e5],
            #xtick_labels=['2','3','4','5']
        #else:
        #    xtick_labels=cbar_kwargs['ticks']
        if 'label' not in cbar_kwargs:
            cbar_kwargs['label'] ='Wm$^{-2}$'
        
        cbar=plt.colorbar(cs, **cbar_kwargs)
        # Need to set xticks prior to calling set_xticklabels
        # somehow the 'ticks' kwarg may not do this in colorbar method
        #cbar.ax.set_xticks(cbar_kwargs['ticks'])
        #cbar.ax.set_xticklabels(xtick_labels)
        
    #plt.xticks([],[])
    #plt.yticks([],[])
    return cs, cbar
    

def map_cartopy(extent,coastlines=True,bluemarble=False):
    """
    Basic map using PlateCarree() projection from cartopy
    """
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    #Set the extent (x0, x1, y0, y1) of the map in the given coordinate system.
    ax.set_extent(extent)

    if coastlines:
        ax.coastlines()
    if bluemarble:
        ax.bluemarble()
    return plt.gcf(), ax

def map_tiff_qgis(fname='sirivan.tiff', extent=None, show_grid=False,
                  locnames=None,
                  fig=None, subplot_row_col_n=[1,1,1], subplot_axes=None,
                  aspect='auto'):
    """
    satellite image from ESRI, some just OSM .tiff files on epsg 4326 (platecarree)
    
    ARGUMENTS:
        fname: "sirivan_map_linescan.tiff" (Searches in data/QGIS/)
        extent: [left,right,bot,top] (lats and lons)
            where is projection zoomed in to
        show_grid: {True|False}
            show lats and lons around map
        locnames: named points to add to map (from utilities/plotting.py)
        fig: matplotlib pyplot figure (default=[1,1,1])
            can add map to an already created figure object
        subplot_row_col_n: [row,col,n] (optional)
            subplot axes to create and add map to
        subplot_extent: [x0,y0,width,height]
            where to put axes inside figure (can use instead of subplot_row_col_n)
        aspect: {'equal' | 'auto'} # default is 'auto', equal prevents pixel stretch
    """
    gdal.UseExceptions()
    
    path_to_tiff = __DATADIR__+"QGIS/"+fname
    
    if subplot_row_col_n is None:
        subplot_row_col_n = [1,1,1]
    
    # gdal dataset
    # this prints a warning statement... pipe output and warning filter don't work
    ds = gdal.Open(path_to_tiff)

    # RGBa image read in as a numpy array
    img = plt.imread(path_to_tiff)
    
    # projection defined in QGIS
    # 4326 is not actually 'projected' - except it is PlateCarree!
    # projection=ccrs.PlateCarree() if EPSG == 4326 else ccrs.epsg(str(EPSG))
    
    # geotransform for tiff coords
    # tells us the image bounding coordinates
    gt = ds.GetGeoTransform()
    imgextent = (gt[0], gt[0] + ds.RasterXSize * gt[1],
                 gt[3] + ds.RasterYSize * gt[5], gt[3])
    
    ## set up figure and axes if needed
    if fig is None:
        fig = plt.figure(figsize=(13, 9))
    if subplot_axes is not None:
        ax = fig.add_axes(subplot_axes, frameon=False)
    else:
        nrows,ncols,n = subplot_row_col_n
        ax = fig.add_subplot(nrows,ncols,n)
    
    # Display tiff and cut to extent
    ax.imshow(img,
              extent=imgextent, 
              origin='upper',
              aspect=aspect)
    if extent is not None:
        #ax.set_extent(extent)
        ax.set_xlim([extent[0],extent[1]]) # E-W
        ax.set_ylim([extent[2],extent[3]]) # S-N
        
        # if we have locations and defined extent, put them in 
        if locnames is not None:
            for locstr in locnames:
                loc=_latlons_[locstr]
                if (loc[0]<extent[2]) or (loc[0]>extent[3]) or (loc[1]<extent[0]) or (loc[1]>extent[1]):
                    locnames.remove(locstr)
        
    if show_grid:
        map_add_grid(ax)
    if locnames is not None:
        # split out fire into it's own thing
        fires = ['fire' in element for element in locnames]
        locations=[_latlons_[locstr] for locstr in locnames]
        LocationsNames = [str.capitalize(locstr) for locstr in locnames]
        markers=['o']*len(locations)
        markercolors=['grey']*len(locations)
        
        for i in range(len(locations)):
            if fires[i]:
                LocationsNames[i]=''
                markers[i] = '*'
                markercolors[i] = 'r'
        
        map_add_nice_text(ax, locations,
                          LocationsNames,
                          markers=markers,
                          markercolors=markercolors,
                          fontsizes=14)        
        
    return fig, ax

def map_tiff_gsky(locname='waroona', fig=None, subplot_row_col_n=None,
             extent=None, show_grid=False, locnames=None):
    """
    satellite image from gsky downloaded into data folder, 
    used as background for figures
    
    ARGUMENTS:
        locname: { 'waroona' | 'sirivan' }
            which tiff are we using
        fig: matplotlib pyplot figure (default=[1,1,1])
            can add map to an already created figure object
        subplot_row_col_n: [row,col,n] (optional)
            subplot axes to create and add map to
        extent: [lon0, lon1, lat0, lat1]
            where is projection zoomed in to
        show_grid: {True|False}
            show lats and lons around map
        add_locations: {True|False}
            show nice text and markers for main points of interest
    """
    gdal.UseExceptions()
    
    path_to_tiff = "data/Waroona_Landsat_8_2015.tiff"
    
    if locname=='sirivan':
        path_to_tiff = "data/Sirivan_Landsat_8_2016.tiff"
    
    elif locname=='waroona_big':
        path_to_tiff = "data/waroona_big.tiff"
    
    
    if subplot_row_col_n is None:
        subplot_row_col_n = [1,1,1]
    
    # units are degrees latitude and longitude, so platecarree should work...
    projection = ccrs.PlateCarree()
    
    # gdal dataset
    ds = gdal.Open(path_to_tiff)
    # ndarray of 3 channel raster (?) [3, 565, 1024]
    data = ds.ReadAsArray()
    # np.sum(data<0) # 9 are less than 0
    data[data<0] = 0
    # 3 dim array of red green blue values (I'm unsure on what scale)
    R,G,B = data[0], data[1], data[2]
    # set them to a float array from 0 to 1.0
    scaled0 = np.array([R/np.max(R), G/np.max(G), B/np.max(B)])
    # image is too dark
    # gain=.3 and bias=.2 to increase contrast and brightness
    scaled = scaled0*1.3 + 0.1
    scaled[scaled>1] = 1.0
    # geotransform for tiff coords (?)
    gt = ds.GetGeoTransform()
    # extent = lon0, lon1, lat0, lat1
    fullextent = (gt[0], gt[0] + ds.RasterXSize * gt[1],
                  gt[3] + ds.RasterYSize * gt[5], gt[3])
    
    ## Third: the figure
    if fig is None:
        fig = plt.figure(figsize=(13, 9))
    #subplot_kw = dict(projection=projection)
    
    # create plot axes
    nrows, ncols, n = subplot_row_col_n
    ax = fig.add_subplot(nrows, ncols, n, projection=projection)
    
    ax.imshow(scaled[:3, :, :].transpose((1, 2, 0)), 
              extent=fullextent,
              origin='upper')
    
    if extent is not None:
        ax.set_extent(extent)
        # Remove any locations outside of the extent
        if locnames is not None:
            for locstr in locnames:
                loc=_latlons_[locstr]
                if (loc[0]<extent[2]) or (loc[0]>extent[3]) or (loc[1]<extent[0]) or (loc[1]>extent[1]):
                    locnames.remove(locstr)
        
    if show_grid:
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=2, color='gray', alpha=0.35, linestyle='--')
        gl.xlabels_top = False
        gl.ylabels_left = False
        #gl.xlines = False
    if locnames is not None:
        # split out fire into it's own thing
        fires = ['fire' in element for element in locnames]
        locations=[_latlons_[locstr] for locstr in locnames]
        LocationsNames = [str.capitalize(locstr) for locstr in locnames]
        markers=['o']*len(locations)
        markercolors=['grey']*len(locations)
        
        for i in range(len(locations)):
            if fires[i]:
                LocationsNames[i]=''
                markers[i] = '*'
                markercolors[i] = 'r'
        
        map_add_nice_text(ax, locations,
                          LocationsNames,
                          markers=markers,
                          markercolors=markercolors,
                          fontsizes=14)        
        
    return fig, ax, projection


def map_quiver(u, v, lats, lons, nquivers=13, **quiver_kwargs):
    """
    wrapper for nice quiver overlay on a lat/lon map
    """
    # just want a fixed number of quivers
    nlats, nlons = u.shape
    latinds = np.round(np.linspace(0,nlats-1,nquivers)).astype(np.int)
    loninds = np.round(np.linspace(0,nlons-1,nquivers)).astype(np.int)
    u0 = u[latinds,:]
    u0 = u0[:,loninds]
    v0 = v[latinds,:]
    v0 = v0[:,loninds]
    # subset lats and lons
    if len(np.shape(lats))==1:
        sublats=lats[latinds]
        sublons=lons[loninds]
    # maybe we want to use 2D lats/lons
    elif len(np.shape(lats))==2:
        sublats=lats[latinds,:]
        sublats=sublats[:,loninds]
        sublons = lons[latinds,:]
        sublons = sublons[:,loninds]
    else:
        print("ERROR: Why are the lats 3 dimensional?")
        assert False
        
    ## Default quiver scale:
    if 'scale' not in quiver_kwargs:
        quiver_kwargs['scale'] = 85
        
    ## Default quiver pivot:
    if 'pivot' not in quiver_kwargs:
        quiver_kwargs['pivot'] = 'middle'
    
    plt.quiver(sublons, sublats, u0, v0, **quiver_kwargs)


def map_add_nice_text(ax, latlons, texts=None, markers=None, 
                      fontsizes=9, fontcolors='wheat', 
                      markercolors='grey', markersizes=None,
                      outlinecolors='k', transform=None):
    '''
    ARGUMENTS:
        ax: plot axis
        latlons: iterable of (lat, lon) pairs
        texts: iterable of strings, optional
        markers: iterable of characters, optional
        transform: if using geoaxes instance (non platecarree map) then set this to ccrs.Geodetic() or PlateCarree()?
    '''
    fontname='Droid Sans'
    ## Adding to a map using latlons can use the geodetic transform
    #geodetic_CRS = ccrs.Geodetic()
    transformargs = {}
    if transform is not None:
        if transform == True:
            transform=ccrs.Geodetic()
        transformargs['transform']=transform
    
    # Make everything iterable
    if texts is None:
        texts = ''*len(latlons)
    if markers is None:
        markers = 'o'*len(latlons)
    if isinstance(fontsizes, (int,float)):
        fontsizes = [fontsizes]*len(latlons)
    if (markersizes is None) or (isinstance(markersizes, (int,float))):
        markersizes = [markersizes]*len(latlons)
    if isinstance(fontcolors, str):
        fontcolors = [fontcolors]*len(latlons)
    if isinstance(markercolors, str):
        markercolors = [markercolors]*len(latlons)
    if isinstance(outlinecolors, str):
        outlinecolors = [outlinecolors]*len(latlons)

    
    
    # add points from arguments:
    for (lat, lon), text, marker, mcolor, msize, fcolor, fsize, outlinecolor in zip(latlons, texts, markers, markercolors, markersizes, fontcolors, fontsizes, outlinecolors):
        
        # outline for text/markers
        marker_effects=[patheffects.Stroke(linewidth=4, foreground=outlinecolor), patheffects.Normal()]
        text_effects = [patheffects.withStroke(linewidth=2, foreground=outlinecolor)]
        
        # Add the point to the map with patheffect
        ax.plot(lon, lat, color=mcolor, linewidth=0, 
                marker=marker, markersize=msize,
                path_effects=marker_effects,
                **transformargs)
        
        if len(text)>0:
            # Add text to map
            txt = ax.text(lon, lat, text, 
                          fontsize=fsize, 
                          color=fcolor,
                          fontname=fontname,
                          **transformargs)
            # Add background (outline)
            txt.set_path_effects(text_effects)
    
def map_topography(topog,lats,lons, cbar=False, **contourfargs):
    '''
    Show topography map matching extents
    '''
    vmax=np.max(topog)
    # defaults
    contourfargs["norm"] = matplotlib.colors.Normalize(-150,np.max(topog))
    contourfargs["cmap"] = "terrain"
    contourfargs["levels"] = np.union1d(np.array([-1,0]), np.linspace(0.05,vmax,num=200))
    
    #topo_cmap, topo_levels, topo_norm = topography_colouring(DA_topog.values)
    #levels = np.linspace(-150,np.max(topog),300,endpoint=True)
    
    plt.contourf(lons,lats,topog,
                 **contourfargs)
    
    cbar_args={}
    if cbar:
        # defaults if not set
        #orientation 	vertical or horizontal
        #fraction 	0.15; fraction of original axes to use for colorbar
        #pad 	0.05 if vertical, 0.15 if horizontal; fraction of original axes between colorbar and new image axes
        #shrink 	1.0; fraction by which to multiply the size of the colorbar
        #aspect 	20; ratio of long to short dimensions
        if 'pad' not in cbar_args:
            cbar_args['pad']=0.01
        if 'aspect' not in cbar_args:
            cbar_args['aspect']=30
        if 'shrink' not in cbar_args:
            cbar_args['shrink'] = 0.85
        if 'fraction' not in cbar_args:
            cbar_args['fraction'] = .075
        plt.colorbar(**cbar_args)
    

def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

def quiverwinds(lats,lons,u,v,
                thresh_windspeed=10,
                n_arrows=23,
                no_defaults=False,
                add_quiver_key=True,
                differential=False,
                **quivargs):
    """
    u, v in metres/second
    arrows and labels converted into km/h
    ARGS:
        lats,lons,u,v,
        thresh_windspeed=10 km/h
        differential=False: set to True to instead plot quiver difference from mean flow
    """
    #set some defaults
    if (not no_defaults) and (not differential):
        if "scale_units" not in quivargs.keys() and 'scale' not in quivargs.keys():
            # I think this makes 50 m/s 1 inch 
            quivargs['scale_units']="inches"
            quivargs['scale']=50 * 3.6
        if "pivot" not in quivargs.keys():
            quivargs['pivot']='middle'
    
    # xskip and yskip for visibility of arrows
    if n_arrows is not None:
        xskip,yskip=xyskip_for_quiver(lats,lons,n_arrows=n_arrows)
    else:
        xskip,yskip=1,1
    
    # may use 2d lats/lons
    if len(lats.shape)==2:
        qlats=lats[::yskip,::xskip]
        qlons=lons[::yskip,::xskip]
    else:
        qlons=lons[::xskip]
        qlats=lats[::yskip]
    
    # if we want to look at differential wind speed we need to subtract the mean flow
    # also change the wind threshold etc...
    if differential:
        ubar = np.nanmean(u)
        vbar = np.nanmean(v)
        u_diff = (u[::yskip,::xskip] - ubar)*3.6  # m/s -> km/h
        v_diff = (v[::yskip,::xskip] - vbar)*3.6  # m/s -> km/h
        diffstr = "(u-%.1f,v-%.1f)\n"%(ubar*3.6,vbar*3.6)
        Q = plt.quiver(qlons,qlats,
                u_diff,
                v_diff,
                **quivargs)
        plt.quiverkey(Q, 0.1, 1.06, 20, diffstr + r' = $20 \frac{km}{h}$', labelpos='W', fontproperties={'weight':'bold','size':'8'})
        return
        
    # wins speed used as threshhold for quiver arrows
    qs = np.sqrt(u[::yskip,::xskip]**2+v[::yskip,::xskip]**2) * 3.6 # km/h
    
    qu = np.ma.masked_where(qs<thresh_windspeed, 
                            u[::yskip,::xskip]) * 3.6
    qv = np.ma.masked_where(qs<thresh_windspeed, 
                            v[::yskip,::xskip]) * 3.6
    
    ## color quivers above 90km/h
    #cmap = matplotlib.colors.ListedColormap(['black', 'magenta'])
    #bounds=[0,90,1000]
    #norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    #print("DEBUG: quiverwinds:", qlons.shape, qlats.shape, qu.shape, qv.shape, yskip,xskip)
    Q = plt.quiver(qlons, qlats, 
               qu, qv, 
               #color=cmap(norm(qs.data)), #doesn't work!?
               **quivargs)
    
    if add_quiver_key:
        plt.quiverkey(Q, 0.1, 1.05, 50, r'$50 \frac{km}{h}$', labelpos='W', fontproperties={'weight': 'bold'})
    
    ## overlay quivers above thresh for clarity
    qu2 = np.ma.masked_where(qs<90, 
                            u[::yskip,::xskip]) * 3.6
    qv2 = np.ma.masked_where(qs<90, 
                            v[::yskip,::xskip]) * 3.6
    Q2 = plt.quiver(qlons, qlats, 
                qu2, qv2,
                color="magenta",
                **quivargs)
    

def set_spine_color(ax,color,spines=['bottom','top'], cap="butt", linewidth=2, linestyle="dashed"):
    """
        set spine color
    """
    for spine in [ax.spines[spinestr] for spinestr in spines]:
        spine.set_linestyle(linestyle)
        spine.set_capstyle(cap)
        spine.set_color(color)
        spine.set_linewidth(linewidth)

def scale_bar(ax, proj, length, location=(0.5, 0.05), linewidth=3,
              units='km', m_per_unit=1000):
    """
    http://stackoverflow.com/a/35705477/1072212
    ax is the axes to draw the scalebar on.
    proj is the projection the axes are in
    location is center of the scalebar in axis coordinates ie. 0.5 is the middle of the plot
    length is the length of the scalebar in km.
    linewidth is the thickness of the scalebar.
    units is the name of the unit
    m_per_unit is the number of meters in a unit
    """
    # find lat/lon center to find best UTM zone
    x0, x1, y0, y1 = ax.get_extent(proj.as_geodetic())
    #x0, x1, y0, y1 = ax.get_extent(proj)
    # Projection in metres
    utm = cartopy.crs.UTM(utm_from_lon((x0+x1)/2))
    # Get the extent of the plotted area in coordinates in metres
    x0, x1, y0, y1 = ax.get_extent(utm)
    # Turn the specified scalebar location into coordinates in metres
    sbcx, sbcy = x0 + (x1 - x0) * location[0], y0 + (y1 - y0) * location[1]
    # Generate the x coordinate for the ends of the scalebar
    bar_xs = [sbcx - length * m_per_unit/2, sbcx + length * m_per_unit/2]
    # buffer for scalebar
    buffer = [patheffects.withStroke(linewidth=5, foreground="w")]
    # Plot the scalebar with buffer
    ax.plot(bar_xs, [sbcy, sbcy], transform=utm, color='k',
        linewidth=linewidth, path_effects=buffer)
    # buffer for text
    buffer = [patheffects.withStroke(linewidth=3, foreground="w")]
    # Plot the scalebar label
    t0 = ax.text(sbcx, sbcy, str(length) + ' ' + units, transform=utm,
        horizontalalignment='center', verticalalignment='bottom',
        path_effects=buffer, zorder=2)
    left = x0+(x1-x0)*0.05
    # Plot the N arrow
    #t1 = ax.text(left, sbcy, u'\u25B2\nN', transform=utm,
    #    horizontalalignment='center', verticalalignment='bottom',
    #    path_effects=buffer, zorder=2)
    # Plot the scalebar without buffer, in case covered by text buffer
    ax.plot(bar_xs, [sbcy, sbcy], transform=utm, color='k',
        linewidth=linewidth, zorder=3)



def streamplot_regridded(x,y,u,v,**kwargs):
    """
    JeffK transect interpolation and streamplotting
    dynamically calculates gridspacing based on smallest gridsize in u,v
    ARGUMENTS:
        x = array [[y],x] (metres)
        y = array [y,[x]](metres)
        u = 2darray [y,x] (m/s)
        v = 2darray [y,x] (m/s)
    """
    #print("DEBUG: x:",x)
    #print("DEBUG: y:",y)
    if 'minlength' not in kwargs:
        # increase minimum line length (default is 0.1)
        kwargs['minlength']=0.3
    # figure out xn,yn based on smallest grid space in that direction
    if len(x.shape) > 1:
        nx = int(np.ceil((x.max()-x.min())/np.min(np.diff(x,axis=1))))
        ny = int(np.ceil((y.max()-y.min())/np.min(np.diff(y,axis=0))))
    else:
        nx = int(np.ceil((x.max()-x.min())/np.min(np.diff(x))))
        ny = int(np.ceil((y.max()-y.min())/np.min(np.diff(y))))
    xi = np.linspace(x.min(),x.max(),nx)
    yi = np.linspace(y.min(),y.max(),ny)
    #with warnings.catch_warnings():
        #warnings.simplefilter("ignore")
    print("DEBUG: PLOTTING.STREAMPLOT_REGRIDDED INTERP START")
    ui = interp2d(x,y,u,bounds_error=False,fill_value=np.NaN)(xi,yi)
    vi = interp2d(x,y,v,bounds_error=False,fill_value=np.NaN)(xi,yi)
    print("DEBUG: PLOTTING.STREAMPLOT_REGRIDDED INTERP END1")
    # if linewidth is an argument 
    if 'linewidth' in kwargs:
        # on the same grid as input u,v
        if len(kwargs['linewidth'].shape) > 1:
            lwi = kwargs['linewidth']
            kwargs['linewidth'] = interp2d(x,y,lwi,bounds_error=False,fill_value=0)(xi,yi)
            print("DEBUG: PLOTTING.STREAMPLOT_REGRIDDED INTERP END2")
    #print("DEBUG: xi:",xi)
    #print("DEBUG: yi:",yi)
    #print("DEBUG:",np.shape(ui))
    #print("DEBUG:",np.shape(vi))
    print("DEBUG: PLOTTING.STREAMPLOT_REGRIDDED STREAMPLOT START")
    splot = plt.streamplot(xi,yi,ui,vi,**kwargs)
    print("DEBUG: PLOTTING.STREAMPLOT_REGRIDDED STREAMPLOT END")
    # set limits back to latlon limits
    #plt.gca().set_ylim(yi[0],yi[-1])
    #plt.gca().set_xlim(xi[0],xi[-1])
    return splot

def xyskip_for_quiver(lats,lons,n_arrows=23):
    """
    return xskip,yskip to get approx n_arrows for a quiver plot using x[::xskip],y[::yskip]
    """
    xskip=int(np.max([len(lons)//n_arrows-1,1]))
    yskip=int(np.max([len(lats)//n_arrows-1,1]))
    return xskip,yskip

def utm_from_lon(lon):
    """
    utm_from_lon - UTM zone for a longitude

    Not right for some polar regions (Norway, Svalbard, Antartica)

    :param float lon: longitude
    :return: UTM zone number
    :rtype: int
    """
    return np.floor( ( lon + 180 ) / 6) + 1

