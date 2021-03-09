#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 16:48:58 2021
    Trying some pyNGL stuff
@author: jesse
"""


# plotting and maths
import numpy as np
import Ngl

# fetch file lists
from glob import glob

# dataset and dataarray netcdf handling library
import xarray as xr

# datetime stuff
from datetime import datetime

# my own stuff
from utilities import utils


##--  define variables
mr = "exploratory_KI_run1"
plotname = "test_plot_ngl1"
vertcross_lat = -36.0
# UTC for 9 hours after simulation start for KI
UTC = datetime(2020,1,3,0)
pname_vertcross = "test_plot_vertcross"

diri   = "../data/%s/"%mr     #-- data directory
files_mdl_th1 = glob(diri+'/atmos/*mdl_th1.nc')
files_mdl_ro1 = glob(diri+'/atmos/*mdl_ro1.nc')
#file_mdl_th1 = files_mdl_th1
#file_mdl_ro1 = files_mdl_ro1
file_firefront = glob(diri+'/fire/firefront*.nc')[0]
file_u10 = glob(diri+'/fire/10m_uwind*.nc')[0]
file_v10 = glob(diri+'/fire/10m_vwind*.nc')[0]

minval =  250.                              #-- minimum contour level
maxval =  315                               #-- maximum contour level
inc    =    5.                              #-- contour level spacing

#--  open file and read variables

# datasets from file
DS_mdl_th1 = xr.open_mfdataset(files_mdl_th1)    #-- open data file
# 10m winds
DS_u10 = xr.open_dataset(file_u10)
DS_v10 = xr.open_dataset(file_v10)
# air temp
## Pull out data arrays (still lazy I think- not loaded yet)
## get nearest to some particular time step
DA_temp0   = DS_mdl_th1["air_temp"][:,0,:,:].sel(time=UTC,method='nearest')
DA_temp3d   = DS_mdl_th1["air_temp"].sel(time=UTC,method='nearest')
DA_u10      = DS_u10["UWIND_2"].sel(time=UTC,method='nearest')
DA_v10      = DS_v10["VWIND_2"].sel(time=UTC,method='nearest')

vertcross_temp = DA_temp3d.sel(latitude=vertcross_lat,method='nearest').values
lat    = DA_temp0.latitude.values           #-- lats
lon    = DA_temp0.longitude.values          #-- lons
lev    = DA_temp3d.level_height.values
nlon   = len(lon)                           #-- number of longitudes
nlat   = len(lat)                           #-- number of latitudes

if False: # PLOTTING MAPS
    #-- open a workstation
    wkres           =  Ngl.Resources()          #-- generate an resources object for workstation
    wkres.wkWidth   =  1024                     #-- plot resolution 2500 pixel width
    wkres.wkHeight  =  1024                     #-- plot resolution 2500 pixel height
    wks_type        = "png"                     #-- graphics output type
    wks             =  Ngl.open_wks(wks_type,plotname,wkres)
    
    #-- create 1st plot: vectors on global map
    res                           =  Ngl.Resources()
    res.vfXCStartV                =  lon[0]        #-- minimum longitude
    res.vfXCEndV                  =  lon[-1]    #-- maximum longitude
    res.vfYCStartV                =  lat[0]        #-- minimum latitude
    res.vfYCEndV                  =  lat[-1]    #-- maximum latitude
    
    res.tiMainString              = "~F25~Wind velocity vectors"  #-- title string
    res.tiMainFontHeightF         =  0.024                        #-- decrease title font size
    
    res.mpLimitMode               = "Corners"                     #-- select a sub-region
    res.mpLeftCornerLonF          =  lon[0]                #-- left longitude value
    res.mpRightCornerLonF         =  lon[-1]    #-- right longitude value
    res.mpLeftCornerLatF          =  lat[0]                #-- left latitude value
    res.mpRightCornerLatF         =  lat[-1]    #-- right latitude value
    
    res.mpPerimOn                 =  True                     #-- turn on map perimeter
    
    res.vpXF                      =  0.1                          #-- viewport x-position
    res.vpYF                      =  0.92                         #-- viewport y-position
    res.vpWidthF                  =  0.75                         #-- viewport width
    res.vpHeightF                 =  0.75                         #-- viewport height
    
    res.vcMonoLineArrowColor      =  False                        #-- draw vectors in color
    res.vcMinFracLengthF          =   0.33                        #-- increase length of vectors
    res.vcMinMagnitudeF           =   0.001                       #-- increase length of vectors
    res.vcRefLengthF              =   0.045                       #-- set reference vector length
    res.vcRefMagnitudeF           =  20.0                         #-- set reference magnitude value
    res.vcLineArrowThicknessF     =   6.0                         #-- make vector lines thicker (default: 1.0)
    
    res.pmLabelBarDisplayMode     = "Always"                      #-- turn on a labelbar
    res.lbOrientation             = "Horizontal"                  #-- labelbar orientation
    res.lbLabelFontHeightF        =  0.008                        #-- labelbar label font size
    res.lbBoxMinorExtentF         =  0.22                         #-- decrease height of labelbar boxes
    
    #-- draw first plot
    map1 = Ngl.vector_map(wks,DA_u10.values,DA_v10.values,res)           #-- draw a vector plot
    
    #-- create 2nd plot: sub-region colored by temperature variable
    
    res.mpLimitMode               = "LatLon"                      #-- change the area of the map
    #res.mpMinLatF                 =  -40.0                         #-- minimum latitude
    #res.mpMaxLatF                 =  -30.0                         #-- maximum latitude
    #res.mpMinLonF                 = 130.                         #-- minimum longitude
    #res.mpMaxLonF                 = 150.                          #-- minimum longitude
    res.mpMinLatF                 =  lat[0]
    res.mpMaxLatF                 =  lat[-1]
    res.mpMinLonF                 = lon[0]
    res.mpMaxLonF                 = lon[-1]
    
    res.mpFillOn                  =  True                         #-- turn on map fill
    res.mpLandFillColor           =  16                           #-- change land color to grey
    res.mpOceanFillColor          =  -1                           #-- change color for oceans and inlandwater
    res.mpInlandWaterFillColor    =  -1                           #-- set ocean/inlandwater color to transparent
    res.mpGridMaskMode            = "MaskNotOcean"                #-- draw grid over ocean, not land
    res.mpGridLineDashPattern     =   2                           #-- grid dash pattern
    #res.mpOutlineBoundarySets     = "GeophysicalAndUSStates"      #-- outline US States
    
    res.vcFillArrowsOn            =  True                         #-- fill the vector arrows
    res.vcMonoFillArrowFillColor  =  False                        #-- draw vectors with colors
    res.vcFillArrowEdgeColor      =  1                            #-- draw the edges in black
    res.vcGlyphStyle              = "CurlyVector"                 #-- draw nice curly vectors
    res.vcLineArrowThicknessF     =   5.0                         #-- make vector lines thicker (default: 1.0)
    
    res.tiMainString              = "~F25~Wind velocity vectors"  #-- title string
    
    res.lbTitleString             = "TEMPERATURE (~S~o~N~F)"      #-- labelbar title string
    res.lbTitleFontHeightF        =  0.010                        #-- labelbar title font size
    res.lbBoxMinorExtentF         =  0.18                         #-- decrease height of labelbar boxes
    
    #-- draw 2nd plot
    map2 = Ngl.vector_scalar_map(wks,DA_u10.values,DA_v10.values,DA_temp0.values,res)
    
    #-- done
    Ngl.end()


#### SLICE CODE
#-- get the minimum and maximum of the data
minval =  int(np.amin(DA_temp3d.values))                   #-- minimum value
maxval =  int(np.amax(DA_temp3d.values))                   #-- maximum value
inc    =  5                                    #-- contour level spacing
ncn    =  (maxval-minval)/inc + 1              #-- number of contour levels

#-- open a workstation
wkres                     =  Ngl.Resources()   #-- generate an res object for workstation
wkres.wkWidth             =  1024              #-- plot resolution 2500 pixel width
wkres.wkHeight            =  1024              #-- plot resolution 2500 pixel height
wks_type                  = "png"              #-- output type
wks                       =  Ngl.open_wks(wks_type,pname_vertcross,wkres)  #-- open workstation

#-- set resources
res                       =  Ngl.Resources()   #-- generate an res object for plot
res.tiMainString =  DA_temp0.name + " at latitude " + "{:.2f}".format(vertcross_lat) 
                                               #-- set main title
#-- viewport resources
res.vpXF                  =  0.1               #-- start x-position of viewport
res.vpYF                  =  0.9               #-- start y-position of viewport
res.vpWidthF              =  0.7               #-- width of viewport
res.vpHeightF             =  0.7               #-- height of viewport

#-- contour resources
res.cnFillOn              =  True              #-- turn on contour fill
res.cnLineLabelsOn        =  False             #-- turn off line labels
res.cnInfoLabelOn         =  False             #-- turn off info label
res.cnLevelSelectionMode  = "ManualLevels"     #-- select manual levels
res.cnMinLevelValF        =  minval            #-- minimum contour value
res.cnMaxLevelValF        =  maxval            #-- maximum contour value
res.cnLevelSpacingF       =  inc               #-- contour increment

#-- grid resources
res.sfXArray              =  lon               #-- scalar field x
res.sfYArray              =  lev               #-- scalar field y

#-- labelbar resources
res.pmLabelBarDisplayMode = "Always"           #-- turn off the label bar

#-- draw slice contour plot
plot = Ngl.contour(wks,vertcross_temp,res)

Ngl.end()