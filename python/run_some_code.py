#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 2021
    Generic code to send to queue
@author: jesse
"""

## turn off display 
import matplotlib
matplotlib.use('Agg')

import dask.bag as db

## local scripts that can be run
from cross_sections import topdown_view_only, multiple_transects
from weather_summary import weather_summary_model
#from fireplan 
from fire_spread import fire_spread
from winds import rotation_looped, wind_and_heat_flux_looped
# METHODS NEED TO HAVE 1st 3 arguments: run name, WESN, subdir
fnlist = [
    rotation_looped, 
    wind_and_heat_flux_looped, 
    topdown_view_only,
    fire_spread, 
    weather_summary_model, 
    multiple_transects, 
    fire_spread,
    ]

## keep track of used zooms
KI_zoom = [136.5,137.5,-36.1,-35.6]
KI_zoom_name = "zoom1"
badja_zoom=[149.4,150.0, -36.4, -35.99]
badja_zoom_name="zoom1"

## settings for plots
mr='badja_run1'
zoom = None
subdir = None

if 'badja' in mr:
    zoom = badja_zoom
    subdir=badja_zoom_name
elif 'KI' in mr:
    zoom = KI_zoom
    subdir=KI_zoom_name

## Argument list for multiple functions
bag = db.from_sequence([mr,zoom,subdir]) 
## list of functions to be mapped to processors
bag = bag.map(fnlist)
## run functions over whatever available processors
bag.compute() 


