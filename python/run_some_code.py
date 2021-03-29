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

## local scripts that can be run
import cross_sections

## keep track of used zooms
KI_zoom = [136.5,137.5,-36.1,-35.6]
KI_zoom_name = "zoom1"
badja_zoom=[149.4,150.0, -36.4, -35.99]
badja_zoom_name="zoom1"

## settings for plots
mr='badja_run1'
zoom=badja_zoom
subdir=badja_zoom_name
tiffname=None

## Which script am I running?
run_cross_sections=False
run_winds=True

if run_winds:
    rotation_looped(mr,
            extent=zoom,
            subdir=subdir,
            )

    wind_and_heat_flux_looped(mr, 
            extent=zoom,
            subdir=subdir,
            )


if run_cross_sections:
    ## Multiple transects 
    cross_sections.multiple_transects(mr,extent=zoom,subdir=subdir)
    ## TOPDOWN 10m WINDS ONLY
    cross_sections.topdown_view_only(mr,extent=zoom,subdir=subdir)
    
