#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Quick code I can figure out before saving into an appropriate script
@author: jesse
"""
import matplotlib
#matplotlib.use("Agg")

from matplotlib import colors, ticker, patches
from matplotlib import patheffects, gridspec
import matplotlib.dates as mdates

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from datetime import datetime, timedelta
from glob import glob
import os

from utilities import utils, plotting, fio, constants

## RUN
extent=constants.extents['badja']['zoom1']

mr1='badja_run3'
mr2='badja_LRC1'

for mr in [mr1,mr2]:
    print(mr)
    print("---------    -----------")
    #DS = fio.read_model_run_fire(mr)
    DS = fio.read_model_run_hour(mr)
    p = DS['pressure']
    #print(p)
    print(" - - - - ")
    #### HERE IS WHERE ISSUE OCCURS::#####
    p_sub = fio.extract_extent(p,extent)
    #print(p_sub)
    print(" ------- ")
    DS_sub2 = fio.read_model_run_hour(mr,extent=extent)
    print(DS_sub2)


