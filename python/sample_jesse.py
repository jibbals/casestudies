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

#from utilities import utils, plotting, fio, constants

## RUN
fhsns_path="../data/Fuels/KI_Vesta_Prefire_FHSNS.nc"
hns_path="../data/Fuels/KI_Vesta_Prefire_HNS.nc"
fhss_path="../data/Fuels/KI_Vesta_Prefire_FHSS.nc"

DS_FHSS=xr.open_dataset(fhss_path)
DS_FHSNS=xr.open_dataset(fhsns_path)
DS_HNS=xr.open_dataset(hns_path)

plt.figure(figsize=[14,14])
plt.pcolormesh(DS_FHSS['Band1'].values)
plt.colorbar()
plt.title("FHSS")
plt.savefig("FHSS_map.png")

