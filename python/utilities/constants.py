#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 10:04:08 2019

  globals used in analyses

  Style note:
  Trying to only use capital letters in abbreviations (eg. KI, NYE)

@author: jesse
"""

#__DATADIR__="/g/data/en0/jwg574/casestudies/data/"
__DATADIR__="../data/"
#Water content plus ice content demarking cloud (from toivanen2019) .1g/kg
# "liquid water mixing ratio is the ratio ofthe mass of liquid water to the mass of dry air in a given volume of air. Likewise, the ice mixing ratio is the ratio of the mass of frozen water to the mass of dry air"
cloud_threshold = 0.01 # g/kg cloud boundary of liquid + ice /kg air

# dictionaries for named locations/extents
latlons, extents={},{}

### EXTENTS ###
# extents{"location":{"zoomname":[W,E,S,N], }}
extents['KI'] = {
        "zoom1":[136.5,   137.5,   -36.1,   -35.6],
        "zoom2":[136.5887,136.9122,-36.047,-35.7371],
        "early_run":[136.6,136.9,-36.08,-35.79],
        }
extents['badja'] = {
        "zoom1":[149.4,     150.12,  -36.47,  -35.99],
        "Wandella":[149.5843,  149.88, -36.376, -36.223],
        "Belowra":[149.5308,149.9093,-36.2862,-36.0893],
        }
extents['badja_am'] = {
        "zoom1":extents['badja']['zoom1'],
        "zoom2":[149.4,  150.12,    -36.573, -35.99],
        "valley":[149.546,150.005,-36.553,-36.187],
        }


### LOCATIONS ###
####### Kangaroo Island ##########

# towns...
latlons['Parndana'] = -35.79, 137.262
#latlons['Cape Torrens'] = -35.74,136.75
latlons['Karatta'] = -35.9657, 137.0015
latlons['Emu Bay'] = -35.5941, 137.5447
latlons['Cape Borda'] = -35.76191,136.594
# show ignition area (roughly)

####### Badja extents ###############
latlons['Wandella'] = -36.3115, 149.8562
latlons['Tuross Falls'] = -36.2228, 149.5358
latlons['Belowra'] = -36.14807, 149.71676 # updated POST GOOGLE using qgis open street map
latlons['Yowrie'] = -36.31894, 149.73928 # Also updated with openstreet map
latlons['Cobargo'] = -36.3866, 149.9018
latlons['Big Belimbla Creek'] = -36.1001, 149.80365


####### Corryong ###############
latlons['Lightwood'] = -35.94174, 147.63655 # Lightowood, location of FGV and truck incident
latlons['Karumba'] = -36.00827, 147.69578
latlons['Corryong'] = -36.196, 147.902

## Extra locs
latlons['Sydney']    = -33.8688, 151.2093
latlons['Brisbane']  = -27.4698, 153.0251
latlons['Canberra']  = -35.2809, 149.1300
latlons['Melbourne'] = -37.8136, 144.9631

