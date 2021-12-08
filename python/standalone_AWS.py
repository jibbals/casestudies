# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 15:01:18 2021

@author: jgreensl
"""

import pandas
from matplotlib import pyplot as plt
from glob import glob # to read files

def DF_subset_time(DF, dt0=None, dt1=None, timename='localtime'):
    """
    subset AWS to time window dt0,dt1
    can do utc if you change timename to 'utc'
    """
    if dt0 is not None:
        time_mask = DF[timename] >= dt0
        DF = DF.loc[time_mask]
    
    if dt1 is not None:
        time_mask = DF[timename] <= dt1
        DF = DF.loc[time_mask]
    
    return DF


ddir="../data/AWS/"
AWS_allfiles=glob(ddir+"HM01X_Data*") 

li = []
for filename in AWS_allfiles:
    df = pandas.read_csv(filename, index_col=None, header=0)
    li.append(df)
DF_AWS = pandas.concat(li, axis=0, ignore_index=True)
# DF_AWS.rename(columns=rename_dictionary,inplace = True)

# convert the 'Date' column to datetime format
datetime_names = ["Day/Month/Year Hour24:Minutes in DD/MM/YYYY HH24:MI format in Local time",#:"localtime",
                  "Day/Month/Year Hour24:Minutes in DD/MM/YYYY HH24:MI format in Universal coordinated time",]#:"utc"
for dtcolumn in datetime_names:
    DF_AWS[dtcolumn] = pandas.to_datetime(DF_AWS[dtcolumn],dayfirst=True)

print(DF_AWS)

# select one station
station_name = "parndana"
DF_AWS_parn = DF_AWS.loc[DF_AWS['Station Name'].str.contains(str.upper(station_name))]

AWS_s = DF_AWS_parn["Wind speed in m/s"].values
lt_AWS = DF_AWS_parn["Day/Month/Year Hour24:Minutes in DD/MM/YYYY HH24:MI format in Local time"].values

plt.plot_date(lt_AWS, AWS_s, 
              color='k',
              fmt='-', 
              label='AWS')

# fix date formatting
plt.gcf().autofmt_xdate()
plt.xlabel('local time')