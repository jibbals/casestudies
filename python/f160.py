# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 12:28:05 2019
    F160 plot using MetPy
@author: jgreensl
"""

## metpy imports
## conda install -c conda-forge metpy
## 
## NO WAY! Also contained in NCI
## module use /g/data3/hh5/public/modules
## module load conda/analysis3

import matplotlib
#matplotlib.use('Agg',warn=False)

from metpy.units import units
#distance = np.arange(1, 5) * units.meters # easy way to add units (Also units.metre works!)
#g = 9.81 * units.meter / (units.second * units.second)
from metpy.plots import SkewT

# plotting stuff
import matplotlib.pyplot as plt
#import matplotlib.colors as col
#import matplotlib.ticker as tick
import numpy as np
from datetime import datetime
import iris # file reading and constraints etc
#import warnings

# local modules
from utilities import plotting, utils, constants
from utilities import fio_iris as fio

###
## GLOBALS
###
_sn_ = 'F160'


def f160(press,Temp,Tempd, latlon, 
         press_rho=None,uwind=None,vwind=None, 
         nearby=2, alpha=0.3):
    '''
    show skewt logp plot of temperature profile at latlon
    input cubes: pressures, temperatures are [z,lat,lon]
    profile for cubes interpolated to latlon point will be shown with linewidth of 2
    nearest values to lat lon within <nearby> indices will be plotted at low alpha
    
    Wind barbs will be shown if p_ro,u,v are set
    '''
    tcolor='r'
    tdcolor='g'
    
    ## first interpolate pressure and temperature to latlon
    lons=press.coord('longitude')
    lats=press.coord('latitude')
    #press.convert_units('hPa') # convert to hPa
    #Temp.convert_units('C') # convert to sir Kelvin
    #Tempd.convert_units('C') # convert to kelvin
    
    press0 = press.interpolate([('longitude',[latlon[1]]),
                                ('latitude',[latlon[0]])],
                               iris.analysis.Linear())
    temp0  = Temp.interpolate([('longitude',[latlon[1]]),
                            ('latitude',[latlon[0]])],
                           iris.analysis.Linear())
    tempd0 = Tempd.interpolate([('longitude',[latlon[1]]),
                             ('latitude',[latlon[0]])],
                            iris.analysis.Linear())
    
    # Plot T, and Td
    # pull out data array (units don't work with masked arrays)
    p = np.squeeze(press0.data.data) * units(str(press.units))
    p = p.to(units.mbar)
    T = np.squeeze(temp0.data.data) * units(str(Temp.units))
    T = T.to(units.degC)
    Td = np.squeeze(tempd0.data.data) * units(str(Tempd.units))
    Td = Td.to(units.degC)
    
    #print("DEBUG: f160 interp1", p.shape, T.shape, Td.shape)
    #print("DEBUG: f160 interp1", p, T, Td)
    fig = plt.figure(figsize=(9,9))
    skew = SkewT(fig,rotation=45)
    skew.plot(p,T,tcolor, linewidth=2)
    skew.plot(p,Td,tdcolor, linewidth=2)
    
    
    ## Add wind profile if desired
    if uwind is not None and vwind is not None and press_rho is not None:
        # interpolate to desired lat/lon
        u0 = uwind.interpolate([('longitude',[latlon[1]]),
                            ('latitude',[latlon[0]])],
                           iris.analysis.Linear())
        v0 = vwind.interpolate([('longitude',[latlon[1]]),
                            ('latitude',[latlon[0]])],
                           iris.analysis.Linear())
        p_rho0 = press_rho.interpolate([('longitude',[latlon[1]]),
                            ('latitude',[latlon[0]])],
                           iris.analysis.Linear())
        u = np.squeeze(u0.data.data) * units('m/s')#units(str(uwind.units))
        v = np.squeeze(v0.data.data) * units('m/s')#units(str(vwind.units))
        u = u.to(units.knots)
        v = v.to(units.knots)
        pro = (np.squeeze(p_rho0.data.data) * units(str(press_rho.units))).to(units.mbar)
        #print("DEBUG: f160 interp2", u.shape, v.shape, pro.shape)
        #print("DEBUG: f160 interp2", u,v,pro)
        nicer_z=np.union1d(np.union1d(np.arange(0,41,5), np.arange(43,81,3)), np.arange(81,140,1))
        #skip=(slice(None,None,None),nicer_z)
        skew.plot_barbs(pro[nicer_z],u[nicer_z],v[nicer_z])
        
    ## plot a bunch of nearby profiles
    if nearby>0:
        # find index nearest to lat/lon
        lati,loni = utils.lat_lon_index(latlon[0],latlon[1],lats.points,lons.points)
        
        for i in range(1,nearby+1,1):    
            for latii in [lati-i,lati, lati+i]:
                for lonii in [loni-i, loni, loni+i]:
                    # plot a range around the closest
                    p = np.squeeze(press[:,latii,lonii].data.data) * units(str(press.units))
                    T = np.squeeze(Temp[:,latii,lonii].data.data) * units(str(Temp.units))
                    T = T.to(units.degC)
                    Td = np.squeeze(Tempd[:,latii,lonii].data.data) * units(str(Tempd.units))
                    Td = Td.to(units.degC)
                    skew.plot(p,T,tcolor, linewidth=1, alpha=alpha)
                    skew.plot(p,Td,tdcolor, linewidth=1, alpha=alpha)

    # set limits
    skew.ax.set_ylim(1000,100)
    skew.ax.set_xlim(-30,60)
    # Add the relevant special lines
    skew.plot_dry_adiabats()
    skew.plot_moist_adiabats()
    skew.plot_mixing_lines()
    return skew

def f160_hour(dtime=datetime(2016,1,6,7), 
              latlon=plotting._latlons_['pyrocb_waroona1'],
              latlon_stamp='pyrocb1',
              model_version='waroona_run1',
              nearby=2):
    '''
    Look at F160 plots over time for a particular location
    INPUTS: hour of interest, latlon, and label to describe latlon (for plotting folder)
        nearby: how many gridpoints around the desired latlon to plot (can be 0)
    '''

    # Use datetime and latlon to determine what data to read
    extentname=model_version.split('_')[0]
    extent = plotting._extents_[extentname]
    
    if latlon_stamp is None:
        latlon_stamp="%.3fS_%.3fE"%(-latlon[0],latlon[1])
    
    # read pressure and temperature cubes
    cubes = fio.read_model_run(model_version, fdtime=dtime, extent=extent,
                               add_winds = True,
                               add_dewpoint = True,)
    p, t, td, u, v = cubes.extract(['air_pressure','air_temperature',
                                    'dewpoint_temperature','u','v'])
    p_rho = p
    if model_version=='waroona_run1':
        p_rho, = cubes.extract('air_pressure_rho')
    
    ffdtimes = utils.dates_from_iris(p)
    
    for i in range(len(ffdtimes)):
        # Plot name and title
        ptitle="SkewT$_{ACCESS}$   (%s) %s"%(latlon_stamp,ffdtimes[i].strftime("%Y %b %d %H:%M (UTC)"))
        
        # create plot
        f160(p[i],t[i],td[i], latlon,
             press_rho=p_rho[i], uwind=u[i], vwind=v[i])
        plt.title(ptitle)
        
        # save plot
        fio.save_fig(model_version,_sn_,ffdtimes[i],plt,subdir=latlon_stamp)

if __name__ == '__main__':
    
    if True:
        mr="KI_run1_exploratory"
        latlon=constants.latlons['Parndana']
        hours=utils.hours_available(mr)
        f160_hour(dtime=hours[10], 
                  latlon=latlon, 
                  latlon_stamp="Parndana",
                  model_version=mr,)
    
    if False:
        waroona_upwind = []
        waroona_0630_pcb = [-32.9,116.05] # latlon
        waroona_0630_pcb_stamp = "32.90S, 116.05E"
        if False:
            # show waroona f160 at PCB
            f160_hour(dtime=datetime(2016,1,6,6),
                      latlon=waroona_0630_pcb,
                      latlon_stamp=waroona_0630_pcb_stamp,
                      model_version='waroona_run3', 
                      nearby=1)
        
        
        if False: # old stuff
            topleft = [-32.75, 115.8] # random point away from the fire influence
            pyrocb1 = plotting._latlons_['pyrocb_waroona1']
            upwind  = plotting._latlons_['fire_waroona_upwind'] # ~1 km upwind of fire
            loc_and_stamp = ([pyrocb1,topleft,upwind],['pyrocb1','topleft','upwind'])
            #loc_and_stamp = ([upwind],['upwind'])
            #checktimes = [ datetime(2016,1,6,5) + timedelta(hours=x) for x in range(2) ]
            #checktimes = [ datetime(2016,1,5,15) ]
            old_times = fio.model_outputs['waroona_old']['filedates']
            run_times = fio.model_outputs['waroona_run1']['filedates']
            
            # 'waroona_run1','waroona_old'],[run_times,run_times,run_times,old_times]):
            for mv, dtimes in zip(['waroona_run2','waroona_run2uc'],[run_times,run_times]):
                for dtime in dtimes:
                    for latlon,latlon_stamp in zip(loc_and_stamp[0],loc_and_stamp[1]):
                        try:
                            f160_hour(dtime, latlon=latlon,
                                      latlon_stamp=latlon_stamp,
                                      model_version=mv)
                        except OSError as ose:
                            print("WARNING: probably no file for ",mv,dtime)
                            print("       :", ose)

