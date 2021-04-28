#!/bin/bash
#PBS -P en0
#PBS -q express
#PBS -N runstuff
#PBS -l walltime=16:00:00
#PBS -l mem=120000MB
#PBS -l cput=24:00:00
#PBS -l wd
#PBS -l ncpus=1
#PBS -l storage=scratch/en0+gdata/en0+gdata/hh5
#PBS -l software=Python
#PBS -j oe

#---------------------------------
# send to queue with 
# qsub -o log.qsub run.sh
# --------------------------------

# Check if script is called directly, or with qsub -v mr="..."
# if direct, needs an argument
if [ $# -lt 1 ] && [ -z ${mr} ]; then
    echo "EG usage: bash ${0} KI_run3"
    echo "    send some code to qsub for model output linked by ../data/KI_run3"
    exit 0
fi


# if called directly, send to queue with mr input variable set
if [ -z ${PBS_O_LOGNAME} ] || [ -z ${mr} ]; then
    
    echo "Sending some code to queue with mr=${1}"
    echo "qsub -v mr=${1} -N ${1} ${0}"
    
    # can split script into phases if ram limit reached
    qsub -v mr=${1},phase=1 -N ${1}_A ${0}
    qsub -v mr=${1},phase=2 -N ${1}_B ${0}
    qsub -v mr=${1},phase=3 -N ${1}_C ${0}

    # end here if called directly
    exit 0
fi

module use /g/data3/hh5/public/modules
module load conda/analysis3

python <<EOF


## local scripts that can be run
from cross_sections import topdown_view_only, multiple_transects, multiple_transects_SN
from weather_summary import weather_summary_model
from fire_spread import fire_spread, isochrones
from winds import rotation_looped, wind_and_heat_flux_looped

# METHODS NEED TO HAVE 1st 3 arguments: run name, WESN, subdir
fnlist_A = [
    isochrones,
    wind_and_heat_flux_looped, 
    topdown_view_only,
    fire_spread, # this one took a while until I started skipping some times
    weather_summary_model, # High resource
    ] # Uses 4.5 HOURS, 18 GB (50% spatial constraint)
fnlist_B = [
    multiple_transects, # high resource
    ] # 3.6 hours, 15 GB, (50% spatial constraint)
fnlist_C = [
    rotation_looped, 
    multiple_transects_SN, # ~1hrs, 37GB with 50% horizontal subsetting
    ]

## keep track of used zooms
KI_zoom = [136.5,137.5,-36.1,-35.6]
KI_zoom_name = "zoom1"
KI_zoom2 = [136.5887,136.9122,-36.047,-35.7371]
KI_zoom2_name = "zoom2"
badja_zoom=[149.4,150.0, -36.4, -35.99]
badja_zoom_name="zoom1"

## settings for plots
mr="${mr}"
zoom = None
subdir = None

if 'badja' in mr:
    zoom = badja_zoom
    subdir=badja_zoom_name
elif 'KI_run' in mr:
    zoom = KI_zoom
    subdir=KI_zoom_name
# KI_eve zoom?

print("INFO: running methods for [mr, zoom, subdir] :",[mr, zoom, subdir])
## Argument list for multiple functions

if ${phase}==1:
    for func in fnlist_A:
        print("INFO: Calling ",func) 
        func(mr,zoom,subdir)
elif ${phase}==2:
    for func in fnlist_B:
        print("INFO: Calling ",func) 
        func(mr,zoom,subdir)
elif ${phase}==3:
    for func in fnlist_C:
        print("INFO: Calling ",func) 
        func(mr,zoom,subdir)
        

## doing it in parallel Failed, and I don't understand the error messages...
#import dask.bag as db
#bag = db.from_sequence([mr,zoom,subdir]) 
## list of functions to be mapped to processors
#bag = bag.map(fnlist)
## run functions over whatever available processors
#bag.compute() 

EOF


