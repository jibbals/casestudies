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

## List of methods to call
## METHODS NEED TO HAVE 1st 3 arguments: run name, WESN, subdir
methods="isochrones plot_fireseries plume wind_dir_10m wind_and_heat_flux_looped fire_spread weather_summary_model multiple_transects multiple_transects_SN"

# if called directly, send to queue with mr input variable set
if [ -z ${PBS_O_LOGNAME} ] || [ -z ${mr} ]; then
    
    for method in $methods; do
        echo "run method ${method}?"
        select yn in "Yes" "No"; do
            case $yn in
                Yes ) echo "qsub -v mr=${1},method=${method} -N ${1}_${method} ${0}"; 
                    qsub -v mr=${1},method=${method} -N ${1}_${method} ${0};
                    break;;
                No ) break;;
            esac
        done
    done

    # end here if called directly
    exit 0
fi

module use /g/data3/hh5/public/modules
module load conda/analysis3

python <<EOF

## local scripts that can be run
## NEED TO IMPORT METHOD MATCHING NAME IN METHODS LIST
# METHODS NEED TO HAVE 1st 3 arguments: run name, WESN, subdir
from cross_sections import topdown_view_only, multiple_transects, multiple_transects_SN
from weather_summary import weather_summary_model
from fire_spread import fire_spread, isochrones
from winds import rotation_looped, wind_and_heat_flux_looped
from wind_dir import wind_dir_10m
from timeseries_stuff import plot_fireseries
from plume import plume


### keep track of used zooms
KI_zooms = [[136.5,   137.5,   -36.1,   -35.6],
            [136.5887,136.9122,-36.047,-35.7371]]
KI_zoom_names = "zoom1","zoom2"
badja_zooms=[[149.4,   150.0,   -36.4,   -35.99],
             [149.5843,149.88,  -36.376, -36.223],
             [149.5308,149.9093,-36.2862,-36.0893]]
badja_zoom_names="zoom1","Wandella","Belowra"

## settings for plots
mr="${mr}"
method=${method}
zoom = None
subdir = None

if 'badja' in mr:
    zoom = badja_zooms[2]
    subdir=badja_zoom_names[2]
elif 'KI_run' in mr:
    zoom = KI_zooms[0]
    subdir = KI_zoom_names[0]
elif 'KI_eve' in mr:
    zoom = KI_zooms[0]
    subdir=KI_zoom_names[0]


print("INFO: running [mr, zoom, subdir] :",[mr, zoom, subdir])
print("INFO: running method:",method)

method(mr,zoom,subdir)

print("INFO: DONE ",method)


EOF


