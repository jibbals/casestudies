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
methods="isochrones fireseries vorticity vorticity_10m plume wind_dir_10m fire_spread weather_summary_model multiple_transects multiple_transects_SN"

#some runs have multiple interesting extents
extent_inds="0"
if [[ ${1}  == *"KI"* ]]; then
    extent_inds="0 1 2 3"
elif [[ ${1} == *"badja"* ]]; then
    extent_inds="0 1 2"
fi


# if called directly, send to queue with mr input variable set
if [ -z ${PBS_O_LOGNAME} ] || [ -z ${mr} ]; then
    
    for method in $methods; do
        echo "run method ${method}?"
        select yn in "Yes" "No"; do
            case $yn in
                Yes ) echo "qsub -v mr=${1},method=${method},extent_ind=N -N ${1}_${method} ${0}";
                    for ex_ind in $extent_inds; do
                        qsub -v mr=${1},method=${method},extent_ind=${ex_ind} -N ${1}_${method} ${0};
                    done
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
from timeseries_stuff import fireseries
from plume import plume
from vorticity import vorticity_10m, vorticity

### keep track of used zooms
KI_zooms = [None,
            [136.5,   137.5,   -36.1,   -35.6],
            [136.5887,136.9122,-36.047,-35.7371],
            [136.6,136.9,-36.08,-35.79],
           ]
KI_zoom_names = None,"zoom1","zoom2","early_run"
badja_zooms=[[149.4,     150.12,  -36.47,  -35.99],
             [149.5843,  149.88, -36.376, -36.223],
             [149.5308,149.9093,-36.2862,-36.0893]]
badja_zoom_names="zoom1","Wandella","Belowra"

## settings for plots
mr="${mr}"
method=${method}
zooms = [None]
subdirs = [None]

if 'badja' in mr:
    zooms = badja_zooms
    subdirs = badja_zoom_names
elif 'KI_' in mr:
    zooms = KI_zooms
    subdirs = KI_zoom_names

zoom = zooms[${extent_ind}]
subdir = subdirs[${extent_ind}]

print("INFO: running [mr, zoom, subdir] :",[mr, zoom, subdir])
print("INFO: running method:",method)

method(mr,zoom,subdir)

print("INFO: DONE ",method)


EOF


