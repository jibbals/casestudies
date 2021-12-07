#!/bin/bash
#PBS -P en0
#PBS -q normal
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

#some runs have multiple interesting extents
extent_inds="0"
if [[ ${1}  == *"KI"* ]]; then
    extent_inds="0 1 2"
elif [[ ${1} == *"badja"* ]]; then
    extent_inds="0 1 2"
fi

## scripts with a "suitecall" method:
scripts=$(grep -l *.py -e "suitecall(")

# if called directly, send to queue with mr input variable set
if [ -z ${PBS_O_LOGNAME} ] || [ -z ${mr} ]; then
    # loop over lines in scripts to be run:
    for script in $scripts; do
        echo "script: ...${script%%.*}..."
        
        echo "run $script?"
        select yn in "Yes" "No"; do
            case $yn in 
                Yes ) echo "qsub -v mr=${1},script=${script} -N ${1}_${script} ${0}";
                    # send this script to queue with extra variables script, mr, (eventually west,east,south,north)
                    for ex_ind in $extent_inds; do
                        qsub -v mr=${1},script=${script},extent_ind=${ex_ind} -N ${1}_${method} ${0};
                    done
                    break;;
                No ) break;;
            esac
        done
    done
    # end here if called directly
    exit 0
fi

#echo "end of test"
#exit 0


##
## From here is run within compute node
##

module use /g/data3/hh5/public/modules
module load conda/analysis3-21.04

python <<EOF

## local scripts that can be run
# METHODS NEED TO HAVE 1st 3 arguments: run name, WESN, subdir
from ${script%%.*} import suitecall
from utilities import constants

### keep track of used zooms
KI_zoom_names=list(constants.extents['KI'].keys())
KI_zooms = list(constants.extents['KI'].values())
badja_zoom_names=list(constants.extents['badja'].keys())
badja_zooms= list(constants.extents['badja'].values())
badja_am_zoom_names=list(constants.extents['badja_am'].keys())
badja_am_zooms= list(constants.extents['badja_am'].values())

## settings for plots
mr="${mr}"
zooms = [None]
subdirs = [None]

if 'badja_am' in mr:
    zooms = badja_am_zooms
    subdirs = badja_am_zoom_names
elif 'badja' in mr:
    zooms = badja_zooms
    subdirs = badja_zoom_names
elif 'KI_' in mr:
    zooms = KI_zooms
    subdirs = KI_zoom_names

zoom = zooms[${extent_ind}]
subdir = subdirs[${extent_ind}]

print("INFO: running $script on $mr: [zoom, subdir] =",[zoom, subdir])

suitecall(mr,zoom,subdir)

print("INFO: DONE $script")


EOF


