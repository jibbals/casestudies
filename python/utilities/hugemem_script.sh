#!/bin/bash
#PBS -P en0
#PBS -q hugemem
#PBS -N casestudies_py
#PBS -l walltime=10:00:00
#PBS -l mem=500000MB
#PBS -l cput=10:00:00
#PBS -l wd
#PBS -l ncpus=1
#PBS -l storage=scratch/en0+gdata/en0+gdata/hh5
#PBS -l software=Python
#PBS -j oe

#---------------------------------
# send to queue with 
# qsub -o log.qsub run.sh
# --------------------------------
# #PBS -l jobfs=10MB # maybe not needed
if [ -z ${PBS_O_LOGNAME} ] || [ -z ${script} ]; then
    echo "EG usage: qsub -v script=make_pft_file.py ${0}"
    echo "    To run make_pft_file.py"
    exit 0
fi


module use /g/data3/hh5/public/modules
module load conda/analysis3


python ${script}

echo "Done with ${0}: ${PBS_O_LOGNAME}"



