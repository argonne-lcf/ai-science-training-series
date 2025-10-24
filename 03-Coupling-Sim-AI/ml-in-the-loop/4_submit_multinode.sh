#!/bin/bash -l
#PBS -A ALCFAITP
#PBS -l select=2
#PBS -N ml_in_the_loop
#PBS -l walltime=0:30:00
#PBS -l filesystems=home:eagle
#PBS -k doe
#PBS -j oe
#PBS -l place=scatter
#PBS -q debug

cd $PBS_O_WORKDIR

module use /soft/modulefiles
module load conda/2025-09-25
conda activate /eagle/datascience/balin/AI4S/_ai4s_simAI
export TMPDIR=/tmp

echo "Running parsl ml_in_the_loop.py script"
python ./3_ml_in_the_loop.py
