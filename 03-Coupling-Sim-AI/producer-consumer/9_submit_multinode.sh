#!/bin/bash -l
#PBS -A ALCFAITP
#PBS -l select=2
#PBS -N producer-consumer
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
export PATH=$PATH:/opt/pbs/bin

# Set inputs
NUM_SIMS=32
GRID_SIZE=512
echo "Running producer-consumer scripts"
echo "with $NUM_SIMS simulations of $GRID_SIZE x $GRID_SIZE grid"
echo

echo "Running with DragonHPC"
dragon 6_dragon_producer_consumer.py --grid_size $GRID_SIZE --num_sims $NUM_SIMS

echo "Running with Parsl writing to the file system"
python 7_parsl_fs_producer_consumer.py --grid_size $GRID_SIZE --num_sims $NUM_SIMS
    
echo "Running with Parsl tranfering data through futures"
python 8_parsl_fut_producer_consumer.py --grid_size $GRID_SIZE --num_sims $NUM_SIMS
