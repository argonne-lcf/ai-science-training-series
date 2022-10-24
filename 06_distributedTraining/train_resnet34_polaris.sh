#!/bin/sh
#PBS -l select=8:system=polaris
#PBS -l place=scatter
#PBS -l walltime=1:00:00
#PBS -q debug-scaling
#PBS -A datascience
#PBS -l filesystems=home:eagle


# Set up software deps:
module load conda/2022-09-08
conda activate

cd $PBS_O_WORKDIR

export TF_XLA_FLAGS="--tf_xla_auto_jit=2"
aprun -n 32 -N 4 python train_resnet34_hvd.py --num_steps 100000
