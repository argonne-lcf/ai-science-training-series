#!/bin/bash -l
#PBS -q by-gpu
#PBS -l walltime=0:10:0
#PBS -l select=1
#PBS -l filesystems=eagle:home_fs

# meant for running on Sophia

module load conda
conda activate

python 12_tensorflow_mnist.py

