#!/bin/bash -l
#COBALT -t 20
#COBALT -q training-gpu
#COBALT -n 1
#COBALT -A ALCFAITP
#COBALT --attrs filesystems=home,grand

# Set up software deps:
module load conda/2022-07-01
conda activate

# You have to point this to YOUR local copy of ai-science-training-series

export TF_XLA_FLAGS="--tf_xla_auto_jit=2"
mpirun -np 1 python train_resnet34_hvd.py --num_steps 10
mpirun -np 2 python train_resnet34_hvd.py --num_steps 10
mpirun -np 4 python train_resnet34_hvd.py --num_steps 10
mpirun -np 8 python train_resnet34_hvd.py --num_steps 10
