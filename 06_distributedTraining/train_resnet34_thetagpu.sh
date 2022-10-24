#!/bin/bash -l
#COBALT -t 60
#COBALT -q full-node
#COBALT -n 1
#COBALT -A datascience
##COBALT -A ALCFAITP

#  #########3#COBALT --attrs filesystems=home:eagle

# Set up software deps:
module load conda/2022-07-01
conda activate

# You have to point this to YOUR local copy of ai-science-training-series

export TF_XLA_FLAGS="--tf_xla_auto_jit=2"
mpirun -np 8 python train_resnet34_hvd.py --num_steps 1000000
