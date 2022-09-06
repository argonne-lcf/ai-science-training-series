#!/bin/bash -l
#COBALT -q single-gpu
#COBALT -t 10
#COBALT -n 1
#COBALT --attrs filesystems=home,theta-fs0


module load conda/2022-07-01
conda activate

python 12_pytorch_mnist.py

