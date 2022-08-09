#!/bin/bash -l
#COBALT -q single-gpu
#COBALT -t 10
#COBALT -n 1


module load conda/2021-09-22
conda activate

python 03_pytorch_mnist.py

