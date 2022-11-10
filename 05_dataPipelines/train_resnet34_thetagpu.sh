#!/bin/bash -l
#COBALT -t 60
#COBALT -q single-gpu
#COBALT -A ALCFAITP
#COBALT -n 1
#COBALT --attrs filesystems=home,grand


# Set up software deps:
module load conda/2022-07-01
conda activate

# You have to point this to YOUR local copy of ai-science-training-series
cd /home/jugacostase/ThetaGPU/ai-science-training-series/05_dataPipelines

export TF_XLA_FLAGS="--tf_xla_auto_jit=2"
python train_resnet34.py 1 1
python train_resnet34.py 2 1
python train_resnet34.py 4 1
python train_resnet34.py 4 2
python train_resnet34.py 4 4
python train_resnet34.py 8 1
python train_resnet34.py 8 2
python train_resnet34.py 8 4
python train_resnet34.py 8 8
python train_resnet34.py 16 1
python train_resnet34.py 16 2
python train_resnet34.py 16 4
python train_resnet34.py 16 8
python train_resnet34.py 16 16
python train_resnet34.py 1 0
python train_resnet34.py 2 0
python train_resnet34.py 4 0
python train_resnet34.py 8 0
python train_resnet34.py 16 0
python train_resnet34.py 32 0
python train_resnet34.py 64 0
python train_resnet34.py 128 0
python train_resnet34.py 256 0



