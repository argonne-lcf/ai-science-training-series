#!/bin/sh
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=1:00:00
#PBS -q debug
#PBS -A datascience
#PBS -l filesystems=home:eagle


# Set up software deps:
module load conda/2022-09-08
conda activate

cd /home/cadams/Polaris/ai-science-training-series/04_modern_neural_networks

export TF_XLA_FLAGS="--tf_xla_auto_jit=2"
python train_resnet34.py
