#!/bin/bash

module use /soft/modulefiles
module load conda/2025-09-25
conda activate /eagle/ALCFAITP/03-Coupling-Sim-AI/_ai4s_simAI
export TMPDIR=/tmp
export PATH=$PATH:/opt/pbs/bin

