#!/bin/bash

module use /soft/modulefiles/
module load conda/2025-09-25
conda activate

conda activate /lus/eagle/projects/ALCFAITP/04-Inference-Workflows/env/_ai4s_agentic_conda

export http_proxy="proxy.alcf.anl.gov:3128"
export https_proxy="proxy.alcf.anl.gov:3128"
