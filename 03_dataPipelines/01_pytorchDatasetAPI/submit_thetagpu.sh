#!/bin/bash -l
#COBALT -A SDL_Workshop_1
#COBALT -t 10
#COBALT -n 1
#COBALT -q single-gpu
#COBALT -O logdir/$COBALT_JOBID

echo [$SECONDS] setup conda environment
module load conda/2021-09-22
conda activate

echo [$SECONDS] python = $(which python)
echo [$SECONDS] python version = $(python --version)

echo [$SECONDS] setup local env vars
NODES=`cat $COBALT_NODEFILE | wc -l`
PPN=1
RANKS=$((NODES * PPN))
echo [$SECONDS] NODES=$NODES  PPN=$PPN  RANKS=$RANKS

echo [$SECONDS] run example
python ilsvrc_dataset.py -c ilsvrc.json \
   --logdir logdir/$COBALT_JOBID

echo [$SECONDS] done