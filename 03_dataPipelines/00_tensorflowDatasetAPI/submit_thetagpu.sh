#!/bin/bash -l
#COBALT -t 10
#COBALT -n 1
#COBALT -O logdir/$COBALT_JOBID

echo [$SECONDS] setup conda environment
module load conda/2021-09-22
conda activate

echo [$SECONDS] python = $(which python)
echo [$SECONDS] python version = $(python --version)

echo [$SECONDS] setup local env vars
NODES=`cat $COBALT_NODEFILE | wc -l`
RANKS_PER_NODE=1
RANKS=$((NODES * RANKS_PER_NODE))
echo [$SECONDS] NODES=$NODES  RANKS_PER_NODE=$RANKS_PER_NODE  RANKS=$RANKS

export OMP_NUM_THREADS=64

echo [$SECONDS] run parallel example
python ilsvrc_dataset.py -c ilsvrc.json --interop $OMP_NUM_THREADS --intraop $OMP_NUM_THREADS \
   --logdir logdir/$COBALT_JOBID

echo [$SECONDS] run serial example
python ilsvrc_dataset_serial.py -c ilsvrc.json --logdir logdir/${COBALT_JOBID}-serial
   
echo [$SECONDS] done
