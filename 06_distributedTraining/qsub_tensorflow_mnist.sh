#!/bin/bash
#COBALT -n 1
#COBALT -t 15 -q full-node
#COBALT -A ALCFAITP -O results/$jobid.tensorflow2_mnist

#submisstion script for running tensorflow_mnist with horovod

echo "Running Cobalt Job $COBALT_JOBID."

#Loading modules

. /etc/profile.d/z00_lmod.sh
module load conda
conda activate

for n in 1 2 4 8
do
    rm -rf checkpoints
    mpirun -np $n python tensorflow2_mnist.py --epochs 16 >& results/tensorflow2_mnist.n$n.out
done

LD_PRELOAD=/soft/perftools/hpctw/lib/libmpitrace.so HOROVOD_TIMELINE=timeline.json mpirun -np $n python tensorflow2_mnist.py --epochs 16 
