#!/bin/bash
#COBALT -n 1
#COBALT -t 15 -q full-node
#COBALT -A ALCFAITP -O results/$jobid.keras_mnist

#submisstion script for running tensorflow_mnist with horovod

echo "Running Cobalt Job $COBALT_JOBID."

#Loading modules

. /etc/profile.d/z00_lmod.sh
module load conda
conda activate

for n in 1 2 4 8
do
    mpirun -np $n python keras_mnist.py --device gpu --epochs 32 >& results/keras_mnist.out.$n 
done



