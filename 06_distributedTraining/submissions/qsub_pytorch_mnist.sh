#!/bin/bash
#COBALT -n 1
#COBALT -t 15 -q full-node
#COBALT -A ALCFAITP -O results/$jobid.pytorch_mnist

#submisstion script for running tensorflow_mnist with horovod

echo "Running Cobalt Job $COBALT_JOBID."

#Loading modules
. /etc/profile.d/z00_lmod.sh
module load conda
conda activate


for n in 1 2 4 8
do
    mpirun -np $n python pytorch_mnist.py --device gpu --epochs 32 >& results/pytorch_mnist.n$n.out
done




