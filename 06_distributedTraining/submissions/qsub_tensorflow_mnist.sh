#!/bin/bash
#COBALT -n 2
#COBALT -t 1:00:00 -q training-gpu
#COBALT -A SDL_Workshop -O results/thetagpu/$jobid.tensorflow2_mnist

#submisstion script for running tensorflow_mnist with horovod

echo "Running Cobalt Job $COBALT_JOBID."

#Loading modules

. /etc/profile.d/z00_lmod.sh
module load conda
conda activate

COBALT_JOBSIZE=$(cat $COBALT_NODEFILE | wc -l)

echo "Running job on ${COBALT_JOBSIZE} nodes"
ng=$((COBALT_JOBSIZE*8))
if (( ${COBALT_JOBSIZE} > 1 ))
then
    # multiple nodes
    mpirun -x LD_LIBRARY_PATH -x PATH -x PYTHONPATH -np $ng -npernode 8 --hostfile $COBALT_NODEFILE python tensorflow2_mnist.py --device gpu --epochs 32 >& results/thetagpu/tensorflow2_mnist.n$ng.out
else
    # Single node
    for n in 1 2 4 8
    do
	mpirun -np $n python tensorflow2_mnist.py --device gpu --epochs 32 >& results/thetagpu/tensorflow2_mnist.n$n.out
    done
fi
