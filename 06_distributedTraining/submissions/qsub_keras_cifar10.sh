#!/bin/bash
#COBALT -n 2
#COBALT -t 1:00:00 -q training-gpu
#COBALT -A SDL_Workshop -O results/thetagpu/$jobid.keras_cifar10

#submisstion script for running tensorflow_mnist with horovod

echo "Running Cobalt Job $COBALT_JOBID."

#Loading modules
. /etc/profile.d/z00_lmod.sh
module load conda
conda activate

COBALT_JOBSIZE=$(cat $COBALT_NODEFILE | wc -l)
if (( $COBALT_JOBSIZE > 1 ))
then
    mpirun -x LD_LIBRARY_PATH -x PATH -x PYTHONPATH -np $((COBALT_JOBSIZE*8)) -npernode 8 --hostfile ${COBALT_NODEFILE} python tensorflow2_keras_cifar10.py --device gpu --epochs 32 >& results/thetagpu/tensorflow2_keras_cifar10.out.$((COBALT_JOBSIZE*8))
else
    for n in 1 2 4 8
    do
	mpirun -np $n $(which python) tensorflow2_keras_cifar10.py --device gpu --epochs 32 >& results/thetagpu/tensorflow2_keras_cifar10.out.$n
    done
fi

