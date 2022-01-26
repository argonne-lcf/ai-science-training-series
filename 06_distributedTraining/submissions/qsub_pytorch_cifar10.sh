#!/bin/bash
#COBALT -n 2
#COBALT -t 1:00:00 -q training-gpu
#COBALT -A SDL_Workshop -O results/thetagpu/$jobid.pytorch_cifar10

#submisstion script for running tensorflow_mnist with horovod

echo "Running Cobalt Job $COBALT_JOBID."

#Loading modules

. /etc/profile.d/z00_lmod.sh
module load conda
conda activate

COBALT_JOBSIZE=$(cat $COBALT_NODEFILE | wc -l)

if (( $COBALT_JOBSIZE > 1))
then
    mpirun -x LD_LIBRARY_PATH -x PATH -x PYTHONPATH -np $((COBALT_JOBSIZE*8)) -npernode 8 --hostfile $COBALT_NODEFILE python pytorch_cifar10.py --device gpu --epochs 32 >& results/thetagpu/pytorch_cifar10.n$((COBALT_JOBSIZE*8)).out
else
    for n in 1 2 4 8
    do
	mpirun -np $n python pytorch_cifar10.py --device gpu --epochs 32 >& results/thetagpu/pytorch_cifar10.n$n.out
    done
fi

