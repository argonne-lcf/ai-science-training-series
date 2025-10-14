#! /bin/bash -x
#
# Time Stamp
tstamp() {
     date +"%Y-%m-%d-%H%M%S"
}

NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=2

let NRANKS=${NNODES}*${NRANKS_PER_NODE}

N=2
PPN=2
EPOCHS=10

NODES=1

TRACE_DIR_ROOT=./traces/pytorch_2p8
TRACE_DIR=${TRACE_DIR_ROOT}/cuda_pt_2p8_compile_E${EPOCHS}_N${NODES}_R${PPN}_$(tstamp)

module use /soft/modulefiles
module load conda/2025-09-25
conda activate

export DISABLE_PYMODULE_LOG=1

export CPU_AFFINITY="verbose,list:0,1:8,9:16,17:24,25"

#mpiexec -n ${NRANKS} -ppn ${NRANKS_PER_NODE} -l --line-buffer --cpu-bind ${CPU_AFFINITY} python pytorch_2p8_ddp_prof.py

mpiexec -n ${N} -ppn ${PPN} -l --line-buffer --cpu-bind ${CPU_AFFINITY} python pytorch_2p8_ddp_compile_prof.py \
    --epochs ${EPOCHS} --trace-dir ${TRACE_DIR}
