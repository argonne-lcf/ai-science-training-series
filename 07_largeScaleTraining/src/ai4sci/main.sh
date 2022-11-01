#!/bin/bash
# -------------------------------------------------------
# UG Section 2.5, page UG-24 Job Submission Options
# Add another # at the beginning of the line to comment out a line
# NOTE: adding a switch to the command line will override values in this file.

# Highly recommended 
# The first 15 characters of the job name are displayed in the qstat output:
#PBS -N <name>

# If you need a queue other than the default (uncomment to use)
##PBS -q <queue name>
# Controlling the output of your application
# UG Sec 3.3 page UG-40 Managing Output and Error Files
# By default, PBS spools your output on the compute node and then uses scp to move it the
# destination directory after the job finishes.  Since we have globally mounted file systems
# it is highly recommended that you use the -k option to write directly to the destination
# the doe stands for direct, output, error
#PBS -o <path for stdout>
#PBS -k doe
#PBS -e <path for stderr>

# Environment variables (uncomment to use)
# Section 6.12, page UG-126 Using Environment Variables
# Sect 2.59.7, page RG-231 Enviornment variables PBS puts in the job environment
##PBS -v <variable list>
## -v a=10, "var2='A,B'", c=20, HOME=/home/zzz
##PBS -V exports all the environment variables in your environnment to the compute node
# The rest is an example of how an MPI job might be set up
# echo Working directory is $PBS_O_WORKDIR
# cd $PBS_O_WORKDIR
#
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ NOTE: Recommended for running this file    ┃
# ┃--------------------------------------------┃ 
# ┃$ ./main.sh $@ > main.log 2>&1 &            ┃
# ┃$ tail -1 main.log $(tail -1 logs/latest)   ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

TSTAMP=$(date "+%Y-%m-%d-%H%M%S")
HOST=$(hostname)
echo "Job ID: ${PBS_JOBID}"
echo "Job started at: ${TSTAMP}"

# ---- Specify directories and executable for experiment ------------------
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -LP)
echo "DIR:$DIR"
MAIN="${DIR}/main.py"
PARENT=$(dirname $DIR)
ROOT=$(dirname $PARENT)

echo "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓"
echo "┃  Job started at: ${TSTAMP} on ${HOST}           ┃"
echo "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛"

NCPUS=$(getconf _NPROCESSORS_ONLN)

# ---- Check if running on ThetaGPU ----------------------------
if [[ $(hostname) == theta* ]]; then
  module load conda/2022-07-01
  conda activate base
  NRANKS=$(wc -l < ${COBALT_NODEFILE})
  HOSTFILE=${COBALT_NODEFILE}
  NGPU_PER_RANK=$(nvidia-smi -L | wc -l)
  NGPUS=$((${NRANKS}*${NGPU_PER_RANK}))
  MPI_COMMAND=$(which mpirun)
  MPI_FLAGS="--verbose \
    -n ${NGPUS} \
    --hostfile ${HOSTFILE} \
    -npernode ${NGPU_PER_RANK} \
    -x PYTHONUSERBASE \
    -x PYTHONSTARTUP \
    -x http_proxy \
    -x https_proxy \
    -x PATH \
    -x LD_LIBRARY_PATH"
  VENV_DIR="${ROOT}/venvs/thetaGPU/2022-07-01"

# ---- Check if running on Polaris -----------------------------
elif [[ $(hostname) == x* ]]; then
  # echo Working directory is $PBS_O_WORKDIR
  # cd $PBS_O_WORKDIR
  export IBV_FORK_SAFE=1
  export NCCL_COLLNET_ENABLE=1
  NRANKS=$(wc -l < ${PBS_NODEFILE})
  HOSTFILE=${PBS_NODEFILE}
  NGPU_PER_RANK=$(nvidia-smi -L | wc -l)
  NGPUS=$((${NRANKS}*${NGPU_PER_RANK}))
  MPI_COMMAND=$(which mpiexec)
  # --cpu-bind verbose,list:0,8,16,24 \
  MPI_FLAGS="--verbose \
    --envall \
    -n ${NGPUS} \
    --depth ${NCPUS} \
    --ppn ${NGPU_PER_RANK} \
    --hostfile ${HOSTFILE}"
  module load conda/2022-09-08
  conda activate base
  VENV_DIR="${ROOT}/venvs/polaris/2022-09-08"

# ---- Check if running on MacOS --------------------------------
else
  VENV_DIR="${ROOT}/venv/"
  if [[ $(uname) == Darwin* ]]; then
    # ---- Check if environment has an mpirun executable ----------
    if [[ -x $(which mpirun) ]]; then
      MPI_COMMAND=$(which mpirun)
      MPI_FLAGS="-np ${NCPUS}"
    fi
  # ---- Otherwise, run without MPI -------------------------------
  else
      MPI_COMMAND=""
      MPI_FLAGS=""
      echo "HOSTNAME: $(hostname)"
  fi
fi


# -----------------------------------------------------------
# 1. Check if a virtual environment exists in project root: 
# 2. If so, activate environment and make sure we have an 
#    editable install
# -----------------------------------------------------------
if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  echo "Found venv at: ${VENV_DIR}"
  source "${VENV_DIR}/bin/activate"
  python3 -m pip install --upgrade pip
  python3 -m pip install -e "${ROOT}" --no-deps
else
  if [[ -f "${ROOT}/venv/bin/activate" ]]; then
    echo "Found venv at: ${ROOT}/venv/, using that"
    source "${VENV_DIR}/bin/activate"
    python3 -m pip install --upgrade pip
    python3 -m pip install -e "${ROOT}" --no-deps
  else
    echo "Creating new venv at: ${VENV_DIR}"
    python3 -m venv "${ROOT}/venv/" --system-site-packages
    python3 -m pip install --upgrade pip
    source "${VENV_DIR}/bin/activate"
    python3 -m pip install -e "${ROOT}" --no-deps
  fi
fi

# ---- Environment settings -----------------------------------------------
export OMP_NUM_THREADS=$NCPUS
export WIDTH=$COLUMNS
export COLUMNS=$COLUMNS
echo "WIDTH: ${COLUMNS}"
export NCCL_DEBUG=INFO
export KMP_SETTINGS=TRUE
LOGDIR="${DIR}/logs"
LOGFILE="${LOGDIR}/${TSTAMP}-${HOST}_ngpu${NGPUS}_ncpu${NCPUS}.log"
if [ ! -d "${LOGDIR}" ]; then
  mkdir -p ${LOGDIR}
fi


# Keep track of latest logfile for easy access
echo $LOGFILE >> "${DIR}/logs/latest"

# Double check everythings in the right spot
echo "DIR=${DIR}"
echo "MAIN=${MAIN}"
echo "PARENT=${PARENT}"
echo "ROOT=${ROOT}"
echo "LOGDIR=${LOGDIR}"
echo "LOGFILE=${LOGFILE}"
echo "IBV_FORK_SAFE=${IBV_FORK_SAFE}"
printf '%.s─' $(seq 1 $(tput cols))

EXEC="${MPI_COMMAND} ${MPI_FLAGS} $(which python3) ${MAIN}"


# ---- Print job information -------------------------------------------------
echo "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓"
echo "┃  STARTING A NEW RUN ON ${NGPUS} GPUs ${NCPUS}  ┃"
echo "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛"
echo -e '\n'
printf '%.s─' $(seq 1 $(tput cols))
echo "┃  - DATE: ${TSTAMP}"
echo "┃  - NCPUS: ${NCPUS}"
echo "┃  - NRANKS: ${NRANKS}"
echo "┃  - NGPUS PER RANK: ${NGPU_PER_RANK}"
echo "┃  - NGPUS TOTAL: ${NGPUS}"
echo "┃  - MAIN: ${MAIN}"
echo "┃  - Writing logs to ${LOGFILE}"
echo "┃  - python3: $(which python3)"
echo "┃  - mpirun: ${MPI_COMMAND}"
echo "┃  - ai4sci: $(python3 -c 'import ai4sci; print(ai4sci.__file__)')"
echo "┃  - exec: ${EXEC} $@"
printf '%.s─' $(seq 1 $(tput cols))
echo -e '\n'


# Run executable command
WIDTH=$COLUMNS COLUMNS=$COLUMNS ${EXEC} $@ 2>&1 > ${LOGFILE}; ret_code=$?

if [[ $ret_code != 0 ]]; then exit $ret_code; fi

wait
