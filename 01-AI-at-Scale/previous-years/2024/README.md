# 🚀 Parallel Training Methods for AI

[Sam Foreman](https://samforeman.me)  
[Intro to AI-driven Science on Supercomputers](https://www.alcf.anl.gov/alcf-ai-science-training-series)  
_2024-11-05_

- Slides: <https://samforeman.me/talks/ai-for-science-2024/slides>
  - HTML version: <https://samforeman.me/talks/ai-for-science-2024>

## 👋 Hands On

1. Submit interactive job:

    ```bash
    qsub -A ALCFAITP -q by-node -l select=1 -l walltime=01:00:00,filesystems=eagle:home -I
    ```

1. On Sophia:

    ```bash
    export HTTP_PROXY="http://proxy.alcf.anl.gov:3128"
    export HTTPS_PROXY="http://proxy.alcf.anl.gov:3128"
    export http_proxy="http://proxy.alcf.anl.gov:3128"
    export https_proxy="http://proxy.alcf.anl.gov:3128"
    export ftp_proxy="http://proxy.alcf.anl.gov:3128"
    ```

1. Clone repos:

    1. [`saforem2/wordplay`](https://github.com/saforem2/wordplay):

        ```bash
        git clone https://github.com/saforem2/wordplay
        cd wordplay
        ```

    1. [`saforem2/ezpz`](https://github.com/saforem2/ezpz):

        ```bash
        git clone https://github.com/saforem2/ezpz deps/ezpz
        ```

1. Setup python:

    ```bash
    export PBS_O_WORKDIR=$(pwd) && source deps/ezpz/src/ezpz/bin/utils.sh
    ezpz_setup_python
    ezpz_setup_job
    ```

1. Install `{ezpz, wordplay}`:

    ```bash
    python3 -m pip install -e deps/ezpz --require-virtualenv
    python3 -m pip install -e . --require-virtualenv
    ```

1. Setup (or disable) [`wandb`](https://wandb.ai):

    ```bash
    # to setup:
    wandb login
    # to disable:
    export WANDB_DISABLED=1
    ```

1. Test Distributed Setup:

    ```bash
    mpirun -n "${NGPUS}" python3 -m ezpz.test_dist
    ```

    See: [`ezpz/test_dist.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/test_dist.py)

1. Prepare Data:

    ```bash
    python3 data/shakespeare_char/prepare.py
    ```

1. Launch Training:

    ```bash
    mpirun -n "${NGPUS}" python3 -m wordplay \
        train.backend=DDP \
        train.eval_interval=100 \
        data=shakespeare \
        train.dtype=bf16 \
        model.batch_size=64 \
        model.block_size=1024 \
        train.max_iters=1000 \
        train.log_interval=10 \
        train.compile=false
    ```

## 🎒 Homework

Submit _proof_ that you were able to successfully follow the above instructions and launch a distributed data parallel training run.

Where _proof_ can be any of:

- The contents printed out to your terminal during the run
- A path to a logfile containing the output from a run on the ALCF filesystems
- A screenshot of:
  - the text printed out from the run
  - a graph from the W&B Run
  - anything that shows that you clearly were able to run the example
- url to a W&B Run or [W&B Report](https://api.wandb.ai/links/aurora_gpt/7du35js1)
- etc.



<!--[^gpu]: If you do not have access to the ALCF systems, you can install [OpenMPI](https://docs.open-mpi.org/en/v5.0.x/) and run across multiple CPUs as well-->
