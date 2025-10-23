# Coupling Simulation and AI

Material created by Riccardo Balin, Christine Simpson, Bethany Lusch, Logan Ward from Argonne National Laboratory.

## Introduction

This lecture series covers ...

## Slides

Lecture style slides can be found in: [slides/ALCF_AI-students-advanced-03.pdf](slides/ALCF_AI-students-advanced-03.pdf)

## Hands On

0. Clone the repository or pull form the main branch:

    ```bash
    cd /path/to/desired_location
    git clone https://github.com/argonne-lcf/ai-science-training-series.git

    OR

    cd /path/to/desired_location/ai-science-training-series
    git pull origin main
    ```

1. Submit interactive job:

    ```bash
    qsub -I -l select=1 -l walltime=01:00:00 -q ALCFAITP -l filesystems=home:eagle -A ALCFAITP
    ```

2. Source the environment provided:

    ```bash
    cd /path/to/desired_location/ai-science-training-series/03-Coupling-Sim-AI
    source 0_activate_env.sh
    ```

    Instructions for creating the environment are located at [utils/install_env.md](./utils/install_env.md). 

3. Run the Parsl example:

    ```bash
    cd /path/to/desired_location/ai-science-training-series/03-Coupling-Sim-AI/parsl
    python 1_run_simulation.py
    python 2_training_and_inference.py
    python 3_ml_in_the_loop.py
    ```

4. Run the DragonHPC example:

    ```bash
    cd /path/to/desired_location/ai-science-training-series/03-Coupling-Sim-AI/dragonhpc
    dragon 1_run_simulation.py
    dragon 2_training_and_inference.py
    dragon 3_ml_in_the_loop.py
    ```

## Homework

To explore these topics further, please execute and submit the following exercises:
