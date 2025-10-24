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

3. Run the ML-in-the-loop example with Parsl:

    ```bash
    cd /path/to/desired_location/ai-science-training-series/03-Coupling-Sim-AI/ml-in-the-loop
    python 1_run_simulation.py
    python 2_training_and_inference.py
    python 3_ml_in_the_loop.py
    ```

4. Run the producer-consumer example:

    ```bash
    cd /path/to/desired_location/ai-science-training-series/03-Coupling-Sim-AI/producer-consumer
    dragon 5_dragon_ddict.py
    dragon 6_dragon_producer_consumer.py
    python 7_parsl_fs_producer_consumer.py
    ```

## Homework

To explore these topics further, please execute and submit the following exercises:
