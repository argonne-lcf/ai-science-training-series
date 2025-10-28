# Coupling Simulation and AI

Material created by Riccardo Balin, Christine Simpson, Bethany Lusch, Logan Ward from Argonne National Laboratory.

## Introduction

This lecture covers methods and best practices for coupling traditional simulation with AI methods on modern HPC systems. 
We provide a set of slides to introduce why coupling simulation and AI can be beneficial and offer an overview of methods that can be used to create various types of coupling.
We then provide two hands-on examples to be run on the Polaris supercomputer at ALCF, with a homework problem for each.

The first example ([ml-in-the-loop](./ml-in-the-loop/README.md)) demonstrates a full end-to-end, active learning workflow implemented in Parsl inspired from the field of molecular design. This simplified application uses a combination of simulation and machine learning (ML) training and inference to identify which molecules have the largest ionization energies among a large dataset. 

The second example ([producer-consumer](./producer-consumer/)) uses a simplified workflow pattern implemented in Parsl and DragonHPC wherein an ensemble of toy simulations produce data to be consumed by a second ensemble of ML model training instances. This example is designed to investigate some of the various techniques we use to couple simulations and AI/ML on HPC systems, leveraging different software and hardware solutions to store and transfer data between components.

## Slides

The lecture slides can be found in: [slides/ALCF_AI-students-advanced-03.pdf](slides/ALCF_AI-students-advanced-03.pdf).

## Hands On Exercises

0. Clone the repository or pull form the main branch:

    ```bash
    cd /path/to/desired_location
    git clone https://github.com/argonne-lcf/ai-science-training-series.git

    OR

    cd /path/to/desired_location/ai-science-training-series
    git pull origin main
    ```

1. Submit an interactive job:

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

4. Run the producer-consumer example with DragonHPC and Parsl:

    ```bash
    cd /path/to/desired_location/ai-science-training-series/03-Coupling-Sim-AI/producer-consumer
    python 5_parsl_fut_producer_consumer.py
    python 6_parsl_fs_producer_consumer.py
    dragon 7_dragon_ddict.py
    dragon 8_dragon_producer_consumer.py
    ```

## Homework

To explore these topics further, please execute and submit the following exercises:

1. Tune the parameters of the ML-in-the-loop active learning workflow in order to find molecules with the largest ionization energy in the shortest possible time. Use the PBS submit script [4_submit_multinode.sh](./ml-in-the-loop/4_submit_multinode.sh) to run the full workflow on 1 or multiple nodes of Polaris. Note that all that should be needed for this exercise is to change the values of the `initial_training_count`, `max_training_count` and `batch_size` variables at the top of the `3_ml_in_the_loop.py` script. Submit the plot that is produced by the script as well as your code to showcase yor results and how you obtained them.
2. Experiment with the Parsl and DragonHPC implementations of the producer-consumer workflow by scaling up the problem size, both in terms of size of the data being produced/transferred and in terms of the number of nodes (although 2-4 nodes will be sufficient). Collect data from these experiments to fill in the table under the `Data Transfer Performance (Homework)` section at the bottom of the [example README](./producer-consumer/README.md) and write a short paragraph about your observations. See the notes under the `Data Transfer Performance (Homework)` section for more detailed information and some hints. 
