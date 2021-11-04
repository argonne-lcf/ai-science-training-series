# Introduction to Machine Learning

Week 2 of our AI for Science tutorial series focuses is on machine learning, starting on the classic methods and fundamental techniques.

The corresponding notebooks to the lecture focus on first learning how to use Scikit-Learn, a widely-used machine learning package in Python, 
and then illustrate how to use it with scientific data that requires pre-processing, using molecular property prediction as an example.

## Environment Setup

Start by cloning the git repository with this folder. In you are using ALCF, see our [previous tutorial's instructions.](https://github.com/argonne-lcf/ai-science-training-series/blob/main/00_introToAlcf/02_howToSetupEnvironment.md#git-repo)

There are two ways to run the notebooks.

### ALCF Jupyter

We first need to let Jupyter know how to run notebooks from the Python environment created in this class.

[Log in to ThetaGPU via SSSH](https://github.com/argonne-lcf/ai-science-training-series/blob/main/00_introToAlcf/01_howToLogin.md) then execute the following instructions:

```bash
# Load in Anaconda in your terminal
module load conda/2021-09-22

# Use Anaconda to activate the environment we've prepared for you
source activate /lus/grand/projects/ALCFAITP/conda/rapids-21.10;

# Install a new Python environment ("kernel") for Jupyter to use
python -m ipykernel install --user --name rapids-21.10

```

Once you do that, you will need to tell Jupyter to use that Python kernel **each time you open a notebook for the first time.**
Do so by following the following steps:

1. select *Kernel* in the menu bar
1. select *Change kernel...*
1. select *rapids-21.10* from the drop-down menu

### Local Installation

The `environment.yml` file provided with this README describes how to build the environment with anaconda.

Once you have anaconda installed, build the environment by calling:

```bash
conda env create --file environment.yml
```

from the command line. Once installed, follow the instructions Anaconda generates to activate the environment and then launch Jupyter:

```bash
jupyter lab
```

**NOTE**: Some parts of the notebooks in Part 2 will not work for local installation unless you add [Rapids](https://rapids.ai/) to your Anaconda environment. 
After activating your environment, use the following command to add Rapids:

```bash
conda install -c rapidsai -c nvidia -c conda-forge    rapids-blazing=21.10 cudatoolkit=11.0
```