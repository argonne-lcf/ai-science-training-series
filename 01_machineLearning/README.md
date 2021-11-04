# Introduction to Machine Learning

Week 2 of our AI for Science tutorial series focuses is on machine learning, starting on the classic methods and fundamental techniques.

The corresponding notebooks to the lecture focus on first learning how to use Scikit-Learn, a widely-used machine learning package in Python, 
and then illustrate how to use it with scientific data that requires pre-processing, using molecular property prediction as an example.

## Environment Setup

Start by cloning the git repository with this folder. In you are using ALCF, see our [previous tutorial's instructions.](https://github.com/argonne-lcf/ai-science-training-series/blob/main/00_introToAlcf/02_howToSetupEnvironment.md#git-repo)

There are two ways to run the notebooks.

### ALCF Jupyter-Hub

You can run the notebooks of this session on ALCF's Jupyter-Hub. 

1. [Log in to a ThetaGPU compute node via Jupyter-Hub](https://github.com/argonne-lcf/ai-science-training-series/blob/main/00_introToAlcf/04_jupyterNotebooks.md)

2. **Install the Jupyter kernel `rapids-21.10`**. There are two ways of doing this, from the jupyter notebook or from the terminal via ssh:

    - *from the jupyter notebook*: copy the following code, paste it in a new cell on the notebook and run it
      ```
      !source activate /lus/grand/projects/ALCFAITP/conda/rapids-21.10;\
      python -m ipykernel install --user --name rapids-21.10
      ```
    
    - *from the terminal via ssh*: 

      ```bash
      # Log in to Theta
      ssh username@theta.alcf.anl.gov

      # Log in to a ThetaGPU service node
      ssh thetagpusn1

      # Load Anaconda
      module load conda/2021-09-22

      # Use Anaconda to activate the environment we've prepared for you
      conda activate /lus/grand/projects/ALCFAITP/conda/rapids-21.10

      # Install the new Jupyter kernel to use
      python -m ipykernel install --user --name rapids-21.10
      ```

3. Change the notebook's kernel to `rapids-21.10` (you will need to change kernel each time you open a notebook for the first time):

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
