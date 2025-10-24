# Instructions for Installing the Environment for the AI4S Sim+AI Examples on ALCF Polaris

1. Load the ML module to get access to conda
    ```bash
    module use /soft/modulefiles/
    module load conda/2025-09-25
    conda activate
    ```

2. Create a new conda env (can't install xtb-python via pip, so can't use a Python venv)
    ```bash
    conda create -y --prefix /eagle/datascience/balin/AI4S/_ai4s_simAI python=3.12 pip
    conda activate /eagle/datascience/balin/AI4S/_ai4s_simAI
    ```

3. Install the packages related to chemistry problem
    ```bash
    pip install ase rdkit pandas scikit-learn tqdm
    conda install -c conda-forge -y xtb-python
    ```

3. Install Parsl
    ```bash
    pip install parsl
    ```

5. Install DragonHPC
    ```bash
    pip install dragonhpc==0.12.3
    dragon-config add --ofi-runtime-lib=/opt/cray/libfabric/1.22.0/lib64
    ```

6. Install the `chemfunctions` package which comes with the example repository
    ```bash
    cd /path/to/desired_location
    git clone https://github.com/argonne-lcf/ai-science-training-series.git
    cd ai-science-training-series/03-Coupling-Sim-AI/ml-in-the-loop/chemfunctions
    pip install .
    ```

7. Export important environment variables
   ```bash
   export TMPDIR=/tmp
   export PATH=$PATH:/opt/pbs/bin
   ```

