#!/bin/bash

module use /soft/modulefiles/
module load conda/2025-09-25
conda activate

# Conda env
conda create -p env/_ai4s_agentic_conda python=3.10
conda activate /lus/eagle/projects/ALCFAITP/04-Inference-Workflows/env/_ai4s_agentic_conda

# LangChain/LangGraph
pip install langchain langgraph langchain-openai 

# globus_sdk for ALCF Inference Endpoints
pip install globus_sdk

# Pubchempy
pip install "pubchempy @ git+https://github.com/keceli/PubChemPy.git@main"

# ASE, MACE, RDKit
pip install rdkit ase mace-torch

#Jupyter
pip install jupyter