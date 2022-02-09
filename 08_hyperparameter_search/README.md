# Hyperparameter and Neural Architecture Search with DeepHyper

DeepHyper is a distributed machine learning (AutoML) package for automating the development of deep neural networks for scientific applications. It can run on a single laptop as well as on 1,000 of nodes.

The [DeepHyper Documentation](https://deephyper.readthedocs.io/en/latest/index.html) provides more examples and details about the functionnalities of the software.

For a list of scientific applications using DeepHyper refer to the [Research & Publications](https://deephyper.readthedocs.io/en/latest/research.html) page.

## Environment Setup

Start by cloning the git repository with this folder. If you are using ALCF, see our [previous tutorial's instructions.](https://github.com/argonne-lcf/ai-science-training-series/blob/main/00_introToAlcf/02_howToSetupEnvironment.md#git-repo)

From a terminal run the following commands:

```
ssh username@theta.alcf.anl.gov
```
```
git clone https://github.com/argonne-lcf/ai-science-training-series.git
```


### ALCF Jupyter-Hub

You can run the notebooks of this session on ALCF's Jupyter-Hub. 

1. [Log in to a ThetaGPU compute node via Jupyter-Hub](https://github.com/argonne-lcf/ai-science-training-series/blob/main/00_introToAlcf/04_jupyterNotebooks.md)

2. Change the notebook's kernel to `conda/2021-11-30` (you may need to change kernel each time you open a notebook for the first time):

    1. select *Kernel* in the menu bar
    1. select *Change kernel...*
    1. select *conda/2021-11-30* from the drop-down menu

## Hyperparameter search for classification with Tabular data (Keras)

This tutorial can be retrieved in the [DeepHyper Documentation](https://deephyper.readthedocs.io/en/latest/tutorials/tutorials/colab/HPS_basic_classification_with_tabular_data/notebook.html) and is directly executable on Google Colab.

Once your are connected on ALCF Jupyter-Hub, start a server on a ThetaGPU `single-gpu` queue and open the `Hyperparameter-Search-With-DeepHyper.ipynb` notebook.
