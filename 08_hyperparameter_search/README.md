# DeepHyper


DeepHyper is a distributed machine learning (AutoML) package for automating the development of deep neural networks for scientific applications. DeepHyper uses artificial intelligence (AI) techniques and parallel computing to automate the design and development of machine learning (ML) models for scientific and engineering applications. DeepHyper reduces the barrier to entry for using AI/ML models by reducing manually intensive trial-and-error efforts for developing predictive models. It can run on a single laptop as well as on 1,000 of nodes.

# Hyperparameter Search with DeepHyper
ML methods used for predictive modeling typically require user-specified values for hyperparameters, which include the number of hidden layers and units per layer, sparsity/overfitting regularization parameters, batch size, learning rate, type of initialization, optimizer, and activation function specification. Traditionally, to find performance-optimizing hyperparameter settings, researchers have used a trial-and-error process or a brute-force grid/random search. Such approaches lead to far-from-optimal performance, however, or are impractical for addressing large numbers of hyperparameters. DeepHyper provides a set of scalable hyperparameter search methods for automatically searching for high-performing hyperparameters for a given DNN architecture. DeepHyper uses an asynchronous model-based search that relies on fitting a dynamically updated surrogate model that tries to learn the relationship between the hyperparameter configurations (input) and their validation errors (output). The surrogate model is cheap to evaluate and can be used to prune the search space and identify promising regions, where the model then is iteratively refined by obtaining new outputs at inputs that are predicted by the model to be high-performing.  

# Neural Architecture Search with DeepHyper
Scientific data sets are diverse and often require data-set-specific DNN models. Nevertheless, designing high-performing DNN architecture for a given data set is an expert-driven, time-consuming, trial-and-error manual task. To that end, DeepHyper provides a NAS for automatically identifying high-performing DNN architectures for a given set of training data. DeepHyper adopts an evolutionary algorithm that generates a population of DNN architectures, trains them concurrently by using multiple nodes, and improves the population by performing mutations on the existing architectures within a population. To reduce the training time of each architecture evaluation, DeepHyper adopts a distributed data-parallel training technique, splitting the training data and distributing the shards to multiple processing units. Multiple models with the same architecture are trained on different data shards, and the gradients from each model are averaged and used to update the weights of all the models. To maintain accuracy and reduce training time, DeepHyper combines aging evolution and an asynchronous Bayesian optimization method for tuning the hyperparameters of the data-parallel training simultaneously. 

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
