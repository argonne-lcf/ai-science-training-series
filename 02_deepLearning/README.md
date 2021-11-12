# Introduction to Deep Learning


This tutorial covers the basics of Deep Learning with Convolutional Neural Nets. The tutorial is broken into three notebooks. The topics covered in each notebook are:

1. **Intro.ipynb**: 

      - *Linear Regression* as _single layer, single neuron model_ to motivate the introduction of Neural Networks as Universal Approximators that are modeled as collections of neurons connected in an acyclic graph
      - _Convolutions_ and examples of simple _image filters_ to motivate the construction of _Convolutionlal Neural Networks._
      - Loss/Error functions, Gradient Decent, Backpropagation, etc

2. **Mnist.ipynb**: 

    - Visualizing Data
    - Constructing simple Convolutional Neural Networks
    - Training and Inference
    - Visualizing/Interpreting trained Neural Nets

3. **CIFAR-10.ipynb**: 

    - Data Generators
    - Overfitting
    - Data Augmentation



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

2. Change the notebook's kernel to `conda/2021-09-22` (you may need to change kernel each time you open a notebook for the first time):

    1. select *Kernel* in the menu bar
    1. select *Change kernel...*
    1. select *conda/2021-09-22* from the drop-down menu



## __References:__

The code examples presented here are mostly taken (verbatim) or inspired from the following sources. I made this curation to give a quick exposure to very basic but essential ideas/practices in deep learning to get you started fairly quickly, but I recommend going to some or all of the actual sources for an in depth survey:

- [tensorflow.org Turorials](https://www.tensorflow.org/tutorials)
- [keras.io tutorials](https://keras.io/examples/)
- [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
- [Deep Learnig Specialization, Andrew Ng](https://www.coursera.org/specializations/deep-learning?utm_source=deeplearningai&utm_medium=institutions&utm_campaign=WebsiteCoursesDLSTopButton)
- [PyTorch Challenge, Udacity](https://www.udacity.com/facebook-pytorch-scholarship)
- [Deep Learning with Python](https://www.amazon.com/Deep-Learning-Python-Francois-Chollet/dp/1617294438)
- [Keras Blog](https://blog.keras.io/)
