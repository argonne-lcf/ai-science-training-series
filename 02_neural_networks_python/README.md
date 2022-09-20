# Neural Networks in Python

Author: Bethany Lusch, combining and adapting materials evolved over time by Asad Khan, Prasanna Balaprakash, Taylor Childers, Corey Adams, Kyle Felker, and Tanwi Mallick 

This tutorial covers the basics of neural networks. We will learn about the mathematics of neural networks by building them "by hand." In next week's tutorial, we will learn about how to use the higher-level functions in the Python module TensorFlow. 

The tutorial is broken into two notebooks. The topics covered in each notebook are:

1. **Intro.ipynb**: 

      - *Linear Regression* as _single layer, single neuron model_ to motivate the introduction of Neural Networks as Universal Approximators that are modeled as collections of neurons connected in an acyclic graph
      - _Convolutions_ and examples of simple _image filters_ to motivate the construction of _Convolutionlal Neural Networks._
      - Loss/Error functions, Gradient Decent, Backpropagation, etc

2. **Mnist.ipynb**: 

    - Visualizing Data
    - Constructing simple Convolutional Neural Networks
    - Training and Inference
    - Visualizing/Interpreting trained Neural Nets




## Environment Setup
1. If you are using ALCF, first log in. From a terminal run the following command:
```
ssh username@theta.alcf.anl.gov
```

2. Although last week we cloned the repo, you'll want the updated version. To be reminded of last week's instructions for syncing your fork, click [here](https://github.com/argonne-lcf/ai-science-training-series/blob/main/00_introToAlcf/03_githubHomework.md). 

3. Now that we have the updated notebooks, we can open them. If you are using ALCF JupyterHub, you can be reminded of the steps [here](https://github.com/argonne-lcf/ai-science-training-series/blob/main/00_introToAlcf/02_jupyterNotebooks.md). 

4. Change the notebook's kernel to `conda/2021-09-22` (you may need to change kernel each time you open a notebook for the first time):

    1. select *Kernel* in the menu bar
    1. select *Change kernel...*
    1. select *conda/2021-09-22* from the drop-down menu



## __References:__

Some of the code examples presented here are inspired from the following sources. In this tutorial series, we are exposing you to very basic but essential ideas/practices in deep learning, but here are some suggestions for further reading:

- [tensorflow.org turorials](https://www.tensorflow.org/tutorials)
- [keras.io tutorials](https://keras.io/examples/)
- [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
- [Deep Learnig Specialization, Andrew Ng](https://www.coursera.org/specializations/deep-learning?utm_source=deeplearningai&utm_medium=institutions&utm_campaign=WebsiteCoursesDLSTopButton)
- [PyTorch Challenge, Udacity](https://www.udacity.com/facebook-pytorch-scholarship)
- [Deep Learning with Python](https://www.amazon.com/Deep-Learning-Python-Francois-Chollet/dp/1617294438)
- [Keras Blog](https://blog.keras.io/)
