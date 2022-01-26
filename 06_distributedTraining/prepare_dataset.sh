#!/bin/sh
# Please run this script on a login node

# TensorFlow MNIST dataset
[ -e mnist.npz ] || wget https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
[ -e ~/.keras/ ] || mkdir ~/.keras/
[ -e ~/.keras/datasets/ ] || mkdir ~/.keras/datasets
cp mnist.npz ~/.keras/datasets

# TensorFlow cifar10 dataset
[ -e cifar-10-python.tar.gz ] || wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
cp cifar-10-python.tar.gz ~/.keras/datasets/cifar-10-batches-py.tar.gz
cd ~/.keras/datasets/
tar -xzf cifar-10-batches-py.tar.gz
cd -


# PyTorch MNIST dataset
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz 
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz 
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

[ -e datasets ] || mkdir datasets
[ -e datasets/MNIST ] || mkdir datasets/MNIST
[ -e datasets/MNIST/raw ] || mkdir datasets/MNIST/raw
mv *.gz datasets/MNIST/raw/


# PyTorch cifar10 dataset
[ -e cifar-10-python.tar.gz ] || wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
cp cifar-10-python.tar.gz datasets/cifar-10-python.tar.gz

