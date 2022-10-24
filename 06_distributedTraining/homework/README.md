# Scaling MNIST example
The goal of this homework is to modify a sequential mnist code into a data parallel code with Horovod and test the scaling efficiency

* 50%: Modify the [rain_mnist.py](./train_mnist.py) to Horovod (save it as "./train_mnist_hvd.py"
* 50%: Run scaling test on ThetaGPU or Polaris, unto 8 GPU, and generate the scaling plot (save it as mnist_scaling.png). (y axis is the time per epoch and x-axis is the number of GPU)

Provide the link to your ./homework folder on your personal GitHub repo. 