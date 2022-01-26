# Distributed training with Horovod
Led by Huihuo Zheng from ALCF (<huihuo.zheng@anl.gov>)

**Goal of this tutorial**
* Understand model parallelism and data parallelism
* Know how to modify your code with Horovod
* Know how to run distributed training on supercomputer

## Introduction to distributed Deep Learning
The goal for train the model at large scale is to reduce the time-to-solution to reasonable amount. By using training the model in parallel, it reduces the total training time from weeks to minutes.
![acc](./images/resnet50.png)


## Model Parallelism and Data Parallelism

1. **Model parallelism**: in this scheme, disjoint subsets of a neural network are assigned to different devices. Therefore, all the computations associated to the subsets are distributed. Communication happens between devices whenever there is dataflow between two subsets. Model parallelization is suitable when the model is too large to be fitted into a single device (CPU/GPU) because of the memory capacity. However, partitioning the model into different subsets is not an easy task, and there might potentially introduce load imbalance issues limiting the scaling efficiency.  
2. **Data parallelism**: in this scheme, all the workers own a replica of the model. The global batch of data is split into multiple minibatches, and processed by different workers. Each worker computes the corresponding loss and gradients with respect to the data it posseses. Before the updating of the parameters at each epoch, the loss and gradients are averaged among all the workers through a collective operation. This scheme is relatively simple to implement. MPI_Allreduce is the only commu

![acc](./images/distributed.png)

Our recent presentation about the data parallel training can be found here: https://youtu.be/930yrXjNkgM

## Horovod Data Parallel Framework
![Horovod](./images/Horovod.png)
Reference: https://horovod.readthedocs.io/en/stable/
1. Sergeev, A., Del Balso, M. (2017) Meet Horovod: Uber’s Open Source Distributed Deep Learning Framework for TensorFlow. Retrieved from https://eng.uber.com/horovod/
2. Sergeev, A. (2017) Horovod - Distributed TensorFlow Made Easy. Retrieved from https://www.slideshare.net/AlexanderSergeev4/horovod-distributed-tensorflow-made-easy

3. Sergeev, A., Del Balso, M. (2018) Horovod: fast and easy distributed deep learning in TensorFlow. Retrieved from arXiv:1802.05799

**Steps to modify your code with Horovod**:
  1. Initialize Horovod
  2. Pin GPU to each process
  3. Scale the learning rate
  4. Set distributed optimizer / gradient tape
  5. Broadcast the model & optimizer parameters to other rank
  6. Checking pointing on rank 0
  7. Adjusting dataset loading: number of steps (or batches) per epoch, dataset sharding, etc.
  8. Average metric across all the workers

## TensorFlow with Horovod
1) **Initialize Horovod**
```python
import horovod.tensorflow as hvd 
hvd.init()
```
After this initialization, the rank ID and the number of processes can be refered as ```hvd.rank()``` and ```hvd.size()```. Besides, one can also call ```hvd.local_rank()``` to get the local rank ID within a node. This is useful when we are trying to assign GPUs to each rank. 

2) **Assign GPUs to each rank**
```python
# Get the list of GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
# Ping GPU to the rank
tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
```
In this case, we set one GPU per process: ID=```hvd.local_rank()```

3) **Scale the learning rate with number of workers**

Typically, since we use multiple workers, if we keep the local batch size on each rank the same, the global batch size increases by $n$ times ($n$ is the number of workers). The learning rate should increase proportionally (assuming that the learning rate initially is 0.01).
```python
opt = tf.train.AdagradOptimizer(0.01*hvd.size())
```

4) **Wrap tf.GradientTape with Horovod Distributed Gradient Tape**

```python
tape = hvd.DistributedGradientTape(tape)
```
So that this can also ```tape``` operator will average the weights and gradients among the workers in the back propagating stage. 

5) **Broadcast the model from rank 0**

This is to make sure that all the workers will have the same starting point.
```python
hvd.broadcast_variables(model.variables, root_rank=0)
hvd.broadcast_variables(opt.variables(), root_rank=0)
```
**Note: broadcast should be done after the first gradient step to ensure optimizer initialization.**

6) **Checkpointing on root rank**

It is important to let only one process to do the checkpointing I/O. 
```python
if hvd.rank() == 0: 
     checkpoint.save(checkpoint_dir)
```

7) **Loading data according to rank ID and ajusting the number of time steps**

In data parallelism, we distributed the dataset to different workers. It is important to make sure different workers work on different part of the dataset, and they together can cover the entire dataset at each epoch. 

In general, one has two ways to deal with the data loading. 
1. Each worker randomly selects one batch of data from the dataset at each step. In such case, each worker can see the entire dataset. It is important to make sure that the different worker have different random seeds so that they will get different data at each step.
2. Each worker accesses a subset of dataset. One manually partition the entire dataset into different partions, and each rank access one of the partions. 

8) **Average the metrics across all the workers**
```python
total_loss = hvd.allreduce(running_loss, average=True)
total_acc = hvd.allreduce(running_acc, average=True)
```



Example in: [Horovod](Horovod/) 
* [tensorflow2_mnist.py](tensorflow2_mnist.py)


## Keras with Horovod
1) **Initialize Horovod**
```python
import horovod.tensorflow.keras as hvd 
hvd.init()
```
After this initialization, the rank ID and the number of processes can be refered as ```hvd.rank()``` and ```hvd.size()```. Besides, one can also call ```hvd.local_rank()``` to get the local rank ID within a node. This is useful when we are trying to assign GPUs to each rank. 

2) **Assign GPUs to each rank**
```python
# Get the list of GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
# Ping GPU to the rank
tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
```
In this case, we set one GPU per process: ID=```hvd.local_rank()```

3) **Scale the learning rate with number of workers**

Typically, since we use multiple workers, if we keep the local batch size on each rank the same, the global batch size increases by $n$ times ($n$ is the number of workers). The learning rate should increase proportionally (assuming that the learning rate initially is 0.01).
```python
opt = tf.optimizers.Adam(args.lr * hvd.size())
```

4) **Wrap tf.optimizer with Horovod DistributedOptimizer**

```python
opt = hvd.DistributedOptimizer(opt)
```
So that this optimizer can average the weights and gradients among the workers in the back propagating stage. 

5) **Broadcast the model from rank 0**

This is to make sure that all the workers will have the same starting point.
```python
callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0), ...
]

```

6) **Checkpointing on root rank**

It is important to let only one process to do the checkpointing I/O. 
```python
if hvd.rank() == 0:
    callbacks.append(tf.keras.callbacks.ModelCheckpoint('./checkpoints/keras_mnist-{epoch}.h5'))
```

7) **Loading data according to rank ID and adjusting the number of steps**

In data parallelism, we distributed the dataset to different workers. It is important to make sure different workers work on different part of the dataset, and they together can cover the entire dataset at each epoch. 

In general, one has two ways to deal with the data loading. 
1. Each worker randomly selects one batch of data from the dataset at each step. In such case, each worker can see the entire dataset. It is important to make sure that the different worker have different random seeds so that they will get different data at each step.
2. Each worker accesses a subset of dataset. One manually partition the entire dataset into different partions, and each rank access one of the partions. 

8) **Average the metrics across all the workers**
```python
total_loss = hvd.allreduce(running_loss, average=True)
total_acc = hvd.allreduce(running_acc, average=True)
```

We provided some examples in: [Horovod](Horovod/) 
* [tensorflow2_keras_mnist.py](tensorflow2_keras_mnist.py)
* [tensorflow2_keras_cifar10.py](tensorflow2_keras_cifar10.py)


## PyTorch with Horovod
It is very similar for PyTorch with Horovod
1) **Initialize Horovod**
```python
import horovod.torch as hvd 
hvd.init()
```
After this initialization, the rank ID and the number of processes can be refered as ```hvd.rank()``` and ```hvd.size()```. Besides, one can also call ```hvd.local_rank()``` to get the local rank ID within a node. This is useful when we are trying to assign GPUs to each rank. 

2) **Assign GPUs to each rank**
```python
torch.cuda.set_device(hvd.local_rank())
```
In this case, we set one GPU per process: ID=```hvd.local_rank()```

3) **Scale the learning rate.**

Typically, since we use multiple workers, the global batch is usually increases n times (n is the number of workers). The learning rate should increase proportionally as follows (assuming that the learning rate initially is 0.01).
```python
optimizer = optim.SGD(model.parameters(), lr=args.lr * hvd.size(), momentum=args.momentum)
```

4) **Wrap the optimizer with Distributed Optimizer**
```python
optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters(), compression=compression)
```

5) **Broadcast the model from rank 0**

This is to make sure that all the workers will have the same starting point.
```python
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)
```

6) **Loading data according to rank ID**

In general, one has two ways to deal with the data loading. 
1. Each worker randomly select one batch of data from the dataset at each step. In such case, each worker can see the entire dataset. It is important to make sure that the different worker have different random seeds so that they will get different data at each step.  
2. Each worker accesses a subset of dataset. One manually partition the entire dataset into different partions, and each rank access one of the partions. 

PyTorch has some functions for parallel distribution of data. 
```python
train_dataset = \
    datasets.MNIST('datasets/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)
```

In both cases, the total number of steps per epoch is ```nsamples / hvd.size()```.

7) **Checkpointing on root rank**
It is important to let only one process to do the checkpointing I/O lest perhaps the file been corrupted. 
```python
if hvd.rank() == 0: 
     checkpoint.save(checkpoint_dir)
```

8) **Average metric across all the workers**
Notice that in the distributed training, any tensor are local to each worker. In order to get the global averaged value, one can use Horovod allreduce. Below is an example on how to do the average. 
```python
def tensor_average(val, name):
    tensor = torch.tensor(val)
    if (with_hvd):
        avg_tensor = hvd.allreduce(tensor, name=name)
    else:
        avg_tensor = tensor
    return avg_tensor.item()
```
We provided some examples in: [Horovod](Horovod/) 
* [pytorch_mnist.py](pytorch_mnist.py)
* [pytorch_cifar10.py](pytorch_cifar10.py)

## Handson
* [thetagpu.md](thetagpu.md): running on ThetaGPU (```--device gpu```)

For submitting jobs in the script (non-interactive) job mode, check the submission scripts in the [submissions](./submission/) folder. 
