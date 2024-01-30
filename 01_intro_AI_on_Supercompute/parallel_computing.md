# Distributed trainng on Supercomputer
Led by Huihuo Zheng from ALCF (<huihuo.zheng@anl.gov>)

**Goal of this tutorial**
* Understand parallelization 
	- Model parallelism
	- Data parallelism
* Know how to modify your code with Horovod
* Know how to run distributed training on Polaris / ThetaGPU and measuring the scaling efficiency

## Concept of Parallel Computing  - pi examples

![PI](https://www.101computing.net/wp/wp-content/uploads/estimating-pi-monte-carlo-method.png)

```python
from mpi4py import MPI
import numpy as np
import random
import time
comm = MPI.COMM_WORLD

N = 5000000
Nin = 0
t0 = time.time()
for i in range(comm.rank, N, comm.size):
    x = random.uniform(-0.5, 0.5)
    y = random.uniform(-0.5, 0.5)
    if (np.sqrt(x*x + y*y) < 0.5):
        Nin += 1
res = np.array(Nin, dtype='d')
res_tot = np.array(Nin, dtype='d')
comm.Allreduce(res, res_tot, op=MPI.SUM)
t1 = time.time()
if comm.rank==0:
    print(res_tot/float(N/4.0))
    print("Time: %s" %(t1 - t0))
```


```bash
ssh <username>@theta.alcf.anl.gov
ssh thetagpusn1 
qsub -A ALCFAITP -n 1 -q training-gpu -t 20 -I 
module load conda/2022-07-01
conda activate
cd YOUR_GITHUP_REPO
mpirun -np 1 python pi.py   # 3.141988,   8.029037714004517  s
mpirun -np 2 python pi.py   # 3.1415096   4.212774038314819  s
mpirun -np 4 python pi.py   # 3.1425632   2.093632459640503  s
mpirun -np 8 python pi.py   # 3.1411632   1.0610620975494385 s
```

## Introduction to distributed Deep Learning
![acc](./images/need.png)
The goal for train the model at large scale is to reduce the time-to-solution to reasonable amount. By using training the model in parallel, it reduces the total training time from weeks to minutes.
![acc](./images/resnet50.png)


## Model Parallelism and Data Parallelism

1. **Model parallelism**: in this scheme, disjoint subsets of a neural network are assigned to different devices. Therefore, all the computations associated to the subsets are distributed. Communication happens between devices whenever there is dataflow between two subsets. Model parallelization is suitable when the model is too large to be fitted into a single device (CPU/GPU) because of the memory capacity. However, partitioning the model into different subsets is not an easy task, and there might potentially introduce load imbalance issues limiting the scaling efficiency.  
2. **Data parallelism**: in this scheme, all the workers own a replica of the model. The global batch of data is split into multiple minibatches, and processed by different workers. Each worker computes the corresponding loss and gradients with respect to the data it posseses. Before the updating of the parameters at each epoch, the loss and gradients are averaged among all the workers through a collective operation. This scheme is relatively simple to implement. MPI_Allreduce is the only commu

![acc](./images/distributed.png)
![acc](./images/pipeline.png)

Our recent presentation about the data parallel training can be found here: https://youtu.be/930yrXjNkgM

## Horovod Data Parallel Framework
![Horovod](./images/Horovod.png)
Reference: https://horovod.readthedocs.io/en/stable/
1. Sergeev, A., Del Balso, M. (2017) Meet Horovod: Uber’s Open Source Distributed Deep Learning Framework for TensorFlow. Retrieved from https://eng.uber.com/horovod/
2. Sergeev, A. (2017) Horovod - Distributed TensorFlow Made Easy. Retrieved from https://www.slideshare.net/AlexanderSergeev4/horovod-distributed-tensorflow-made-easy

3. Sergeev, A., Del Balso, M. (2018) Horovod: fast and easy distributed deep learning in TensorFlow. Retrieved from arXiv:1802.05799

**8 Steps to modify your code with Horovod**:
  1. Initialize Horovod
  2. Pin GPU to each process
  3. Sharding / partioning the dataset
  4. Scale the learning rate
  5. Set distributed optimizer / gradient tape
  6. Broadcast the model & optimizer parameters to other rank
  7. Checking pointing on rank 0
  8. Average metric across all the workers

## Example: TensorFlow with Horovod

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
	for gpu in gpus:
	        tf.config.experimental.set_memory_growth(gpu, True)
	if gpus:
   		tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
	```
	In this case, we set one GPU per process: ID=```hvd.local_rank()```
	
	**For Tensorflow with Horovod, it is important to set tf.config.experimental.set_memory_growth(gpu, True)**

3) **Loading data according to rank ID and ajusting the number of time steps**

	In data parallelism, we distributed the dataset to different workers. It is important to make sure different workers work on different part of the dataset, and they together can cover the entire dataset at each epoch. 

	For TensorFlow, if you are using ```tf.data.Dataset```, you can use the sharding functionality 
	```python
	dataset = dataset.shard(num_shards=hvd.size(), index=hvd.rank())
	```
	where dataset is a ```tf.data.Dataset``` object. 

4) **Scale the learning rate with number of workers**

	Typically, since we use multiple workers, if we keep the local batch size on each rank the same, the global batch size increases by $n$ times ($n$ is the number of workers). The learning rate should increase proportionally (assuming that the learning rate initially is 0.01).
	```python
	opt = tf.train.AdagradOptimizer(0.01*hvd.size())
	```

5) **Wrap tf.GradientTape with Horovod Distributed Gradient Tape**

	```python
	tape = hvd.DistributedGradientTape(tape)
	```
	So that this can also ```tape``` operator will average the weights and gradients among the workers in the back propagating stage. 

6) **Broadcast the model from rank 0**

	This is to make sure that all the workers will have the same starting point.
	```python
	hvd.broadcast_variables(model.variables, root_rank=0)
	hvd.broadcast_variables(opt.variables(), root_rank=0)
	```
	**Note: broadcast should be done AFTER the first gradient step to ensure optimizer initialization.**

7) **Checkpointing on root rank**

	It is important to let only one process to do the checkpointing I/O. 
	```python
	if hvd.rank() == 0: 
		checkpoint.save(checkpoint_dir)
	```

8) **Average the metrics across all the workers**
	```python
	loss = hvd.allreduce(loss, average=True)
	acc = hvd.allreduce(acc, average=True)
	```

Example in: 
* [train_resnet34_hvd.py](train_resnet34_hvd.py)


Examples for other frameworks (PyTorch, Keras, MxNet) can be found [here](https://github.com/horovod/horovod/tree/master/examples). 

## Handson 
* Changing the code into Horovod (during break time)
```bash
ssh <username>@theta.alcf.anl.gov
ssh thetagpusn1 
cd /lus/grand/projects/ALCFAITP/hzheng/ai-science-training-series/06_distributedTraining
cp train_resnet34.py train_resnet34_parallel.py 
```
Implement```train_resnet34_parallel.py``` with Horovod

* Throughput scaling
```
ssh <username>@theta.alcf.anl.gov
ssh thetagpusn1 
qsub -A ALCFAITP -n 1 -q training-gpu -t 20 -I 
```

```bash
	module load conda/2022-07-01
	conda activate
    mpirun -n 1 python train_resnet34_hvd.py --num_steps 10 
    mpirun -n 2 python train_resnet34_hvd.py --num_steps 10 
    mpirun -n 4 python train_resnet34_hvd.py --num_steps 10 
    mpirun -n 8 python train_resnet34_hvd.py --num_steps 10 
```

1 GPU: mean image/s =   281.22   standard deviation:    75.79
2 GPU: mean image/s =   382.01   standard deviation:     8.42
4 GPU: mean image/s =   689.22   standard deviation:    77.78
8 GPU: mean image/s =  1341.25   standard deviation:    52.51 
...


* Visualizing communication 
```
HOROVOD_TIMELINE=timeline.json mpirun -n 8 python train_resnet34_hvd.py --num_steps 10
```
Horovod timeline
![./images/horovod_timeline.png](./images/horovod_timeline.png)

## Homework
### Scaling MNIST example
The goal of this homework is to modify a sequential mnist code into a data parallel code with Horovod and test the scaling efficiency

* 50%: Modify the [./homework/tensorflow2_mnist.py](./homework/tensorflow2_mnist.py) to Horovod (save it as "./homework/tensorflow2_mnist_hvd.py"

* 25%: Run scaling test upto 16 nodes, and check the overall timing
```bash
    mpirun -n 1 python tensorflow2_mnist_hvd.py
    mpirun -n 2 python tensorflow2_mnist_hvd.py
    mpirun -n 4 python tensorflow2_mnist_hvd.py
    mpirun -n 8 python tensorflow2_mnist_hvd.py
```

* 25%: Plot the training accuracy and validation accuracy curve for different scales. (x-asix: epoch; y-axis: accuracy)
Save your plots as pdf files in the [./homework](./homework) folder "accuracy_1.pdf, accuracy_2.pdf, accuracy_4.pdf, accuracy_8.pdf"

Provide the link to your ./homework folder on your personal GitHub repo. 

* Bonus: 
The accuracy for large scale training can be improved by using smaller learning rate in the beginning few epochs (warmup epochs). Implement the warmup epochs 
