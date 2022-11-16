# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
import argparse
import numpy as np

import sys, os
import time,math

# This limits the amount of memory used:
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"

t0 = time.time()

parser = argparse.ArgumentParser(description='TensorFlow MNIST Example with Horovod')
parser.add_argument('--epochs', default=10, type=int, help='Number of epochs to run')
parser.add_argument('--num_steps', default=1000000, type=int, help="Number of steps")
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--use_profiler', action='store_true')
args = parser.parse_args()

#parser.add_argument('--batch_size', type=int, default=256, metavar='N',
#                    help='input batch size for training (default: 256)')
#parser.add_argument('--epochs', type=int, default=1, metavar='N',
#                    help='number of epochs to train (default: 16)')
#parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
#                    help='learning rate (default: 0.01)')
#parser.add_argument('--device', default='gpu',
#                    help='Wheter this is running on cpu or gpu')
#parser.add_argument('--num_inter', default=2, help='set number inter', type=int)
#parser.add_argument('--num_intra', default=0, help='set number intra', type=int)
#parser.add_argument('--use_profiler', action='store_true')
#args = parser.parse_args()

# This control parallelism in Tensorflow
parallel_threads = 128
# This controls how many batches to prefetch
prefetch_buffer_size = 8 # tf.data.AUTOTUNE

# how many training steps to take during profiling
num_steps = args.num_steps
use_profiler = True

import tensorflow as tf
from tensorflow.python.profiler import trace

# HVD-1 - initialize Horovd
import horovod.tensorflow as hvd
hvd.init()

print("# I am rank %d of %d" %(hvd.rank(), hvd.size()))
parallel_threads = parallel_threads//hvd.size()
os.environ['OMP_NUM_THREADS'] = str(parallel_threads)
num_parallel_readers = parallel_threads
#num_parallel_readers = tf.data.AUTOTUNE

# HVD-2 - Assign GPUs to each rank
# Get the list of GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
# Ping GPU to the rank
tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


#---------------------------------------------------
# Dataset
#---------------------------------------------------
(mnist_images, mnist_labels), (x_test, y_test) = \
    tf.keras.datasets.mnist.load_data(path='mnist.npz')

dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32),
             tf.cast(mnist_labels, tf.int64))
)
test_dset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(x_test[..., tf.newaxis] / 255.0, tf.float32),
             tf.cast(y_test, tf.int64))
)

#HVD-3 Divide data according to rank ID and ajusting the number of time steps
dataset = dataset.shard(num_shards=hvd.size(), index=hvd.rank())

nsamples = len(list(dataset))
ntests = len(list(test_dset))

# shuffle the dataset, with shuffle buffer to be 10000
dataset = dataset.repeat().shuffle(10000).batch(args.batch_size)
test_dset  = test_dset.repeat().batch(args.batch_size)


#----------------------------------------------------
# Model
#----------------------------------------------------
mnist_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
    tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])
loss = tf.losses.SparseCategoricalCrossentropy()

# HVD-4 Scale the learning rate with number of workers
#opt = tf.train.AdagradOptimizer(0.01*hvd.size())
opt = tf.optimizers.Adam(0.01*hvd.size())

#opt = tf.optimizers.Adam(args.lr)
checkpoint_dir = './checkpoints/tf2_mnist'
checkpoint = tf.train.Checkpoint(model=mnist_model, optimizer=opt)

#------------------------------------------------------------------
# Training
#------------------------------------------------------------------
@tf.function
def training_step(images, labels):
    with tf.GradientTape() as tape:
        probs = mnist_model(images, training=True)
        loss_value = loss(labels, probs)
        pred = tf.math.argmax(probs, axis=1)
        equality = tf.math.equal(pred, labels)
        accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))

    # HVD-5 tf.GradientTape with Horovod Distributed Gradient Tape
    tape = hvd.DistributedGradientTape(tape)
    grads = tape.gradient(loss_value, mnist_model.trainable_variables)
    opt.apply_gradients(zip(grads, mnist_model.trainable_variables))
    return loss_value, accuracy

@tf.function
def validation_step(images, labels):
    probs = mnist_model(images, training=False)
    pred = tf.math.argmax(probs, axis=1)
    equality = tf.math.equal(pred, labels)
    accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))
    loss_value = loss(labels, probs)
    return loss_value, accuracy

from tqdm import tqdm 
t0 = time.time()
nstep = nsamples//args.batch_size
ntest_step = ntests//args.batch_size
metrics={}
metrics['Epochs'] = []
metrics['train_acc'] = []
metrics['valid_acc'] = []
metrics['train_loss'] = []
metrics['valid_loss'] = []
metrics['time_per_epochs'] = []
for ep in range(args.epochs):
    training_loss = 0.0
    training_acc = 0.0
    tt0 = time.time()
    for batch, (images, labels) in enumerate(dataset.take(nstep)):
        loss_value, acc = training_step(images, labels)
        training_loss += loss_value/nstep
        training_acc += acc/nstep
        # HVD-6 broadcast model and parameters from rank 0 to the other ranks
        if (nstep==0 and ep == 0):
            hvd.broadcast_variables(mnist_model.variables, root_rank=0)
            hvd.broadcast_variables(opt.variables(), root_rank=0)
        # HVD-7 Checkpointing on root rank
        if hvd.rank() == 0: 
            checkpoint.save(checkpoint_dir)
            #print('Epoch - %d, step #%06d/%06d\tLoss: %.6f' % (ep, batch, nstep, loss_value))
        #if batch % 100 == 0: 
        #    checkpoint.save(checkpoint_dir)
        #    print('Epoch - %d, step #%06d/%06d\tLoss: %.6f' % (ep, batch, nstep, loss_value))
    
    training_loss = hvd.allreduce(training_loss, average=True)
    training_acc = hvd.allreduce(training_acc, average=True)
    
    # Testing
    test_acc = 0.0
    test_loss = 0.0
    for batch, (images, labels) in enumerate(test_dset.take(ntest_step)):
        loss_value, acc = validation_step(images, labels)
        test_acc += acc/ntest_step
        test_loss += loss_value/ntest_step
    test_loss = hvd.allreduce(test_loss, average=True)
    test_acc = hvd.allreduce(test_acc, average=True)

    tt1 = time.time()
    print('E[%d], train Loss: %.6f, training Acc: %.3f, val loss: %.3f, val Acc: %.3f\t Time: %.3f seconds' % (ep, training_loss, training_acc, test_loss, test_acc, tt1 - tt0))
    metrics['Epochs'].append(ep)
    metrics['train_acc'].append(training_acc.numpy())
    metrics['train_loss'].append(training_loss.numpy())
    metrics['valid_acc'].append(test_acc.numpy())
    metrics['valid_loss'].append(test_loss.numpy())
    metrics['time_per_epochs'].append(tt1 - tt0) 


np.savetxt("metrics.dat", np.array([metrics['Epochs'],metrics['train_acc'], metrics['train_loss'], metrics['valid_acc'], metrics['valid_loss'], metrics['time_per_epochs']]).transpose())
t1 = time.time()
print("Total training time: %s seconds" %(t1 - t0))
