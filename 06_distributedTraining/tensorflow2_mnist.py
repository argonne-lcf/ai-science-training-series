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
#Horovod: import horovod module 

import horovod.tensorflow as hvd
import time

parser = argparse.ArgumentParser(description='TensorFlow MNIST Example')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--epochs', type=int, default=16, metavar='N',
                    help='number of epochs to train (default: 4)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--device', default='cpu',
                    help='Wheter this is running on cpu or gpu')
parser.add_argument('--num_inter', default=2, help='set number inter', type=int)
parser.add_argument('--num_intra', default=0, help='set number intra', type=int)

args = parser.parse_args()
    
# Horovod: initialize Horovod.
hvd.init()
print("I am rank %s of %s" %(hvd.rank(), hvd.size()))

# Horovod: pin GPU to be used to process local rank (one GPU per process)
if args.device == 'cpu':
    tf.config.threading.set_intra_op_parallelism_threads(args.num_intra)
    tf.config.threading.set_inter_op_parallelism_threads(args.num_inter)
else:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

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

nsamples = len(list(dataset))
ntests = len(list(test_dset))
# shuffle the dataset, with shuffle buffer to be 10000
dataset = dataset.repeat().shuffle(10000).batch(args.batch_size)

test_dset  = test_dset.shard(num_shards=hvd.size(), index=hvd.rank()).repeat().batch(args.batch_size)

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

# Horovod: adjust learning rate based on number of GPUs.
opt = tf.optimizers.Adam(args.lr * hvd.size())

checkpoint_dir = './checkpoints/tf2_mnist'
checkpoint = tf.train.Checkpoint(model=mnist_model, optimizer=opt)


@tf.function
def training_step(images, labels, first_batch):
    with tf.GradientTape() as tape:
        probs = mnist_model(images, training=True)
        pred = tf.math.argmax(probs, axis=1)
        equality = tf.math.equal(pred, labels)
        accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))
        loss_value = loss(labels, probs)
    
    # Horovod: add Horovod Distributed GradientTape.
    tape = hvd.DistributedGradientTape(tape)

    grads = tape.gradient(loss_value, mnist_model.trainable_variables)
    opt.apply_gradients(zip(grads, mnist_model.trainable_variables))

    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    #
    # Note: broadcast should be done after the first gradient step to ensure optimizer
    # initialization.
    if first_batch:
        hvd.broadcast_variables(mnist_model.variables, root_rank=0)
        hvd.broadcast_variables(opt.variables(), root_rank=0)

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
# Horovod: adjust number of steps based on number of GPUs.
nstep = nsamples//hvd.size()//args.batch_size
ntest_step = ntests//args.batch_size//hvd.size()
t0 = time.time()
for ep in range(args.epochs):
    running_loss = 0.0
    running_acc = 0.0
    tt0 = time.time()
    # Training
    for batch, (images, labels) in enumerate(dataset.take(nstep)):
        loss_value, acc = training_step(images, labels, (batch == 0) and (ep == 0))
        running_acc += acc/nstep
        running_loss += loss_value/nstep
        if hvd.rank() == 0 and batch % 100 == 0: 
            checkpoint.save(checkpoint_dir)
    total_loss = hvd.allreduce(running_loss, average=True)
    total_acc = hvd.allreduce(running_acc, average=True)
    tt1 = time.time()
    # Testing
    test_acc = 0.0
    test_loss = 0.0
    for batch, (images, labels) in enumerate(test_dset.take(ntest_step)):
        loss_value, acc = validation_step(images, labels)
        test_acc += acc/ntest_step
        test_loss += loss_value/ntest_step
    
    total_test_loss = hvd.allreduce(test_loss, average=True)
    total_test_acc = hvd.allreduce(test_acc, average=True)
    tt1 = time.time()
    if (hvd.rank()==0):
        print('Epoch - %d, \ttrain Loss: %.6f, \t training Acc: %.6f, \tval loss: %.6f, \t val Acc: %.6f\t Time: %.6f seconds' % (ep, total_loss, total_acc, total_test_loss, total_test_acc, tt1 - tt0))

# Horovod: save checkpoints only on worker 0 to prevent other workers from
# corrupting it.
if hvd.rank() == 0:
    checkpoint.save(checkpoint_dir)
t1 = time.time()
if hvd.rank()==0:
    print("Total training time: %s seconds" %(t1 - t0))
