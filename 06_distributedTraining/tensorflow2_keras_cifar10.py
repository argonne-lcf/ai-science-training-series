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
import time
# Horovod: initialize Horovod.

try:
    import horovod.tensorflow.keras as hvd
    with_hvd=True
except:
    with_hvd=False
    class Hvd:
        def init():
            print("I could not find Horovod package, will do things sequentially")
        def rank():
            return 0
        def size():
            return 1
    hvd=Hvd; 

hvd.init()
t0 = time.time()
parser = argparse.ArgumentParser(description='TensorFlow MNIST Example')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--device', default='cpu',
                    help='Wheter this is running on cpu or gpu')
parser.add_argument('--num_inter', default=2, help='set number inter', type=int)
parser.add_argument('--num_intra', default=0, help='set number intra', type=int)

args = parser.parse_args()

# Horovod: pin GPU to be used to process local rank (one GPU per process)

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

(cifar10_images, cifar10_labels), _ = \
    tf.keras.datasets.cifar10.load_data()
(train_images, train_labels), (test_images, test_labels) =    tf.keras.datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(cifar10_images[..., tf.newaxis] / 255.0, tf.float32),
             tf.cast(cifar10_labels, tf.int64))
)
nsamples = len(list(dataset))
dataset = dataset.repeat().shuffle(10000).batch(args.batch_size)


# Horovod: adjust learning rate based on number of GPUs.
opt = tf.optimizers.Adam(args.lr * hvd.size())

# Horovod: add Horovod DistributedOptimizer.
if (with_hvd):
    opt = hvd.DistributedOptimizer(opt)

input_shape=(32, 32, 3)
num_classes = 10
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import *

#model = Sequential()

model = Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

'''
model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
'''
if (hvd.rank()==0):
    model.summary()
cifar10_model = model;
'''
cifar10_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
    tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])
'''

# Horovod: Specify `experimental_run_tf_function=False` to ensure TensorFlow
# uses hvd.DistributedOptimizer() to compute gradients.
cifar10_model = tf.keras.applications.resnet50.ResNet50(include_top=True,
                                                        input_tensor=None, input_shape=(32, 32, 3),
                                                        pooling=None, classes=10, weights=None)
cifar10_model.compile(loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                    optimizer=opt,
                    metrics=['accuracy'],
                    experimental_run_tf_function=False)
#print(cifar10_model.summary())
if (with_hvd):
    callbacks = [
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        
        # Horovod: average metrics among workers at the end of every epoch.
        #
        # Note: This callback must be in the list before the ReduceLROnPlateau,
        # TensorBoard or other metrics-based callbacks.
        hvd.callbacks.MetricAverageCallback(),
        
        # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
        # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
        # the first three epochs. See https://arxiv.org/abs/1706.02677 for details.
        hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=3, verbose=1, initial_lr = args.lr),
    ]
else:
    callbacks=[]
    # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.

if hvd.rank() == 0:
    callbacks.append(tf.keras.callbacks.ModelCheckpoint('./checkpoints/keras_cifar10-{epoch}.h5'))

# Horovod: write logs on worker 0.
verbose = 1 if hvd.rank() == 0 else 0

# Train the model.
# Horovod: adjust number of steps based on number of GPUs.
#cifar10_model.fit(dataset, steps_per_epoch=nsamples // hvd.size() // args.batch_size, callbacks=callbacks, epochs=args.epochs, verbose=verbose)

history = cifar10_model.fit(train_images, train_labels, epochs=args.epochs, steps_per_epoch=nsamples//hvd.size()//args.batch_size, callbacks=callbacks, verbose=verbose, validation_data=(test_images, test_labels), shuffle = True)

t1 = time.time()
if (hvd.rank()==0):
    print("Total training time: %s seconds" %(t1 - t0))
