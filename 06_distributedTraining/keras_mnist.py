import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf

# (1) Initializing Horovod
import horovod.tensorflow.keras as hvd
hvd.init()
print("I am rank %s of %s" %(hvd.rank(), hvd.size()))

import numpy
import time
import argparse
parser = argparse.ArgumentParser(description='Horovod',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--device', default='gpu',
                    help='Whether this is running on cpu or gpu')
parser.add_argument('--epochs', default=16, type=int, help='Number of epochs to run')
args = parser.parse_args()

from tensorflow.python.client import device_lib

def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU' or x.device_type == 'CPU']

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    # (2) Pin one GPU to specific horovod worker
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
        
tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(2)
        


# MNIST dataset 
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype(numpy.float32)
x_test  = x_test.astype(numpy.float32)

x_train /= 255.
x_test  /= 255.

y_train = y_train.astype(numpy.int32)
y_test  = y_test.astype(numpy.int32)





# Convolutional model

class MNISTClassifier(tf.keras.models.Model):

    def __init__(self, activation=tf.nn.tanh):
        tf.keras.models.Model.__init__(self)

        self.conv_1 = tf.keras.layers.Conv2D(32, [3, 3], activation='relu')
        self.conv_2 = tf.keras.layers.Conv2D(64, [3, 3], activation='relu')
        self.pool_3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.drop_4 = tf.keras.layers.Dropout(0.25)
        self.dense_5 = tf.keras.layers.Dense(128, activation='relu')
        self.drop_6 = tf.keras.layers.Dropout(0.5)
        self.dense_7 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):

        x = self.conv_1(inputs)
        x = self.conv_2(x)
        x = self.pool_3(x)
        x = self.drop_4(x)
        x = tf.keras.layers.Flatten()(x)
        x = self.dense_5(x)
        x = self.drop_6(x)
        x = self.dense_7(x)

        return x



def train_network_concise(_batch_size, _n_training_epochs, _lr):

    cnn_model = MNISTClassifier()
    # (3) scale the learning rate
    opt = tf.optimizers.Adam(_lr*hvd.size())
    # (4) add Horovod Distributed Optimizer
    opt = hvd.DistributedOptimizer(opt)
    # Specify `experimental_run_tf_function=False`
    cnn_model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=['accuracy'],
                      experimental_run_tf_function=False)
    # (5) Define call back
    callbacks = [
        # broad cast 
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        # Average metric at the end of every epoch
        hvd.callbacks.MetricAverageCallback(),
        # Warmup 
        hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=3, verbose=1, initial_lr=_lr),
    ]
    # (6) save checkpoints only on worker 0
    #if hvd.rank()==0:
    #    callbacks.append(tf.keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))
    verbose=0
    if hvd.rank()==0:
        verbose=1
    x_train_reshaped = numpy.expand_dims(x_train, -1)
    # (7) Adjust the number of steps per epochs
    if (args.device=='cpu'):
        with tf.device('/device:CPU:0'):
            history = cnn_model.fit(x_train_reshaped, y_train, batch_size=_batch_size, epochs=_n_training_epochs, callbacks=callbacks, steps_per_epoch=60000//hvd.size()//_batch_size, verbose=verbose)
    else:
        history = cnn_model.fit(x_train_reshaped, y_train, batch_size=_batch_size, epochs=_n_training_epochs, callbacks=callbacks, steps_per_epoch=60000//hvd.size()//_batch_size, verbose=verbose)
    return history, cnn_model

batch_size = 512
epochs = args.epochs
lr = .01
history, cnn_model = train_network_concise(batch_size, 1, lr)

t0 = time.time()
history, cnn_model = train_network_concise(batch_size, epochs, lr)
t1 = time.time()
if (hvd.rank()==0):
    print("Total time: %s second" %(t1-t0))
