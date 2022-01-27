# Turn of TF logs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf

import numpy
import time


import argparse
parser = argparse.ArgumentParser(description='Horovod',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--device', default='gpu',
                    help='Wheter this is running on cpu or gpu')
args = parser.parse_args()


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


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

    cnn_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    
    x_train_reshaped = numpy.expand_dims(x_train, -1)

    history = cnn_model.fit(x_train_reshaped, y_train, batch_size=_batch_size, epochs=_n_training_epochs)
    return history, cnn_model

batch_size = 512
epochs = 16
lr = .01
t0 = time.time()
history, cnn_model = train_network_concise(batch_size, epochs, lr)
t1 = time.time()
print("Total time is %s sec" %(t1-t0))
