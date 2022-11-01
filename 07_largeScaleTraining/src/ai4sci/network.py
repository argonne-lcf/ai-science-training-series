"""
ai4sci/horovod/tensorflow
"""
from __future__ import absolute_import, annotations, division, print_function
import sys
import os
import time
from pathlib import Path

# this limits the amount of memory used
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
PARALLEL_THREADS = 128
PREFETCH_BUFFER_SIZE = 8
os.environ['OMP_NUM_THREADS'] = str(PARALLEL_THREADS)
NUM_PARALLEL_READERS = PARALLEL_THREADS


HERE = Path(os.path.abspath(__file__)).parent

import tensorflow as tf


class ConvNet(tf.keras.Model):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation='softmax'),
        ])

    @tf.function
    def call(self, inputs):
        return self.model(inputs)


class ResidualLayer(tf.keras.Model):
    def __init__(self, nfilters: int) -> None:
        super(ResidualLayer, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=nfilters,
            kernel_size=(3, 3),
            padding='same'
        )
        self.norm1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(
            filters=nfilters,
            kernel_size=(3, 3),
            padding='same',
        )

        self.norm2 = tf.keras.layers.BatchNormalization()

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = inputs
        out1 = self.norm1(self.conv1(inputs))
        out1 = tf.keras.activations.relu(out1)
        out2 = self.norm2(self.conv2(out1))
        assert out2 is not None
        return tf.keras.activations.relu(out2 + x)


class ResidualDownSample(tf.keras.Model):
    def __init__(self, nfilters: int) -> None:
        super(ResidualDownSample, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=nfilters,
            kernel_size=(3, 3),
            padding='same',
            strides=(2, 2)
        )
        self.identity = tf.keras.layers.Conv2D(
            filters=nfilters,
            kernel_size=(1, 1),
            strides=(2, 2),
            padding='same',
        )
        self.norm1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(
            filters=nfilters,
            padding='same',
            kernel_size=(3, 3),
        )
        self.norm2 = tf.keras.layers.BatchNormalization()

    @tf.function
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = self.identity(inputs)
        out1 = self.norm1(self.conv1(inputs))
        out1 = tf.keras.activations.relu(out1)
        out2 = self.norm2(self.conv2(out1))
        assert out2 is not None
        return tf.keras.activations.relu(out2 + x)


class ResNet34(tf.keras.Model):
    def __init__(self):
        super(ResNet34, self).__init__()
        self.conv_init = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=(7, 7),
                strides=(2, 2),
                padding='same',
                use_bias=False,
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(
                pool_size=(3, 3),
                strides=(2, 2),
                padding='same'
            )
        ])
        self.residual_series1 = tf.keras.Sequential([
            ResidualLayer(64),
            ResidualLayer(64),
            ResidualLayer(64)
        ])
        # Increase the number of filters
        self.downsample1 = ResidualDownSample(128)
        self.residual_series2 = tf.keras.Sequential([
            ResidualLayer(128),
            ResidualLayer(128),
            ResidualLayer(128)
        ])
        self.downsample2 = ResidualDownSample(256)
        self.residual_series3 = tf.keras.Sequential([
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
        ])
        self.downsample3 = ResidualDownSample(512)
        self.residual_series4 = tf.keras.Sequential([
            ResidualLayer(512),
            ResidualLayer(512)
        ])
        self.final_pool = tf.keras.layers.AveragePooling2D(
            pool_size=(8, 8),
        )
        self.flatten = tf.keras.layers.Flatten()
        self.classifier = tf.keras.layers.Dense(1000)

    @tf.function
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = self.conv_init(inputs)
        x = self.residual_series1(x)
        x = self.downsample1(x)
        x = self.residual_series2(x)
        x = self.downsample2(x)
        x = self.residual_series3(x)
        x = self.downsample3(x)
        x = self.residual_series4(x)
        x = self.final_pool(x)
        x = self.flatten(x)
        logits = self.classifier(x)

        assert logits is not None
        return logits

@tf.function
def calculate_accuracy(logits, labels):
    selected_class = tf.argmax(logits, axis=1)
    correct = tf.cast(selected_class, tf.float32) == tf.cast(labels, tf.float32)
    return tf.reduce_mean(tf.cast(correct, tf.float32))


@tf.function
def calculate_loss(logits, labels):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
    return tf.reduce_mean(loss)


@tf.function
def training_step(network, optimizer, images, labels):
    with tf.GradientTape() as tape:
        logits = network(images)
        loss = calculate_loss(logits, labels)

    gradients = tape.gradient(loss, network.trainable_variables)
    optimizer.apply_gradients(zip(gradients, network.trainable_variables))

    acc = calculate_accuracy(logits, labels)

    return loss, acc


def train_epoch(
        i_epoch,
        step_in_epoch,
        train_ds,
        val_ds,
        network,
        optimizer,
        BATCH_SIZE,
        checkpoint,
) -> None:
    steps_per_epoch = int(1281167 / BATCH_SIZE)
    steps_validation = int(50000 / BATCH_SIZE)
    start = time.time()
    for train_images, train_labels in train_ds.take(steps_per_epoch):
        if step_in_epoch > steps_per_epoch:
            break
        else:
            step_in_epoch.assign_add(1)

        loss, acc = training_step(  # type:ignore
            network,
            optimizer,
            train_images,
            train_labels
        )
        end = time.time()
        images_per_second = BATCH_SIZE / (end - start)
        prefix = (
            f'Finished step {step_in_epoch.numpy()} of {steps_per_epoch}'
            f' in epoch {i_epoch.numpy()}'
        )
        stats = f'loss={loss:.3f}, acc={acc:.3f}, ({images_per_second:.3f} img/s)'
        print(', '.join([prefix, stats]))
        start = time.time()

    # Save the network after every epoch
    checkpoint.save('resnet34/model')

    # Compute the validation accuracy
    mean_acc = 0.0
    for idx, (val_images, val_labels) in enumerate(val_ds.take(steps_validation)):
        logits = network(val_images)
        acc = calc_acc(logits, val_labels)  # type:ignore
        if idx == 0:
            mean_acc = acc
        else:
            mean_acc += acc

    mean_acc /= steps_validation
    print(f'Validation acc after epoch {i_epoch.numpy()}: {mean_acc:.4f}')


def prepare_data_loader(BATCH_SIZE):
    # tf.config.threading.set_inter_op_parallelism_threads(8)
    # tf.config.threading.set_intra_op_parallelism_threads(8)
    import json
    from ai4sci.ilsvrc_dataset import get_datasets

    config = {}
    config_file = Path(HERE).joinpath('ilsvrc.json')
    if config_file.exists():
        with open(config_file.as_posix(), 'r') as f:
            config = json.load(f)

    config['data']['batch_size'] = BATCH_SIZE
    config['data']['num_parallel_readers'] = NUM_PARALLEL_READERS
    config['data']['prefetch_buffer_size'] = PREFETCH_BUFFER_SIZE
    print(json.dumps(config, indent=4))
    import horovod.tensorflow as hvd
    config['hvd'] = hvd
    train_ds, val_ds = get_datasets(config)
    options = tf.data.Options()
    options.threading.private_threadpool_size = PARALLEL_THREADS
    train_ds = train_ds.with_options(options)
    val_ds = val_ds.with_options(options)
    return train_ds, val_ds

