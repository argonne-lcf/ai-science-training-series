import sys, os
import time,math

# This limits the amount of memory used:
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"
# This control parallelism in Tensorflow
parallel_threads = 128
# This controls how many batches to prefetch
prefetch_buffer_size = 8 # tf.data.AUTOTUNE
os.environ['OMP_NUM_THREADS'] = str(parallel_threads)
num_parallel_readers = parallel_threads

import argparse
parser = argparse.ArgumentParser(description='Horovod',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--device', default='gpu',
                    help='Whether this is running on cpu or gpu')
parser.add_argument('--epochs', default=10, type=int, help='Number of epochs to run')
parser.add_argument('--num_steps', default=10, type=int, help="Number of steps")
args = parser.parse_args()

# how many training steps to take during profiling
num_steps = args.num_steps
use_profiler = True

import tensorflow as tf
from tensorflow.python.profiler import trace

#########################################################################
# Here's the Residual layer from the first half again:
#########################################################################
class ResidualLayer(tf.keras.Model):

    def __init__(self, n_filters):
        # tf.keras.Model.__init__(self)
        super(ResidualLayer, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(
            filters     = n_filters,
            kernel_size = (3,3),
            padding     = "same"
        )

        self.norm1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(
            filters     = n_filters,
            kernel_size = (3,3),
            padding     = "same"
        )

        self.norm2 = tf.keras.layers.BatchNormalization()

    def call(self, inputs):

        x = inputs

        output1 = self.norm1(self.conv1(inputs))

        output1 = tf.keras.activations.relu(output1)

        output2 = self.norm2(self.conv2(output1))

        return tf.keras.activations.relu(output2 + x)

#########################################################################
# Here's layer that does a spatial downsampling:
#########################################################################
class ResidualDownsample(tf.keras.Model):

    def __init__(self, n_filters):
        # tf.keras.Model.__init__(self)
        super(ResidualDownsample, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(
            filters     = n_filters,
            kernel_size = (3,3),
            padding     = "same",
            strides     = (2,2)
        )

        self.identity = tf.keras.layers.Conv2D(
            filters     = n_filters,
            kernel_size = (1,1),
            strides     = (2,2),
            padding     = "same"
        )

        self.norm1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(
            filters     = n_filters,
            kernel_size = (3,3),
            padding     = "same"
        )

        self.norm2 = tf.keras.layers.BatchNormalization()

    @tf.function
    def call(self, inputs):

        x = self.identity(inputs)
        output1 = self.norm1(self.conv1(inputs))
        output1 = tf.keras.activations.relu(output1)

        output2 = self.norm2(self.conv2(output1))

        return tf.keras.activations.relu(output2 + x)


#########################################################################
# Armed with that, let's build ResNet (this particular one is called ResNet34)
#########################################################################

class ResNet34(tf.keras.Model):

    def __init__(self):
        super(ResNet34, self).__init__()

        self.conv_init = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                filters     = 64,
                kernel_size = (7,7),
                strides     = (2,2),
                padding     = "same",
                use_bias    = False
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding="same")

        ])

        self.residual_series_1 = tf.keras.Sequential([
            ResidualLayer(64),
            ResidualLayer(64),
            ResidualLayer(64),
        ])

        # Increase the number of filters:
        self.downsample_1 = ResidualDownsample(128)

        self.residual_series_2 = tf.keras.Sequential([
            ResidualLayer(128),
            ResidualLayer(128),
            ResidualLayer(128),
        ])

        # Increase the number of filters:
        self.downsample_2 = ResidualDownsample(256)

        self.residual_series_3 = tf.keras.Sequential([
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
        ])

        # Increase the number of filters:
        self.downsample_3 = ResidualDownsample(512)


        self.residual_series_4 = tf.keras.Sequential([
            ResidualLayer(512),
            ResidualLayer(512),
        ])

        self.final_pool = tf.keras.layers.AveragePooling2D(
            pool_size=(8,8)
        )

        self.flatten = tf.keras.layers.Flatten()
        self.classifier = tf.keras.layers.Dense(1000)

    @tf.function
    def call(self, inputs):

        x = self.conv_init(inputs)

        x = self.residual_series_1(x)


        x = self.downsample_1(x)


        x = self.residual_series_2(x)

        x = self.downsample_2(x)

        x = self.residual_series_3(x)

        x = self.downsample_3(x)


        x = self.residual_series_4(x)


        x = self.final_pool(x)

        x = self.flatten(x)

        logits = self.classifier(x)

        return logits




@tf.function()
def calculate_accuracy(logits, labels):
    # We calculate top1 accuracy only here:
    selected_class = tf.argmax(logits, axis=1)

    correct = tf.cast(selected_class, tf.float32) == tf.cast(labels, tf.float32)

    return tf.reduce_mean(tf.cast(correct, tf.float32))


@tf.function()
def calculate_loss(logits, labels):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
    return tf.reduce_mean(loss)

@tf.function()
def training_step(network, optimizer, images, labels):
    with tf.GradientTape() as tape:
        logits = network(images)
        loss = calculate_loss(logits, labels)

    gradients = tape.gradient(loss, network.trainable_variables)

    optimizer.apply_gradients(zip(gradients, network.trainable_variables))

    accuracy = calculate_accuracy(logits, labels)

    return loss, accuracy

@trace.trace_wrapper('train_epoch')
def train_epoch(i_epoch, step_in_epoch, train_ds, val_ds, network, optimizer, BATCH_SIZE, checkpoint):
    # Here is our training loop!

    steps_per_epoch = int(1281167 / BATCH_SIZE)
    steps_validation = int(50000 / BATCH_SIZE)

    # added for profiling
    if use_profiler:
        print('start profiler')
        tf.profiler.experimental.start('logdir/m%03d_w%02d_p%02d' % (parallel_threads,num_parallel_readers,prefetch_buffer_size))
    
    start = time.time()
    i = 0
    sum = 0.
    sum2 = 0.
    for train_images, train_labels in train_ds.take(steps_per_epoch):
        if step_in_epoch > steps_per_epoch: break
        else: step_in_epoch.assign_add(1)

        # Peform the training step for this batch
        loss, acc = training_step(network, optimizer, train_images, train_labels)
        end = time.time()
        images_per_second = BATCH_SIZE / (end - start)
        if i > 0: # skip the first measurement because it includes compile time
            sum += images_per_second
            sum2 += images_per_second * images_per_second
        print(f"Finished step {step_in_epoch.numpy()} of {steps_per_epoch} in epoch {i_epoch.numpy()},loss={loss:.3f}, acc={acc:.3f} ({images_per_second:.3f} img/s).")
        start = time.time()
        # added for profiling to stop after some steps
        i += 1
        if i > num_steps and use_profiler: break
    
    # added for profiling to stop after some steps
    if use_profiler:
        print('stop profiler')
        i = i - 1
        mean_rate = sum / i
        stddev_rate = math.sqrt( sum2/i - mean_rate * mean_rate )
        print(f'mean image/s = {mean_rate:8.2f}   standard deviation: {stddev_rate:8.2f}')
        tf.profiler.experimental.stop()
        sys.exit(0)

    # Save the network after every epoch:
    checkpoint.save("resnet34/model")
    
    # Compute the validation accuracy:
    mean_accuracy = None
    for val_images, val_labels in val_ds.take(steps_validation):
        logits = network(val_images)
        accuracy = calculate_accuracy(logits, val_labels)
        if mean_accuracy is None:
            mean_accuracy = accuracy
        else:
            mean_accuracy += accuracy

    mean_accuracy /= steps_validation

    print(f"Validation accuracy after epoch {i_epoch.numpy()}: {mean_accuracy:.4f}.")


# @trace.trace_wrapper('prepare_data_loader')
def prepare_data_loader(BATCH_SIZE):

    tf.config.threading.set_inter_op_parallelism_threads(parallel_threads)
    tf.config.threading.set_intra_op_parallelism_threads(parallel_threads)
    print('threading set: ',tf.config.threading.get_inter_op_parallelism_threads(),tf.config.threading.get_intra_op_parallelism_threads())

    print("Parameters set, preparing dataloading")
    #########################################################################
    # Here's the part where we load datasets:
    import json


    # What's in this function?  Tune in next week ...
    from ilsvrc_dataset import get_datasets


    class FakeHvd:

        def size(self): return 1

        def rank(self): return 0


    with open("ilsvrc.json", 'r') as f:
        config = json.load(f)

    config['data']['batch_size'] = BATCH_SIZE
    config['data']['num_parallel_readers'] = num_parallel_readers
    config['data']['prefetch_buffer_size'] = prefetch_buffer_size 

    print(json.dumps(config, indent=4))

    config['hvd'] = FakeHvd()

    train_ds, val_ds = get_datasets(config)

    options = tf.data.Options()
    options.threading.private_threadpool_size = parallel_threads
    train_ds = train_ds.with_options(options)
    val_ds = val_ds.with_options(options)

    print("Datasets ready, creating network.")
    #########################################################################

    return train_ds, val_ds


def main():
    #########################################################################
    # Here's some configuration:
    #########################################################################
    BATCH_SIZE = 256
    N_EPOCHS = 10

    train_ds, val_ds = prepare_data_loader(BATCH_SIZE)


    example_images, example_labels = next(iter(train_ds.take(1)))


    print("Initial Image size: ", example_images.shape)
    network = ResNet34()

    output = network(example_images)
    print("output shape:", output.shape)

    print(network.summary())

    epoch = tf.Variable(initial_value=tf.constant(0, dtype=tf.dtypes.int64), name='epoch')
    step_in_epoch = tf.Variable(
        initial_value=tf.constant(0, dtype=tf.dtypes.int64),
        name='step_in_epoch')


    # We need an optimizer.  Let's use Adam:
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    checkpoint = tf.train.Checkpoint(
        network       = network,
        optimizer     = optimizer,
        epoch         = epoch,
        step_in_epoch = step_in_epoch)

    # Restore the model, if possible:
    latest_checkpoint = tf.train.latest_checkpoint("resnet34/")
    if latest_checkpoint:
        checkpoint.restore(latest_checkpoint)

    while epoch < N_EPOCHS:
        train_epoch(epoch, step_in_epoch, train_ds, val_ds, network, optimizer, BATCH_SIZE, checkpoint)
        epoch.assign_add(1)
        step_in_epoch.assign(0)

if __name__ == "__main__":
    main()
