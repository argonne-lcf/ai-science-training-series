#!/bin/env python

# Example of large scale dataset processing in Tensorflow.
# Processes the ImageNet dataset into a one-hot classificaiton
# dataset.
#
# ImageNet is a mixture of images, with 1000 labeled classes.
# Each image can have one or more class objects.
# The annotations for each image includes class ID and bounding
# box dimensions. The functions below use these bounding boxes
# to chop up the original images to create single images
# corresponding to single class labels. This simplifies the
# network needed to label the data, but effects the final
# network accuracy.
#
# questions? Taylor Childers, jchilders@anl.gov

import os
import glob

import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
import tensorflow as tf
import horovod.tensorflow as hvd
from tensorflow.python.profiler import trace
import numpy as np
import xml.etree.ElementTree as ET
# import horovod.tensorflow as hvd
# hvd.init()

RANK = hvd.rank()
SIZE = hvd.size()

log = logging.getLogger(__name__)

# these are initialized in the get_datasets function and used later
labels_hash = None
crop_size = None


@trace.trace_wrapper('get_datasets')
def get_datasets(config):
    # these global variables will be initizlized
    global labels_hash,crop_size

    # set the crop size of the output images, e.g. [256,256]
    crop_size = tf.constant(config['data']['crop_image_size'])
    # these are paths to text files containing a list, one entry per line,
    # of all the training JPEGs and testing JPEGs
    # it's assumed the full path to the JPEGs is like this:
    # /.../ILSVRC/Data/CLS-LOC/train/n02437312/n02437312_8688.JPEG
    # because the class label comes from the last folder text.
    train_filelist = config['data']['train_filelist']
    test_filelist = config['data']['test_filelist']

    assert os.path.exists(train_filelist)
    assert os.path.exists(test_filelist)

    # this function uses that class label from the filename path
    # and builds a map from the string labels like the above "n02537312"
    # to a unique integer value 0-999. This is more suitable for
    # network classifciation than a string.
    labels_hash = get_label_tables(config, train_filelist)

    # this function creates the tf.dataset.Dataset objects for each list
    # of input JPEGs.
    train_ds = build_dataset_from_filelist(config,train_filelist)
    valid_ds = build_dataset_from_filelist(config,test_filelist)

    return train_ds, valid_ds


## Create a hash table for labels from string to int 
@trace.trace_wrapper('get_label_tables')
def get_label_tables(config, train_filelist):
    # get the first filename
    with open(train_filelist) as file:
        filepath = file.readline().strip()
    # parse the filename to extract the "n02537312" string
    # from the full path which is assumed to be similar to this
    # /.../ILSVRC/Data/CLS-LOC/train/n02437312/n02437312_8688.JPEG
    # and convert that string to a unique value from 0-999

    # this extracts the path up to: /.../ILSVRC/Data/CLS-LOC/train/
    label_path = '/'.join(filepath.split('/')[:-2])
    # this globs for all the directories like "n02537312" to get 
    # list of the string labels
    labels = glob.glob(label_path + os.path.sep + '*')
    if config['hvd'].rank() == 0:
       print(f'num labels: {len(labels)}')
    # this removes the leading path from the label directories
    labels = [os.path.basename(i) for i in labels]
    # create a list of integers as long as the number of labels
    hash_values = tf.range(len(labels))
    # convert python list of strings to a tensorflow vector
    hash_keys = tf.constant(labels, dtype=tf.string)
    # build a key-value lookup using Tensorflow tools
    labels_hash_init = tf.lookup.KeyValueTensorInitializer(hash_keys, hash_values)
    # build a lookup table based on those key-value pairs (returns -1 for undefined keys)
    labels_hash = tf.lookup.StaticHashTable(labels_hash_init, -1)
 
    return labels_hash


# take a config dictionary and a path to a filelist
# return a tf.dataset.Dataset object that will iterate over the JPEGs in filelist
@trace.trace_wrapper('build_dataset_from_filelist')
def build_dataset_from_filelist(config,filelist_filename):
    # if config['hvd'].rank() == 0:
    if RANK == 0:
        print(f'build dataset {filelist_filename}')

    dc = config['data']
    # if running horovod(MPI) need to shard the dataset based on rank
    # numranks = 1
    numranks = SIZE
    #    numranks = config['hvd'].size()

    # loading full filelist
    filelist = []
    with open(filelist_filename) as file:
        for line in file:
            filelist.append(line.strip())

    # provide user with estimated batches in epoch
    batches_per_rank = int(len(filelist) / dc['batch_size'] / numranks)
    if RANK == 0:
        log.info(
            f'input filelist contains {len(filelist)} files'
            f', estimated batches per rank {batches_per_rank}'
        )

    # convert python list to tensorflow vector object
    filelist = tf.data.Dataset.from_tensor_slices(filelist)
    # if using horovod (MPI) shard the data based on total ranks (size) and rank
    # if config['hvd']:
    filelist = filelist.shard(SIZE, RANK)
    # shuffle the data, set shuffle buffer (needs to be large), and reshuffle after each epoch
    filelist = filelist.shuffle(dc['shuffle_buffer'],reshuffle_each_iteration=dc['reshuffle_each_iteration'])

    # run 'load_image_label_bb' on each input image file, process multiple files in parallel
    # this function opens the JPEG, converts it to a tensorflow vector and gets the truth class label
    ds = filelist.map(load_image_label_bb,
                      num_parallel_calls=config['data']['num_parallel_readers'])

    # unbatch called because some JPEGs result in more than 1 image returned
    ds = ds.apply(tf.data.Dataset.unbatch)

    # batch the data
    ds = ds.batch(dc['batch_size'])

    # setup a pipeline that pre-fetches images before they are needed (keeps CPU busy)
    ds = ds.prefetch(buffer_size=config['data']['prefetch_buffer_size'])  

    return ds


# this function parses the image path, uses the label hash to convert the string
# label in the path to a numerical label, decodes the input JPEG, and returns
# the input image and label
@trace.trace_wrapper('load_image_label_bb')
def load_image_label_bb(image_path):
    # for each JPEG, there is an Annotation file that contains a list of
    # classes contained in the image and a bounding box for each object.
    # However, some images contain a single class, in which case the
    # dataset contains no annotation file which is annoying, but...
    # Images with multiple objects per file are always the same class.
    label = tf.strings.split(image_path, os.path.sep)[-2]
    annot_path = tf.strings.regex_replace(image_path,'Data','Annotations')
    annot_path = tf.strings.regex_replace(annot_path,'JPEG','xml')

    # open the annotation file and retrieve the bounding boxes and indices
    bounding_boxes,box_indices = tf.py_function(  # type:ignore
        get_bounding_boxes,
        [annot_path],
        [tf.float32, tf.int32]
    )

    # open the JPEG
    img = tf.io.read_file(image_path)
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # add batching index [batch,width,height,channel]
    img = tf.expand_dims(img,0)

    # create individual images based on bounding boxes
    imgs = tf.image.crop_and_resize(img,bounding_boxes,box_indices,crop_size)

    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    imgs = tf.image.convert_image_dtype(imgs, tf.float16)
    # resize the image to the desired size. networks don't like variable sized arrays.
    imgs = tf.image.resize(imgs, crop_size)
    # convert string label to numerical label
    label = labels_hash.lookup(label)
    # duplicate labels to match the number of images created from bounding boxes
    labels = tf.fill([tf.shape(imgs)[0]],label)
    # return images and labels
    return imgs, labels


# this function opens the annotation XML file and parses the contents
# the contents include a list of objects in the JPEG, a label and
# bounding box for each object
@trace.trace_wrapper('get_bounding_boxes')
def get_bounding_boxes(filename):
    filename = bytes.decode(filename.numpy())
    try:
        with tf.profiler.experimental.Trace('read_xml'):
            tree = ET.parse(filename)
            root = tree.getroot()

        img_size = root.find('size')
        img_width = int(img_size.find('width').text)
        img_height = int(img_size.find('height').text)
        # img_depth = int(img_size.find('depth').text)

        objs = root.findall('object')
        bndbxs = []
        # label = None
        for object in objs:
            # label = object.find('name').text
            bndbox = object.find('bndbox')
            bndbxs.append([
                float(bndbox.find('ymin').text) / (img_height - 1),
                float(bndbox.find('xmin').text) / (img_width - 1),
                float(bndbox.find('ymax').text) / (img_height - 1),
                float(bndbox.find('xmax').text) / (img_width - 1)
            ])
    except FileNotFoundError:
        bndbxs = [[0,0,1,1]]

    return np.asarray(bndbxs,float),np.zeros(len(bndbxs))



if __name__ == '__main__':
    # parse command line
    import argparse,json,time
    parser = argparse.ArgumentParser(description='test this')
    parser.add_argument('-c', '--config', dest='config_filename',
                        help='configuration filename in json format',
                        required=True)
    parser.add_argument('-l','--logdir', dest='logdir',
                        help='log output directory',default='logdir')
    parser.add_argument('-n','--nsteps', dest='nsteps',
                        help='number of steps to run',default=10,type=int)
    parser.add_argument('--interop',type=int,help='set Tensorflow "inter_op_parallelism_threads" session config varaible ',default=None)
    parser.add_argument('--intraop',type=int,help='set Tensorflow "intra_op_parallelism_threads" session config varaible ',default=None)

    args = parser.parse_args()

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

        print("GPUs Available: %s" % tf.config.get_visible_devices('GPU'))

    # parse config file
    config = json.load(open(args.config_filename))
    config['hvd'] = hvd

    # define some parallel processing sizes
    if args.interop is not None:
        tf.config.threading.set_inter_op_parallelism_threads(args.interop)
    if args.intraop is not None:
        tf.config.threading.set_intra_op_parallelism_threads(args.intraop)

    # use the tensorflow profiler here
    if hvd.rank() == 0:
        tf.profiler.experimental.start(args.logdir)
    # call function to build dataset objects
    # both of the returned objects are tf.dataset.Dataset objects
    trainds, testds = get_datasets(config)
    # can iterate over a dataset object
    trainds = iter(trainds)
    start = time.time()
    for i in range(args.nsteps):
        # profile data pipeline
        with tf.profiler.experimental.Trace('train_%02d' % i, step_num=i, _r=1):
            inputs,labels = next(trainds)
    # measure performance in images per second
    duration = time.time() - start
    if hvd.rank() == 0:
        tf.profiler.experimental.stop()
    images = config['data']['batch_size'] * args.nsteps
    if hvd.rank() == 0:
        print('imgs/sec = %5.2f' % ((images/duration)*hvd.size()))
