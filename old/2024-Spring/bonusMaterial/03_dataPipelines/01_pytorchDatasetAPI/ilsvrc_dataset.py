#!/bin/env python

# Example of large scale dataset processing in PyTorch.
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

from PIL import Image as PIL_Image
import torch
import torchvision
import logging,os,glob,time
import numpy as np
import xml.etree.ElementTree as ET
logger = logging.getLogger(__name__)


# these are initialized in the get_datasets function and used later
labels_hash = None
crop_size = None


class Dataset(torch.utils.data.Dataset):
    def __init__(self,filelist,config):
        self.config         = config
        self.resize_shape   = config['data']['crop_image_size']
        self.filelist       = filelist
        self.len            = len(self.filelist)

        self.labels_hash    = self.get_label_tables(config['data']['train_filelist'])

    ## Create a hash table for labels from string to int 
    @staticmethod
    def get_label_tables(filelist):

        # get the first filename
        with open(filelist) as file:
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
        logger.info(f'num labels: {len(labels)}')
        # this removes the leading path from the label directories
        labels = [os.path.basename(i) for i in labels]
        
        # create map from text to number
        labels_hash = {}
        for i,label in enumerate(labels):
            labels_hash[label] = i

        return labels_hash

    @classmethod
    def from_filelist(cls,filelist_filename,config):
        filelist = []
        with open(filelist_filename) as file:
            for line in file:
                filelist.append(line.strip())

        if filelist[-1] == '':
            filelist.pop()
        return cls(filelist,config)

    def __getitem__(self,index):
        filename = self.filelist[index]
        
        imgs, labels = self.read_jpeg(filename)

        return imgs, labels

    def __len__(self):
        return self.len

    def read_jpeg(self,filename):

        try:
            label = filename.split('/')[-2]
            
            annotation_filename = filename.replace('Data','Annotations').replace('JPEG','xml')

            # open the JPEG
            img = PIL_Image.open(filename).convert(mode='RGB')
            # convert to a tensor with NCHW
            #img = torchvision.transforms.functional.to_tensor(img)

            # open the annotation file and retrieve the bounding boxes and indices
            bounding_boxes = self.get_bounding_boxes(annotation_filename)
            
            # crop based on bonuding boxes and resize to same size
            if bounding_boxes is not None:
                img = self.crop_and_resize(img,bounding_boxes,self.resize_shape)
            else:
                img = torchvision.transforms.Resize(self.resize_shape)(img)
                img = torchvision.transforms.functional.to_tensor(img)

            imgs = torch.unsqueeze(img,0)

            # convert string label to numerical label
            label = self.labels_hash[label]
            # duplicate labels to match the number of images created from bounding boxes
            labels = torch.full([imgs.shape[0]],label)
            # return images and labels
            # logger.info('img = %s  imgs = %s',img.shape,imgs.shape)
            return imgs, labels
        except:
            logger.exception('filename: %s',filename)
            raise


    @staticmethod
    def crop_and_resize(image,bounding_boxes,resize_shape):
        timg = torchvision.transforms.functional.to_tensor(image)
        imgs = torch.zeros(len(bounding_boxes),timg.shape[0],resize_shape[0],resize_shape[1])
        for i in range(bounding_boxes.shape[0]):
            bb = bounding_boxes[i]
            #img = image[...,bb[0]:bb[2],bb[1]:bb[3]]
            timg = torchvision.transforms.Resize(resize_shape)(image)
            imgs[i,...] = torchvision.transforms.functional.to_tensor(timg)
        return imgs[0]



    @staticmethod
    def get_bounding_boxes(filename):

        logger.debug(filename)
        try:
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
                    int(bndbox.find('ymin').text),
                    int(bndbox.find('xmin').text),
                    int(bndbox.find('ymax').text),
                    int(bndbox.find('xmax').text)
                ])

            return np.asarray(bndbxs,np.int)
        except FileNotFoundError:
            return None




def get_datasets(config):
   

   # this function creates the tf.dataset.Dataset objects for each list
   # of input JPEGs.
   train_ds = Dataset.from_filelist(config['data']['train_filelist'],config)
   valid_ds = Dataset.from_filelist(config['data']['test_filelist'],config)

   return train_ds,valid_ds


if __name__ == '__main__':
    # configure logging module
    logging_format = '%(asctime)s %(levelname)s:%(process)s:%(thread)s:%(name)s:%(message)s'
    logging_datefmt = '%Y-%m-%d %H:%M:%S'
    logging_level = logging.INFO
    logging.basicConfig(level=logging_level,
                       format=logging_format,
                       datefmt=logging_datefmt)
    # parse command line
    import argparse,json
    parser = argparse.ArgumentParser(description='test this')
    parser.add_argument('-c', '--config', dest='config_filename',
                       help='configuration filename in json format',
                       required=True)
    parser.add_argument('-l','--logdir', dest='logdir',
                       help='log output directory',default='logdir')
    parser.add_argument('-n','--nsteps', dest='nsteps',
                       help='number of steps to run',default=10,type=int)
    parser.add_argument('-t', '--nthreads',help='number of threads, if set overrides config file',default=0,type=int)
    args = parser.parse_args()

    # parse config file
    config = json.load(open(args.config_filename))
    if args.nthreads > 0:
        config['data']['num_parallel_readers'] = args.nthreads

    logger.info('configruation file settings: \n %s',json.dumps(config,indent=4, sort_keys=True))

    # these would be set by horovod if run in a parallel setup
    rank = 0
    num_ranks = 1

    # call function to build dataset objects
    # both of the returned objects are tf.dataset.Dataset objects
    trainds, testds = get_datasets(config)

    ## create samplers for these datasets
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainds,num_ranks,rank,shuffle=True,drop_last=True)
    test_sampler = torch.utils.data.distributed.DistributedSampler(testds,num_ranks,rank,shuffle=True,drop_last=True)

    ## create data loaders
    batch_size = config['data']['batch_size']
    train_loader = torch.utils.data.DataLoader(trainds,shuffle=False,
                                               sampler=train_sampler,num_workers=config['data']['num_parallel_readers'],
                                               batch_size=batch_size,persistent_workers=True)
    test_loader = torch.utils.data.DataLoader(testds,shuffle=False,
                                               sampler=test_sampler,num_workers=config['data']['num_parallel_readers'],
                                               batch_size=batch_size,persistent_workers=True)

    # epoch loop
    for epoch in range(1):
        logger.info(f'epoch = {epoch}')

        # calling this is required to get the shuffle to work and to reset the dataset 
        train_sampler.set_epoch(epoch)

        # can iterate over a dataset object
        start = time.time()
        for batch_number,(inputs,labels) in enumerate(train_loader):
            logger.info('batch_number = %s input shape = %s    labels shape = %s',batch_number,inputs.shape,labels.shape)
            logger.info('batch_number = %s labels = %s',batch_number,np.squeeze(labels[0:10].numpy()).tolist())
            
            if batch_number > args.nsteps: break
        duration = time.time() - start
        images = config['data']['batch_size'] * args.nsteps
        logger.info('imgs/sec: %5.2f',images/duration)
