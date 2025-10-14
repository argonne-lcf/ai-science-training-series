# Building a CPU-side data pipeline
*Led by [J. Taylor Childers](jchilders@anl.gov) from ALCF*

New AI systems largely depend on CPU-GPU hybrid architectures. This makes efficient use of CPU-side resources important in order to feed sufficient data to the GPU algorithms. In most cases, the best setup is to have the CPU process the training data and build training batches, while the GPU performs the compute intensive model inference ("forward pass")
 and gradient calculations ("backward pass" or "back-prop").

This section demonstrates building a data pipeline for both TensorFlow and PyTorch. TensorFlow's data pipeline API is a bit more advanced than PyTorch so we'll focus on that one, though we include an example in PyTorch.

[THIS NOTEBOOK](00_tensorflowDatasetAPI/inspect_pipeline.ipynb) is a good way to first inspect the dataset.

[THIS REPO](https://github.com/jtchilders/tensorflow_skeleton) contains a full example of training ResNet on the ImageNet dataset.

# ImageNet Dataset

This example uses the ImageNet dataset to build training batches.

![Turtle](images/n01667778_12001.JPEG) ![Dog](images/n02094114_1205.JPEG)

This dataset includes JPEG images and an XML annotation for each file that defines a bounding box for each class. Building a training batch requires pre-processing the images and annotations. In our example, we have created text files that list all the files in the training set and validation set. For each text file, we need to use the input JPEG files and build tensors that include multiple images per training batch.

# TensorFlow Dataset example

TensorFlow has some very nice tools to help us build the pipeline. You'll find the [example here](00_tensorflowDatasetAPI/ilsvrc_dataset.py).

## Build from file list
We'll start in the function `build_dataset_from_filelist`.

1. Open the filelist
```python
# loading full filelist
filelist = []
with open(filelist_filename) as file:
   for line in file:
      filelist.append(line.strip())
```
2. Parse the list of files into a TF Tensor
```python
filelist = tf.data.Dataset.from_tensor_slices(filelist)
```
3. If we are using Horovod for MPI parallelism, we want to "shard" the data across nodes so each node processes unique data
```python
filelist = filelist.shard(config['hvd'].size(), config['hvd'].rank())
```
4. Shuffle our filelist at each epoch barrier
```python
filelist = filelist.shuffle(dc['shuffle_buffer'],reshuffle_each_iteration=dc['reshuffle_each_iteration'])
```
5. Run a custom function on the filelist, which effectively opens the JPEG file, loads the data into a TF Tensor and extracts the class labels. If there are multiple objects in the image, this function will return more than one image using the bounding boxes. `num_parallel_calls` allows this function to run in parallel so many JPEG files can be read into memory and processed in parallel threads.
```python
ds = filelist.map(load_image_label_bb,
                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
```
6. Since the previous map fuction may return one or more images, we need to unbatch the output before we batch it into our fixed batch size
```python
ds = ds.apply(tf.data.Dataset.unbatch)
ds = ds.batch(dc['batch_size'])
```
7. Tell the dataset it can prepare the next batch(es) prior to them being requested
```python
ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
```

Done.

We can now iterate over this dataset in a loop:
```python
for inputs,labels in ds:
   prediction = model(inputs)
   loss = loss_func(prediction,labels)
   # ...
```

## Parallel Processing on ThetaGPU

The example `00_tensorflowDatasetAPI/ilsvrc_dataset.py` can be run in an effective "serial" mode on ThetaGPU (single-gpu queue) worker nodes using
```bash
# module load conda/2021-11-30; conda activate
CUDA_VISIBLE_DEVICES=-1 python 00_tensorflowDatasetAPI/ilsvrc_dataset.py -c 00_tensorflowDatasetAPI/ilsvrc.json --num-parallel-readers 1 --prefetch-buffer-size 0
# shows about 33 images per second
```
Note, we do not need the GPU for this and to avoid memory copies to/from the device, we simply set `CUDA_VISIBLE_DEVICES=-1` to disable the GPU in this example.

You will see very poor performance as this is an example of serial data pipeline that only uses one process for reading JPEGs and does not pre-stage batch data. You can see in this screenshot from the [TensorFlow Profiler](https://github.com/argonne-lcf/sdl_ai_workshop/tree/master/04_profilingDeepLearning/TensorflowProfiler) how your processes are being utilized. The profile shows a single process handling all the data pipeline processes. All `ReadFile` calls are being done serially when they could be done in parallel. One long IO operation holds up the entire application.
![serial](images/ilsvrc_serial.png)

Now switch to running the parallel version on a ThetaGPU (single-gpu queue):
```bash
# module load conda/2021-11-30; conda activate
CUDA_VISIBLE_DEVICES=-1 python 00_tensorflowDatasetAPI/ilsvrc_dataset.py -c 00_tensorflowDatasetAPI/ilsvrc.json --num-parallel-readers 8 --prefetch-buffer-size 2
# shows about 120 images per second
```

You will see much better performance in this case. Batch processing time is down from 3 seconds to 1 second. The profiler shows we are running with our 8 parallel processes, all of which are opening JPEGs, processing them into tensors, extracting truth information, and so on.
![parallel](images/ilsvrc_parallel.png)


