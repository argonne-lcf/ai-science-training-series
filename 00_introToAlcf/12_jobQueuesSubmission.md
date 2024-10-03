# How to run jobs on the system

![gpu-qsub](./img/theta-gpu-qsub.gif)

Most of the time, there are not enough Sophia or Polaris nodes to run everyone's
workloads simultaneously.  Batch queueing systems solve this problem by
requiring that you submit your workload as a **batch job script** to a
queue.  All jobs wait in the queue until it's their turn to start,
whereupon the [PBS scheduler](https://docs.alcf.anl.gov/running-jobs/job-and-queue-scheduling/) allocates compute resources and launches the shell script onto the appropriate node.

When we submit a batch job script, we need to specify:

- The **shell script** that contains the commands to run
- The **number of compute nodes**  needed for the job.  In our examples, we will
stick to code that runs on  a single GPU within a node, and therefore always ask
for 1 node (`-l select=<number of nodes>`).
- The maximum **walltime** in minutes that the job will run for.  We will ask for
fifteen minutes (`-l walltime=<HH:MM::SS>`).
-  The **queue** to which the job should be submitted.  Cobalt manages several queues
for each system; each queue is a separate line that jobs wait in until
they are eligible to start.  We will submit jobs to the "single GPU" queue designed
for routing jobs onto 1/8th of a DGX node (`-q by-gpu`).
- The **allocation** to which the used node-hours will be charged. Multiply the
number of nodes by the walltime to get the *node-hours* used by a batch job.  This is
the unit by which teams are charged for their usage of ALCF systems.  For instance, a
a single-node, 15-minute job would cost 0.25 node hours.  For the training exercises,
we can use a special training allocation set aside (`-A ALCFAITP`).
- The **filesystems** needed by the job should also be included in the job submission command. This lets the scheduler know which files systems should be mounted for use on the nodes. It is specified using `-l filesystems=<colon-separated-list>`. The options on Polaris are `home_fs` for your home directory, `grand` for `/lus/grand`, and/or `eagle` for `/lus/eagle`.

## Submitting a batch job script

To submit a minimal shell script, we need to make an executable file that starts with the very first
line `#!/bin/bash -l` indicating the `bash` shell interpreter will be used to run the job in login mode (`-l`):

```bash
#!/bin/bash -l

echo "Hello world!"

module load conda
conda activate
python -c 'print(1 + 1)'
```

Save this script into a file called `hello.sh` then run:

```shell
# Make the hello.sh script executable:
$ chmod +x hello.sh

# Submit hello.sh to the single-GPU Training queue:
$ qsub-gpu -A ALCFAITP -q single-gpu -l select=1 -l walltime=0:15:0 -l filesystems=home_fs hello.sh
```

After running these commands, we should find that our job is
now waiting in the queue by checking `qstat-gpu`:

```shell
$ qstat -u $USER
```

Once the job starts running, you should find files ending with the `.output` and `.error` suffixes, which represent the standard output and standard error streams written from our executing batch job script:

```shell
$ ls
<PBS-job-id>.OU <PBS-job-id>.ER

$ cat <PBS-job-id>.OU
Hello world!
2
```

Instead of passing PBS parameters on the command line to `qsub-gpu`, we can also include these flags as **PBS directives** directly underneath the `#!/bin/bash -l` line:

```bash
#!/bin/bash -l
#PBS -l select=1
#PBS -l walltime=00:15:00
#PBS -l filesystems=home_fs
#PBS -q single-gpu
#PBS -A ALCFAITP

echo "Hello world!"

module load conda/2021-09-22
python -c 'print(1 + 1)'
```

If you change `hello.sh` to contain these `#PBS` directives, you can submit the script without repeating any flags in the shell:

```bash
$ qsub hello.sh
```

There are many other PBS flags that you can use to control how your jobs run.
You should visit the ALCF website to read more about:
- [Help with Sophia](https://docs.alcf.anl.gov/sophia/getting-started/)
- [Help with Polaris](https://docs.alcf.anl.gov/polaris/getting-started/)
- [Help with PBS](https://docs.alcf.anl.gov/running-jobs/job-and-queue-scheduling/) 

## Example using PyTorch MNIST ML training
![pytorch gif](img/qsub_pytorch_mnist.gif)

Some more realistic example job scripts are attached to this repository: try submitting `12_pytorch_mnist.sh` or `12_tensorflow_mnist.sh` to train a neural network on the MNIST classification task, using either PyTorch or Tensorflow, respectively.  You will find that these scripts tend to follow a simple pattern:

```bash
# Set up the Python environment
module load conda
conda activate

# Run the Python model training script
python 12_pytorch_mnist.py
```

In upcoming sessions, you will learn more about these AI frameworks and how to write your own Python programs to build, train, and test deep learning models.

## Interactive jobs

Once you have figured out exactly what to run, Batch jobs are a great way to submit workloads and allow the system to take over scheduling. You can go do something else and log back onto the system another day to check on the status of your jobs.

When testing new ideas or developing a project, however, it's more useful to be able to **SSH directly onto a compute node** and run commands locally.  If you get something wrong, you can stay on the node while fixing bugs, instead of being kicked off and having to wait repeatedly for the next batch job to start.  To obtain a node interactively, all you have to do is **replace the batch job script** (`hello.sh`) **with the interactive flag** (`-I`):


```bash
$ qsub -A ALCFAITP -q single-gpu -l select=1 -l walltime=0:15:0 -l filesystems=home_fs -I
```

This command will block until the node is available, and PBS will open a new SSH terminal having you logged into the compute node directly.
