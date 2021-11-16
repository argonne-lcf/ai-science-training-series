# Tensorflow Example Data Pipeline
by J. Taylor Childers (jchilders@anl.gov)

This is the example covered in the [README](../README.md) one level up.

Example submission scripts for ThetaKNL & ThetaGPU are provided.

Submit to ThetaGPU using:
```bash
qsub -A <project> -q <queue> submit_thetagpu.sh
```

Submit to ThetaKNL using:
```bash
qsub -A <project> -q <queue> submit_thetknl.sh
```

All log files go into the `logdir` folder.


# Profiler View

You can view the processes and how they occupy the compute resources in Tensorflow using tensorboard.

You can login to Theta using:
```bash
# our proxy port, must be > 1024
export PORT=10001
# login to theta with a port forwarding
ssh -D $PORT user@theta.alcf.anl.gov
# load any conda environment that has a compatible tensorboard installation
module load miniconda-3/2021-07-28
# add CUDA libraries if you are running on ThetaGPU
export LD_LIBRARY_PATH=/lus/theta-fs0/software/thetagpu/cuda/cuda_11.3.0_465.19.01_linux/lib64:/lus/theta-fs0/software/thetagpu/cuda/cudnn-11.3-linux-x64-v8.2.0.53/lib64
# start tensorboard (load_fast==false is a recent setting that seems to be needed until Tensorflow work's out the bugs)
tensorboard --bind_all --logdir . --load_fast=false
```
Only 1 user can use a specific port so if you get an error choose another port number larger than `1024`.

Once you have that setup. Set the Socks5 proxy of your favorite browser to host `localhost` and port `$PORT` (where `$PORT` is the value you used in the above script, like `10001`). Now in the browser URL enter the login node on which you started `tensorboard`. For instance, if you are on `thetalogin6`, now you can type in `thetalogin6.alcf.anl.gov:6006`. Here `6006` is the port that `tensorboard` uses by default to start up it's web service, but may vary if you customize it. 
