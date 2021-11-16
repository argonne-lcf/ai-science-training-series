# PyTorch Example Data Pipeline
by J. Taylor Childers (jchilders@anl.gov)

This is very similar to the Tensorflow example covered in the [README](../README.md) one level up.

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
