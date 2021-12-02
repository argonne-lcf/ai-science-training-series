# PyTorch Example Data Pipeline
*Led by [J. Taylor Childers](jchilders@anl.gov) from ALCF*

This is very similar to the TensorFlow example covered in the [README](../README.md) one level up.

Example submission script for ThetaGPU is provided.

Submit to ThetaGPU using:
```bash
qsub -A <project> -q <queue> submit_thetagpu.sh
```
During this training program, you should use:
```bash
qsub -A ALCFAITP -q single-gpu submit_thetagpu.sh
```
All log files go into the `logdir/` folder.
