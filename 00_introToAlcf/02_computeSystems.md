# Computing System Overview

[Video presentation of our System Introduction](https://www.alcf.anl.gov/support-center/training-assets/getting-started-theta)

## Theta
![Theta](https://www.alcf.anl.gov/sites/default/files/styles/965x543/public/2019-10/09_ALCF-Theta_111016_rgb.jpg?itok=lcvZKE6k)
Theta is an 11.7-petaflops supercomputer based on Intel processors and interconnect technology, an advanced memory architecture, and a Lustre-based parallel file system, all integrated by Cray’s HPC software stack.

Theta Machine Specs
* Architecture:  Intel-Cray XC40
* Speed: 11.7 petaflops
* Processor per node: 64-core, 1.3-GHz Intel Xeon Phi 7230
* Nodes: 4,392
* Cores: 281,088
* Memory: 843 TB (192GB / node)
* High-bandwidth memory: 70 TB (16GB / node)
* Interconnect: Aries network with Dragonfly topology
* Racks: 24

## ThetaGPU
ThetaGPU is an NVIDIA DGX A100-based system. The DGX A100 comprises eight NVIDIA A100 GPUs that provide a total of 320 gigabytes of memory for training AI datasets, as well as high-speed NVIDIA Mellanox ConnectX-6 network interfaces.

ThetaGPU Machine Specs
* Architecture: NVIDIA DGX A100
* Speed: 3.9 petaflops
* Processors: AMD EPYC 7742
* Nodes: 24
* DDR4 Memory: 24 TB
* GPU Memory: 7,680 GB
* Racks: 7


# Cluster/HPC Computing Hardware Setup

![Hardware](https://user-images.githubusercontent.com/10742392/138757649-5a780c4c-b185-4a3b-9a41-7490fe8da777.png)

In large supercomputers like Theta, you combine multiple computer processors (CPUs) and/or graphics processors (GPUs) into a single _node_. A _node_ is effectively like your desktop computer. It has a CPU on which the local operating system runs. It has local memory for running software. It may have GPUs for doing intensive calculations. 
