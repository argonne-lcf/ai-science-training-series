# Hands on for Data Parallel Deep Learning on ThetaGPU

0. Modify tensorflow2_mnist_orig.py and instrument the code with Horoovd. [This can be done on the login node!]
    * You can go to https://jupyter.alcf.anl.gov/, login in, and select local host process; 
    * Open a terminal
    * git clone https://github.com/argonne-lcf/ai-science-training-series.git
    * cd ai-science-training-series/

1. Request an interactive session on ThetaGPU:
```bash
# Login to theta
ssh -CY user@theta.alcf.anl.gov
# Login to ThetaGPU login node
ssh -CY thetagpusn1 
# Requesting 1 node  
qsub -n 1 -q training -A ALCFAITP -I -t 15 --attrs=pubnet
```

You can also login in 

2. Setup the Python environment to include TensorFlow, Keras, PyTorch, and Horovod:
   ```bash
   . /etc/profile.d/z00_lmod.sh
   module load conda
   conda activate
   ```
   Notice that the first line is needed if you are setting up the environment in a submission script. It is not needed if you are running in interactive mode. 
   
3. Run examples on a single node
   -  TensorFlow MNIST - 8 GPUs
      ```bash
      mpirun -np 1 python tensorflow2_mnist.py --device gpu
      mpirun -np 2 python tensorflow2_mnist.py --device gpu
      mpirun -np 4 python tensorflow2_mnist.py --device gpu
      mpirun -np 8 python tensorflow2_mnist.py --device gpu
      ```
     
   - PyTorch MNIST - 8 GPUs
     ```bash
     mpirun -np 1 python pytorch_mnist.py --device gpu
     mpirun -np 2 python pytorch_mnist.py --device gpu
     mpirun -np 4 python pytorch_mnist.py --device gpu
     mpirun -np 8 python pytorch_mnist.py --device gpu
     ```
  
   We have prepared some (non-interactive) submission scripts in `./submissions/qsub_*`
   
   Time for 16 epochs 
   
   | GPUs |     Total time (s)     |  
   | ---- | ---------------------- |
   |    1 |            27.59       |
   |    2 |            23.31       |
   |    4 |            12.80       |
   |    8 |            14.41       |

Time per epoch

   | GPUs |     Total time (s)     |  
   | ---- | ---------------------- |
   |    1 |            1.31        |
   |    2 |            1.01        |
   |    4 |            0.37        |
   |    8 |            0.21        |


4. Profiling

   * MPI profiling
   ```bash
   LD_PRELOAD=/soft/perftools/hpctw/lib/libmpitrace.so
   ```

   * Horovod timeline trace
   ```bash
   HOROVOD_TIMELINE=timeline.json 
   ```
  
   ```bash
   LD_PRELOAD=/soft/perftools/hpctw/lib/libmpitrace.so HOROVOD_TIMELINE=timeline.json mpirun -np 8 python pytorch_mnist.py 
   ```

   One can then open timeline.json with Chrome 
   ![acc](./images/timeline.png)

