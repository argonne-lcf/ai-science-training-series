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
qsub -n 1 -q full-node -A ALCFAITP -I -t 15 --attrs=pubnet
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
   - PyTorch MNIST - 8 GPUs
     ```bash
     mpirun -np 8 python pytorch_mnist.py --device gpu
     ```

   - PyTorch CIFAR10 - 8 GPUs
     ```bash
     mpirun -np 8 python pytorch_cifar10.py --device gpu
     ```

   -  TensorFlow MNIST - 8 GPUs
      ```bash
      mpirun -np 8 python tensorflow2_mnist.py --device gpu
      ```

   - TensorFlow Keras MNIST - 8 GPUs
     ```bash
     mpirun -np 8 python  tensorflow2_keras_mnist.py --device gpu
     ```
     
4. Test scaling and investigate the issue of large batch size training
The following script performes a simple scaling test with the MNIST dataset and a PyTorch model:
   ```bash
   for n in 1 2 4 8
   do
     	mpirun -np $n python pytorch_mnist.py --device gpu >& pytorch_mnist.out.$n
   done
   ```
  
   We have prepared some (non-interactive) submission scripts in `./submissions/qsub_*`
   
   
   **PyTorch**
   
   Time for 32 epochs 
   
| GPUs | Cifar10 (s) | MNIST (s) |
| ---- | ---------------------- | -------------------- |
|    1 |            522.3       |         499.8        |
|    2 |            318.8       |         283.9        |
|    4 |            121.4       |         100.4        |
|    8 |             73.5       |         58.8         |
