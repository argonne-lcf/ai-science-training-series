# BERT on Graphcore 

These instructions are to train a BERT Pytorch model on the POD16. 

* Go to directory with BERT example 
  ```bash
  cd ~/graphcore/examples/nlp/bert/pytorch
  ```


* Activate PopTorch Environment 

  ```bash
  source ~/venvs/graphcore/poptorch33_env/bin/activate
  ```

* Install Requirements 

  ```bash
  pip3 install -r requirements.txt
  ```

* Run BERT pre-training on 4 IPUs
  ```bash
  /opt/slurm/bin/srun --ipus=4 python3 run_pretraining.py \
  --config demo_tiny_128  \
  --input-files /software/datasets/graphcore/wikipedia/128/large_wikicorpus_sample.tfrecord
  ```

  <details>
    <summary>Sample Output</summary>
    
    ```bash
    $ /opt/slurm/bin/srun --ipus=4 python3 run_pretraining.py --config demo_tiny_128  --input-files /software/datasets/graphcore/
  wikipedia/128/large_wikicorpus_sample.tfrecord
  srun: job 20279 queued and waiting for resources
  srun: job 20279 has been allocated resources
      Registered metric hook: total_compiling_time with object: <function get_results_for_compile_time at 0x7fb019cde160>
  Using config: demo_tiny_128
  [warning] With replication_factor == 1 you may need to set embedding_serialization_factor > 1 for the model to fit
  Building (if necessary) and loading residual_add_inplace_pattern.
  ------------------- Data Loading Started ------------------
  Extension horovod.torch has not been built: /projects/datascience/sraskar/gc-pod/envs/poptorch33_env/lib/python3.8/site-packages/horovod/torch/mpi_lib/_mpi_lib.cpython-38-x86_64-linux-gnu.so not found
  If this is not expected, reinstall Horovod with HOROVOD_WITH_PYTORCH=1 to debug the build error.
  Warning! MPI libs are missing, but python applications are still available.
      Registered metric hook: total_compiling_time with object: <function get_results_for_compile_time at 0x7f211c238670>
      Registered metric hook: total_compiling_time with object: <function get_results_for_compile_time at 0x7f9fde7f48b0>
  Data loaded in 9.856430063955486 secs
  -----------------------------------------------------------
  -------------------- Device Allocation --------------------
  Embedding  --> IPU 0
  Encoder 0  --> IPU 1
  Encoder 1  --> IPU 2
  Encoder 2  --> IPU 3
  Pooler     --> IPU 0
  Classifier --> IPU 0
  -----------------------------------------------------------
  ---------- Compilation/Loading from Cache Started ---------
  [17:47:29.814] [poptorch:cpp] [warning] [DISPATCHER] Type coerced from Long to Int for tensor id 6
  [17:47:29.814] [poptorch:cpp] [warning] [DISPATCHER] Type coerced from Long to Int for tensor id 7
  [17:47:29.824] [poptorch:cpp] [warning] [DISPATCHER] Type coerced from Long to Int for tensor id 67
  [17:47:29.824] [poptorch:cpp] [warning] [DISPATCHER] Type coerced from Long to Int for tensor id 68
  [17:47:29.825] [poptorch:cpp] [warning] [DISPATCHER] Type coerced from Long to Int for tensor id 69
  [17:47:29.825] [poptorch:cpp] [warning] [DISPATCHER] Type coerced from Long to Int for tensor id 70
  [17:47:29.825] [poptorch:cpp] [warning] [DISPATCHER] Type coerced from Long to Int for tensor id 71
  [17:47:29.825] [poptorch:cpp] [warning] [DISPATCHER] Type coerced from Long to Int for tensor id 72
  [17:47:29.830] [poptorch:cpp] [warning] [DISPATCHER] Tensor (ptr 0x9a55530) type coerced from Double to Float
  [17:47:29.830] [poptorch:cpp] [warning] [DISPATCHER] Tensor (ptr 0x9a55530) type coerced from Double to Float
  [17:47:29.837] [poptorch:cpp] [warning] [DISPATCHER] Tensor (ptr 0x9aa5be0) type coerced from Double to Float
  [17:47:29.849] [poptorch:cpp] [warning] [DISPATCHER] Tensor (ptr 0x9af7850) type coerced from Double to Float
  [17:47:29.860] [poptorch:cpp] [warning] ...repeated messages suppressed...
  [17:47:29.878] [poptorch:cpp] [warning] [DISPATCHER] Type coerced from Long to Int for tensor id 372
  [17:47:29.878] [poptorch:cpp] [warning] [DISPATCHER] Type coerced from Long to Int for tensor id 373
  [17:47:29.879] [poptorch:cpp] [warning] [DISPATCHER] Type coerced from Long to Int for tensor id 388
  [17:47:29.879] [poptorch:cpp] [warning] [DISPATCHER] Type coerced from Long to Int for tensor id 389
  Graph compilation: 100%|██████████| 100/100 [01:32<00:00]
  Compiled/Loaded model in 97.88873339025304 secs
  -----------------------------------------------------------
  --------------------- Training Started --------------------
  Step: 149 / 149 - LR: 1.00e-03 - total loss: 7.389 - mlm_loss: 7.008 - nsp_loss: 0.380 - mlm_acc: 0.161 % - nsp_acc: 0.875 %:  99%|█████████▉| 149/150 [00:03<00:00, 45.20it/s, throughput: 749.6 samples/sec]
  -----------------------------------------------------------
  -------------------- Training Metrics ---------------------
  global_batch_size: 16
  device_iterations: 1
  training_steps: 150
  Training time: 3.297 secs
  -----------------------------------------------------------
  /usr/lib/python3.8/multiprocessing/resource_tracker.py:216: UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
    warnings.warn('resource_tracker: There appear to be %d '
    
    

    ```
  </details>

