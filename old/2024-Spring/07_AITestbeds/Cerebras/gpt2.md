# GPT-2 on Cerebras

* Go to directory with the bert example. 
  ```bash
  cd ~/R_2.3.0/modelzoo/transformers/pytorch/gpt2
  ```

* Activate PyTroch virtual Environment 
  ```bash
  source ~/R_2.3.0/venv_cerebras_pt/bin/activate
  ```

* Modify config file path to datasets on the system.
  Commonly used various pre-processed datasets are available the systems for your conveneince. Modify config file `configs/params_gpt2_small.yaml` to point to correct path. 
  ```bash
  vim configs/params_gpt2_small.yaml
  train_input:
      data_processor: "GptHDF5DataProcessor"
      data_dir: "/software/cerebras/dataset/OWT/Pytorch/train_8M_msl2048"

  eval_input:
      data_processor: "GptHDF5DataProcessor"
      data_dir: "/software/cerebras/dataset/OWT/Pytorch/val_msl2048"
  ```

* Run Training Job
  ```bash
  # If MODEL_DIR already exisits from previous runs, delete it. 
  export MODEL_DIR=model_dir_gpt2_pytorch
  if [ -d "$MODEL_DIR" ]; then rm -Rf $MODEL_DIR; fi

  python run.py CSX --job_labels name=gpt2_pt  \
  --params configs/params_gpt2_small.yaml \
  --num_workers_per_csx=1 --mode train \
  --model_dir $MODEL_DIR --mount_dirs /home/ /software/ \
  --python_paths /home/$(whoami)/R_2.3.0/modelzoo/ \
  --compile_dir $(whoami) |& tee mytest.log
  ```
  <details>
    <summary>Sample Output</summary>
    
    ```bash
  $ python run.py CSX --job_labels name=gpt2_pt  --params configs/params_gpt2_small.yaml --num_workers_per_csx=1 --mode train --model_dir $MODEL_DIR --mount_dirs /home/ /software/ --python_paths /home/$(whoami)/R_2.3.0/modelzoo/ --compile_dir $(whoami) |& tee mytest.log
  Initializing module Gpt2Model.accuracy_metric: [00:02, note=Initializing parameters]2024-03-14 03:18:49,076 WARNING:   LinearLR got 1 unexpected and unused parameters: ['alpha'].
  Please ensure that you specified the correct parameters:
  LinearLR(optimizer: torch.optim.optimizer.Optimizer, initial_learning_rate: float = 0.0006, end_learning_rate: float = 0.0, total_iters: int = 150000, cycle: bool = False)
  Passing in unused parameters is deprecated behaviour and support for it will be removed in a future release.
  2024-03-14 03:18:49,079 INFO:   Effective batch size is 112.
  2024-03-14 03:18:49,782 INFO:   Checkpoint autoloading is enabled. Looking for latest checkpoint in "model_dir_gpt2_pytorch" directory with the following naming convention: `checkpoint_(step)(_timestamp)?.mdl`.
  2024-03-14 03:18:49,784 INFO:   No checkpoints were found in "model_dir_gpt2_pytorch".
  2024-03-14 03:18:49,784 INFO:   No checkpoint was provided. Using randomly initialized model parameters.
  Initializing module Gpt2Model.accuracy_metric: [00:03, note=Initializing parameters]2024-03-14 03:18:49,849 INFO:   Saving checkpoint at step 0
  2024-03-14 03:18:59,636 INFO:   Saved checkpoint model_dir_gpt2_pytorch/checkpoint_0.mdl

  2024-03-14 03:19:05,667 INFO:   Compiling the model. This may take a few minutes.
  2024-03-14 03:19:06,821 INFO:   Initiating a new image build job against the cluster server.
  2024-03-14 03:19:06,880 INFO:   Custom worker image build is disabled from server.
  2024-03-14 03:19:07,113 INFO:   Initiating a new compile wsjob against the cluster server.
  2024-03-14 03:19:07,180 INFO:   compile job id: wsjob-sulfy68x6nus3qk97eifxy, remote log path: /n1/wsjob/workdir/wsjob-sulfy68x6nus3qk97eifxy
  2024-03-14 03:19:17,202 INFO:   Poll ingress status: Waiting for job running, current job status: Queueing, msg: job queueing, waiting for lock grant. Cluster status: 0 compile job(s) queued before current job
  2024-03-14 03:36:07,708 INFO:   Poll ingress status: Waiting for job running, current job status: Scheduled, msg: lock grant success. 
  2024-03-14 03:36:17,722 INFO:   Poll ingress status: Waiting for coordinator to be ready.
  2024-03-14 03:36:37,749 INFO:   Ingress is ready: Coordinator service ready, poll ingress success.
  2024-03-14 03:36:42,833 INFO:   Pre-optimization transforms...
  2024-03-14 03:36:43,336 INFO:   Optimizing layouts and memory usage...
  2024-03-14 03:36:43,340 INFO:   Gradient accumulation disabled
  2024-03-14 03:36:43,692 INFO:   Exploring floorplans
  2024-03-14 03:37:54,149 INFO:   Exploring data layouts
  2024-03-14 03:38:06,890 INFO:   Optimizing memory usage
  2024-03-14 03:38:34,008 INFO:   Post-layout optimizations...
  2024-03-14 03:38:36,184 INFO:   Allocating buffers...
  2024-03-14 03:38:37,099 INFO:   Code generation...
  2024-03-14 03:38:40,431 INFO:   Compiling image...
  2024-03-14 03:38:40,435 INFO:   Compiling kernels
  2024-03-14 03:40:09,321 INFO:   Compiling final image
  2024-03-14 03:41:33,511 INFO:   Compile artifacts successfully written to remote compile directory. Compile hash is: cs_1852078805518656824
  2024-03-14 03:41:33,542 INFO:   Heartbeat thread stopped for wsjob-sulfy68x6nus3qk97eifxy.
  2024-03-14 03:41:33,547 INFO:   Compile was successful!
  2024-03-14 03:41:33,554 INFO:   Programming Cerebras Wafer Scale Cluster for execution. This may take a few minutes.
  2024-03-14 03:41:35,919 INFO:   Initiating a new execute wsjob against the cluster server.
  2024-03-14 03:41:35,995 INFO:   execute job id: wsjob-7qam2n3dj6ettzkmdu7ywe, remote log path: /n1/wsjob/workdir/wsjob-7qam2n3dj6ettzkmdu7ywe
  2024-03-14 03:41:46,016 INFO:   Poll ingress status: Waiting for job running, current job status: Scheduled, msg: lock grant success. 
  2024-03-14 03:42:16,048 INFO:   Poll ingress status: Waiting for coordinator to be ready.
  2024-03-14 03:42:56,125 INFO:   Ingress is ready: Coordinator service ready, poll ingress success.
  2024-03-14 03:42:56,272 INFO:   Preparing to execute using 1 CSX
  2024-03-14 03:43:09,259 INFO:   About to send initial weights
  2024-03-14 03:43:19,776 INFO:   Finished sending initial weights
  2024-03-14 03:43:19,778 INFO:   Finalizing appliance staging for the run
  2024-03-14 03:43:19,823 INFO:   Waiting for device programming to complete
  2024-03-14 03:45:36,148 INFO:   Device programming is complete
  2024-03-14 03:45:36,150 INFO:   Waiting for input workers to prime the data pipeline and begin streaming ...
  2024-03-14 03:45:36,204 INFO:   Input workers have begun streaming input data
  2024-03-14 03:45:40,175 INFO:   Appliance staging is complete
  2024-03-14 03:45:40,183 INFO:   Beginning appliance run
  2024-03-14 03:46:02,970 INFO:   | Train Device=CSX, Step=100, Loss=7.93750, Rate=492.40 samples/sec, GlobalRate=492.40 samples/sec
  2024-03-14 03:46:25,665 INFO:   | Train Device=CSX, Step=200, Loss=6.62500, Rate=493.05 samples/sec, GlobalRate=492.94 samples/sec
  2024-03-14 03:46:48,356 INFO:   | Train Device=CSX, Step=300, Loss=6.21875, Rate=493.37 samples/sec, GlobalRate=493.16 samples/sec
  2024-03-14 03:47:10,956 INFO:   | Train Device=CSX, Step=400, Loss=6.09375, Rate=494.70 samples/sec, GlobalRate=493.76 samples/sec
  2024-03-14 03:47:33,566 INFO:   | Train Device=CSX, Step=500, Loss=5.87500, Rate=495.10 samples/sec, GlobalRate=494.08 samples/sec
  2024-03-14 03:47:56,205 INFO:   | Train Device=CSX, Step=600, Loss=5.59375, Rate=494.86 samples/sec, GlobalRate=494.19 samples/sec
  2024-03-14 03:48:18,794 INFO:   | Train Device=CSX, Step=700, Loss=5.40625, Rate=495.44 samples/sec, GlobalRate=494.42 samples/sec
  2024-03-14 03:48:41,455 INFO:   | Train Device=CSX, Step=800, Loss=5.21875, Rate=494.72 samples/sec, GlobalRate=494.40 samples/sec
  2024-03-14 03:49:04,093 INFO:   | Train Device=CSX, Step=900, Loss=5.12500, Rate=494.73 samples/sec, GlobalRate=494.43 samples/sec
  2024-03-14 03:49:26,755 INFO:   | Train Device=CSX, Step=1000, Loss=4.87500, Rate=494.43 samples/sec, GlobalRate=494.41 samples/sec
  2024-03-14 03:49:49,356 INFO:   | Train Device=CSX, Step=1100, Loss=4.81250, Rate=495.10 samples/sec, GlobalRate=494.52 samples/sec
  2024-03-14 03:50:12,057 INFO:   | Train Device=CSX, Step=1200, Loss=4.62500, Rate=494.06 samples/sec, GlobalRate=494.42 samples/sec
  2024-03-14 03:50:34,708 INFO:   | Train Device=CSX, Step=1300, Loss=4.46875, Rate=494.30 samples/sec, GlobalRate=494.42 samples/sec
  2024-03-14 03:50:57,385 INFO:   | Train Device=CSX, Step=1400, Loss=4.31250, Rate=494.06 samples/sec, GlobalRate=494.39 samples/sec
  2024-03-14 03:51:20,018 INFO:   | Train Device=CSX, Step=1500, Loss=4.25000, Rate=494.53 samples/sec, GlobalRate=494.42 samples/sec
  2024-03-14 03:51:42,704 INFO:   | Train Device=CSX, Step=1600, Loss=4.28125, Rate=494.04 samples/sec, GlobalRate=494.37 samples/sec
  2024-03-14 03:52:05,351 INFO:   | Train Device=CSX, Step=1700, Loss=4.15625, Rate=494.34 samples/sec, GlobalRate=494.38 samples/sec
  2024-03-14 03:52:28,001 INFO:   | Train Device=CSX, Step=1800, Loss=4.06250, Rate=494.42 samples/sec, GlobalRate=494.39 samples/sec
  2024-03-14 03:52:50,644 INFO:   | Train Device=CSX, Step=1900, Loss=4.09375, Rate=494.55 samples/sec, GlobalRate=494.40 samples/sec
  2024-03-14 03:53:13,313 INFO:   | Train Device=CSX, Step=2000, Loss=4.00000, Rate=494.27 samples/sec, GlobalRate=494.38 samples/sec
  2024-03-14 03:53:35,922 INFO:   | Train Device=CSX, Step=2100, Loss=3.93750, Rate=494.93 samples/sec, GlobalRate=494.43 samples/sec
  2024-03-14 03:53:58,548 INFO:   | Train Device=CSX, Step=2200, Loss=3.89062, Rate=494.98 samples/sec, GlobalRate=494.46 samples/sec
  2024-03-14 03:54:21,189 INFO:   | Train Device=CSX, Step=2300, Loss=3.90625, Rate=494.79 samples/sec, GlobalRate=494.47 samples/sec
  2024-03-14 03:54:43,858 INFO:   | Train Device=CSX, Step=2400, Loss=3.82812, Rate=494.36 samples/sec, GlobalRate=494.45 samples/sec
  2024-03-14 03:55:06,470 INFO:   | Train Device=CSX, Step=2500, Loss=3.89062, Rate=494.93 samples/sec, GlobalRate=494.48 samples/sec
  2024-03-14 03:55:29,060 INFO:   | Train Device=CSX, Step=2600, Loss=3.85938, Rate=495.45 samples/sec, GlobalRate=494.53 samples/sec
  2024-03-14 03:55:51,716 INFO:   | Train Device=CSX, Step=2700, Loss=3.81250, Rate=494.79 samples/sec, GlobalRate=494.53 samples/sec
  2024-03-14 03:56:14,368 INFO:   | Train Device=CSX, Step=2800, Loss=3.76562, Rate=494.57 samples/sec, GlobalRate=494.52 samples/sec
  2024-03-14 03:56:37,013 INFO:   | Train Device=CSX, Step=2900, Loss=3.81250, Rate=494.60 samples/sec, GlobalRate=494.53 samples/sec
  2024-03-14 03:56:59,684 INFO:   | Train Device=CSX, Step=3000, Loss=3.76562, Rate=494.25 samples/sec, GlobalRate=494.51 samples/sec
  2024-03-14 03:57:22,308 INFO:   | Train Device=CSX, Step=3100, Loss=3.75000, Rate=494.73 samples/sec, GlobalRate=494.53 samples/sec
  2024-03-14 03:57:44,942 INFO:   | Train Device=CSX, Step=3200, Loss=3.76562, Rate=494.79 samples/sec, GlobalRate=494.54 samples/sec
  2024-03-14 03:58:07,575 INFO:   | Train Device=CSX, Step=3300, Loss=3.75000, Rate=494.84 samples/sec, GlobalRate=494.55 samples/sec
  2024-03-14 03:58:30,174 INFO:   | Train Device=CSX, Step=3400, Loss=3.67188, Rate=495.29 samples/sec, GlobalRate=494.58 samples/sec
  2024-03-14 03:58:52,865 INFO:   | Train Device=CSX, Step=3500, Loss=3.65625, Rate=494.26 samples/sec, GlobalRate=494.55 samples/sec
  2024-03-14 03:59:15,528 INFO:   | Train Device=CSX, Step=3600, Loss=3.70312, Rate=494.23 samples/sec, GlobalRate=494.54 samples/sec
  2024-03-14 03:59:38,164 INFO:   | Train Device=CSX, Step=3700, Loss=3.68750, Rate=494.56 samples/sec, GlobalRate=494.55 samples/sec
  2024-03-14 04:00:00,783 INFO:   | Train Device=CSX, Step=3800, Loss=3.67188, Rate=494.92 samples/sec, GlobalRate=494.56 samples/sec
  2024-03-14 04:00:23,426 INFO:   | Train Device=CSX, Step=3900, Loss=3.64062, Rate=494.75 samples/sec, GlobalRate=494.56 samples/sec
  2024-03-14 04:00:46,111 INFO:   | Train Device=CSX, Step=4000, Loss=3.62500, Rate=494.13 samples/sec, GlobalRate=494.54 samples/sec
  2024-03-14 04:01:08,768 INFO:   | Train Device=CSX, Step=4100, Loss=3.59375, Rate=494.25 samples/sec, GlobalRate=494.54 samples/sec
  2024-03-14 04:01:31,432 INFO:   | Train Device=CSX, Step=4200, Loss=3.65625, Rate=494.21 samples/sec, GlobalRate=494.53 samples/sec
  2024-03-14 04:01:54,067 INFO:   | Train Device=CSX, Step=4300, Loss=3.59375, Rate=494.56 samples/sec, GlobalRate=494.54 samples/sec
  2024-03-14 04:02:16,711 INFO:   | Train Device=CSX, Step=4400, Loss=3.62500, Rate=494.60 samples/sec, GlobalRate=494.54 samples/sec
  2024-03-14 04:02:39,394 INFO:   | Train Device=CSX, Step=4500, Loss=3.59375, Rate=494.10 samples/sec, GlobalRate=494.52 samples/sec
  2024-03-14 04:03:02,073 INFO:   | Train Device=CSX, Step=4600, Loss=3.64062, Rate=493.95 samples/sec, GlobalRate=494.51 samples/sec
  2024-03-14 04:03:24,777 INFO:   | Train Device=CSX, Step=4700, Loss=3.59375, Rate=493.56 samples/sec, GlobalRate=494.48 samples/sec
  2024-03-14 04:03:47,407 INFO:   | Train Device=CSX, Step=4800, Loss=3.54688, Rate=494.38 samples/sec, GlobalRate=494.49 samples/sec
  2024-03-14 04:04:10,030 INFO:   | Train Device=CSX, Step=4900, Loss=3.51562, Rate=494.79 samples/sec, GlobalRate=494.50 samples/sec
  2024-03-14 04:04:32,711 INFO:   | Train Device=CSX, Step=5000, Loss=3.57812, Rate=494.21 samples/sec, GlobalRate=494.49 samples/sec
  2024-03-14 04:04:55,366 INFO:   | Train Device=CSX, Step=5100, Loss=3.56250, Rate=494.30 samples/sec, GlobalRate=494.48 samples/sec
  2024-03-14 04:05:18,030 INFO:   | Train Device=CSX, Step=5200, Loss=3.56250, Rate=494.22 samples/sec, GlobalRate=494.48 samples/sec
  2024-03-14 04:05:40,699 INFO:   | Train Device=CSX, Step=5300, Loss=3.50000, Rate=494.14 samples/sec, GlobalRate=494.47 samples/sec
  2024-03-14 04:06:03,349 INFO:   | Train Device=CSX, Step=5400, Loss=3.59375, Rate=494.34 samples/sec, GlobalRate=494.47 samples/sec
  2024-03-14 04:06:25,993 INFO:   | Train Device=CSX, Step=5500, Loss=3.53125, Rate=494.51 samples/sec, GlobalRate=494.47 samples/sec
  2024-03-14 04:06:48,645 INFO:   | Train Device=CSX, Step=5600, Loss=3.53125, Rate=494.46 samples/sec, GlobalRate=494.47 samples/sec
  2024-03-14 04:07:11,290 INFO:   | Train Device=CSX, Step=5700, Loss=3.50000, Rate=494.53 samples/sec, GlobalRate=494.47 samples/sec
  2024-03-14 04:07:33,965 INFO:   | Train Device=CSX, Step=5800, Loss=3.50000, Rate=494.18 samples/sec, GlobalRate=494.47 samples/sec
  2024-03-14 04:07:56,626 INFO:   | Train Device=CSX, Step=5900, Loss=3.57812, Rate=494.21 samples/sec, GlobalRate=494.46 samples/sec
  2024-03-14 04:08:19,300 INFO:   | Train Device=CSX, Step=6000, Loss=3.45312, Rate=494.07 samples/sec, GlobalRate=494.45 samples/sec
  2024-03-14 04:08:41,970 INFO:   | Train Device=CSX, Step=6100, Loss=3.45312, Rate=494.06 samples/sec, GlobalRate=494.45 samples/sec
  2024-03-14 04:09:04,613 INFO:   | Train Device=CSX, Step=6200, Loss=3.51562, Rate=494.39 samples/sec, GlobalRate=494.45 samples/sec
  2024-03-14 04:09:27,280 INFO:   | Train Device=CSX, Step=6300, Loss=3.53125, Rate=494.22 samples/sec, GlobalRate=494.44 samples/sec
  2024-03-14 04:09:49,988 INFO:   | Train Device=CSX, Step=6400, Loss=3.51562, Rate=493.63 samples/sec, GlobalRate=494.43 samples/sec
  2024-03-14 04:10:12,629 INFO:   | Train Device=CSX, Step=6500, Loss=3.50000, Rate=494.26 samples/sec, GlobalRate=494.43 samples/sec
  2024-03-14 04:10:35,364 INFO:   | Train Device=CSX, Step=6600, Loss=3.46875, Rate=493.28 samples/sec, GlobalRate=494.40 samples/sec
  2024-03-14 04:10:58,038 INFO:   | Train Device=CSX, Step=6700, Loss=3.53125, Rate=493.69 samples/sec, GlobalRate=494.40 samples/sec
  2024-03-14 04:11:20,737 INFO:   | Train Device=CSX, Step=6800, Loss=3.46875, Rate=493.52 samples/sec, GlobalRate=494.38 samples/sec
  2024-03-14 04:11:43,449 INFO:   | Train Device=CSX, Step=6900, Loss=3.46875, Rate=493.29 samples/sec, GlobalRate=494.36 samples/sec
  2024-03-14 04:12:06,103 INFO:   | Train Device=CSX, Step=7000, Loss=3.45312, Rate=493.95 samples/sec, GlobalRate=494.36 samples/sec
  2024-03-14 04:12:28,797 INFO:   | Train Device=CSX, Step=7100, Loss=3.46875, Rate=493.70 samples/sec, GlobalRate=494.35 samples/sec
  2024-03-14 04:12:51,466 INFO:   | Train Device=CSX, Step=7200, Loss=3.48438, Rate=493.91 samples/sec, GlobalRate=494.35 samples/sec
  2024-03-14 04:13:14,127 INFO:   | Train Device=CSX, Step=7300, Loss=3.46875, Rate=494.11 samples/sec, GlobalRate=494.35 samples/sec
  2024-03-14 04:13:36,823 INFO:   | Train Device=CSX, Step=7400, Loss=3.45312, Rate=493.73 samples/sec, GlobalRate=494.33 samples/sec
  2024-03-14 04:13:59,510 INFO:   | Train Device=CSX, Step=7500, Loss=3.43750, Rate=493.70 samples/sec, GlobalRate=494.33 samples/sec
  2024-03-14 04:14:22,166 INFO:   | Train Device=CSX, Step=7600, Loss=3.43750, Rate=494.09 samples/sec, GlobalRate=494.33 samples/sec
  2024-03-14 04:14:44,804 INFO:   | Train Device=CSX, Step=7700, Loss=3.43750, Rate=494.48 samples/sec, GlobalRate=494.33 samples/sec
  2024-03-14 04:15:07,454 INFO:   | Train Device=CSX, Step=7800, Loss=3.43750, Rate=494.49 samples/sec, GlobalRate=494.33 samples/sec
  2024-03-14 04:15:30,114 INFO:   | Train Device=CSX, Step=7900, Loss=3.45312, Rate=494.34 samples/sec, GlobalRate=494.33 samples/sec
  2024-03-14 04:15:52,855 INFO:   | Train Device=CSX, Step=8000, Loss=3.48438, Rate=493.24 samples/sec, GlobalRate=494.31 samples/sec
  2024-03-14 04:16:15,532 INFO:   | Train Device=CSX, Step=8100, Loss=3.39062, Rate=493.63 samples/sec, GlobalRate=494.30 samples/sec
  2024-03-14 04:16:38,261 INFO:   | Train Device=CSX, Step=8200, Loss=3.43750, Rate=493.12 samples/sec, GlobalRate=494.29 samples/sec
  2024-03-14 04:17:00,938 INFO:   | Train Device=CSX, Step=8300, Loss=3.42188, Rate=493.58 samples/sec, GlobalRate=494.28 samples/sec
  2024-03-14 04:17:23,619 INFO:   | Train Device=CSX, Step=8400, Loss=3.42188, Rate=493.72 samples/sec, GlobalRate=494.27 samples/sec
  2024-03-14 04:17:46,320 INFO:   | Train Device=CSX, Step=8500, Loss=3.35938, Rate=493.50 samples/sec, GlobalRate=494.26 samples/sec
  2024-03-14 04:18:09,055 INFO:   | Train Device=CSX, Step=8600, Loss=3.42188, Rate=492.99 samples/sec, GlobalRate=494.25 samples/sec
  2024-03-14 04:18:31,756 INFO:   | Train Device=CSX, Step=8700, Loss=3.46875, Rate=493.21 samples/sec, GlobalRate=494.23 samples/sec
  2024-03-14 04:18:54,496 INFO:   | Train Device=CSX, Step=8800, Loss=3.37500, Rate=492.80 samples/sec, GlobalRate=494.22 samples/sec
  2024-03-14 04:19:17,167 INFO:   | Train Device=CSX, Step=8900, Loss=3.46875, Rate=493.53 samples/sec, GlobalRate=494.21 samples/sec
  2024-03-14 04:19:39,861 INFO:   | Train Device=CSX, Step=9000, Loss=3.40625, Rate=493.53 samples/sec, GlobalRate=494.21 samples/sec
  2024-03-14 04:20:02,572 INFO:   | Train Device=CSX, Step=9100, Loss=3.42188, Rate=493.30 samples/sec, GlobalRate=494.19 samples/sec
  2024-03-14 04:20:25,233 INFO:   | Train Device=CSX, Step=9200, Loss=3.43750, Rate=493.86 samples/sec, GlobalRate=494.19 samples/sec
  2024-03-14 04:20:47,950 INFO:   | Train Device=CSX, Step=9300, Loss=3.45312, Rate=493.36 samples/sec, GlobalRate=494.18 samples/sec
  2024-03-14 04:21:10,617 INFO:   | Train Device=CSX, Step=9400, Loss=3.37500, Rate=493.82 samples/sec, GlobalRate=494.18 samples/sec
  2024-03-14 04:21:33,295 INFO:   | Train Device=CSX, Step=9500, Loss=3.39062, Rate=493.84 samples/sec, GlobalRate=494.18 samples/sec
  2024-03-14 04:21:55,995 INFO:   | Train Device=CSX, Step=9600, Loss=3.43750, Rate=493.57 samples/sec, GlobalRate=494.17 samples/sec
  2024-03-14 04:22:18,672 INFO:   | Train Device=CSX, Step=9700, Loss=3.42188, Rate=493.76 samples/sec, GlobalRate=494.17 samples/sec
  2024-03-14 04:22:41,398 INFO:   | Train Device=CSX, Step=9800, Loss=3.40625, Rate=493.20 samples/sec, GlobalRate=494.15 samples/sec
  2024-03-14 04:23:04,056 INFO:   | Train Device=CSX, Step=9900, Loss=3.40625, Rate=493.87 samples/sec, GlobalRate=494.15 samples/sec
  2024-03-14 04:23:26,729 INFO:   | Train Device=CSX, Step=10000, Loss=3.43750, Rate=493.93 samples/sec, GlobalRate=494.15 samples/sec
  2024-03-14 04:23:26,731 INFO:   Saving checkpoint at step 10000
  2024-03-14 04:23:46,836 INFO:   Saved checkpoint model_dir_gpt2_pytorch/checkpoint_10000.mdl
  2024-03-14 04:23:50,448 INFO:   | Train Device=CSX, Step=10100, Loss=3.39062, Rate=480.89 samples/sec, GlobalRate=493.93 samples/sec
  2024-03-14 04:24:12,138 INFO:   | Train Device=CSX, Step=10200, Loss=3.39062, Rate=502.19 samples/sec, GlobalRate=494.14 samples/sec
  2024-03-14 04:24:34,807 INFO:   | Train Device=CSX, Step=10300, Loss=3.40625, Rate=497.31 samples/sec, GlobalRate=494.14 samples/sec
  2024-03-14 04:24:57,508 INFO:   | Train Device=CSX, Step=10400, Loss=3.42188, Rate=494.95 samples/sec, GlobalRate=494.13 samples/sec
  2024-03-14 04:25:20,189 INFO:   | Train Device=CSX, Step=10500, Loss=3.35938, Rate=494.25 samples/sec, GlobalRate=494.12 samples/sec
  2024-03-14 04:25:42,865 INFO:   | Train Device=CSX, Step=10600, Loss=3.34375, Rate=494.05 samples/sec, GlobalRate=494.12 samples/sec
  2024-03-14 04:26:05,558 INFO:   | Train Device=CSX, Step=10700, Loss=3.35938, Rate=493.75 samples/sec, GlobalRate=494.12 samples/sec
  2024-03-14 04:26:28,244 INFO:   | Train Device=CSX, Step=10800, Loss=3.32812, Rate=493.71 samples/sec, GlobalRate=494.11 samples/sec
  2024-03-14 04:26:50,959 INFO:   | Train Device=CSX, Step=10900, Loss=3.35938, Rate=493.33 samples/sec, GlobalRate=494.10 samples/sec
  2024-03-14 04:27:13,636 INFO:   | Train Device=CSX, Step=11000, Loss=3.35938, Rate=493.66 samples/sec, GlobalRate=494.10 samples/sec
  2024-03-14 04:27:36,385 INFO:   | Train Device=CSX, Step=11100, Loss=3.34375, Rate=492.86 samples/sec, GlobalRate=494.09 samples/sec
  2024-03-14 04:27:59,096 INFO:   | Train Device=CSX, Step=11200, Loss=3.32812, Rate=493.04 samples/sec, GlobalRate=494.08 samples/sec
    ```
  </details>