# BERT-Large on Cerebras

* Go to directory with the BERT example. 
  ```bash
  cd ~/R_2.0.3/modelzoo/modelzoo/transformers/pytorch/bert
  ```

* Activate PyTroch virtual Environment 
  ```bash
  source ~/R_2.0.3/venv_cerebras_pt/bin/activate
  ```

* Replace config file with correct configurations file. 
  Commonly used various pre-processed datasets are available the systems for your conveneince. Config file needs to be changed to point to correct data path. For your convenience, modified config files are available. Copy them to replace config file in current directory.
  ```bash
  cp /software/cerebras/dataset/bert_large/bert_large_MSL128_sampleds.yaml configs/bert_large_MSL128_sampleds.yaml
  ```

* Run Training Job
  ```bash
  # If MODEL_DIR already exisits from previous runs, delete it. 
  export MODEL_DIR=model_dir_bert_large_pytorch
  if [ -d "$MODEL_DIR" ]; then rm -Rf $MODEL_DIR; fi

  python run.py CSX --job_labels name=bert_pt \
  --params configs/bert_large_MSL128_sampleds.yaml \
  --num_workers_per_csx=1 --mode train \
  --model_dir $MODEL_DIR --mount_dirs /home/ /software/ \
  --python_paths /home/$(whoami)/R_2.0.3/modelzoo/ \
  --compile_dir $(whoami) |& tee mytest.log
  ```
  <details>
    <summary>Sample Output</summary>
    
    ```bash
  $ python run.py CSX --job_labels name=bert_pt --params configs/bert_large_MSL128_sampleds.yaml --num_workers_per_csx=1 --mode train --model_dir $MODEL_DIR --mount_dirs /home/ /software/ --python_paths /home/$(whoami)/R_2.0.3/modelzoo/ --compile_dir $(whoami) |& tee mytest.log
  Initializing module BertForPreTrainingModel.perplexity_metric: [00:06, note=Initializing parameters]2024-03-14 02:59:32,596 INFO:   Effective batch size is 1024.
  2024-03-14 02:59:32,628 INFO:   Checkpoint autoloading is enabled. Looking for latest checkpoint in "model_dir_bert_large_pytorch" directory with the following naming convention: `checkpoint_(step)(_timestamp)?.mdl`.
  2024-03-14 02:59:32,629 INFO:   No checkpoints were found in "model_dir_bert_large_pytorch".
  2024-03-14 02:59:32,630 INFO:   No checkpoint was provided. Using randomly initialized model parameters.
  Initializing module BertForPreTrainingModel.perplexity_metric: [00:07, note=Initializing parameters]2024-03-14 02:59:32,782 INFO:   Saving checkpoint at step 0
  2024-03-14 02:59:56,348 INFO:   Saved checkpoint model_dir_bert_large_pytorch/checkpoint_0.mdl

  2024-03-14 03:00:06,793 INFO:   Compiling the model. This may take a few minutes.
  2024-03-14 03:00:08,077 INFO:   Initiating a new image build job against the cluster server.
  2024-03-14 03:00:08,144 INFO:   Custom worker image build is disabled from server.
  2024-03-14 03:00:08,417 INFO:   Initiating a new compile wsjob against the cluster server.
  2024-03-14 03:00:08,492 INFO:   compile job id: wsjob-rc49spjqdebcxj4donzdoy, remote log path: /n1/wsjob/workdir/wsjob-rc49spjqdebcxj4donzdoy
  2024-03-14 03:00:18,518 INFO:   Poll ingress status: Waiting for coordinator to be ready.
  2024-03-14 03:00:48,539 INFO:   Ingress is ready: Coordinator service ready, poll ingress success.
  2024-03-14 03:00:55,072 INFO:   Pre-optimization transforms...
  2024-03-14 03:00:56,214 INFO:   Optimizing layouts and memory usage...
  2024-03-14 03:00:56,224 INFO:   Gradient accumulation enabled
  2024-03-14 03:00:56,225 WARNING:   Gradient accumulation will search for an optimal micro batch size based on internal performance models, which can lead to an increased compile time. Specify `micro_batch_size` option in the 'train_input/eval_input' section of your .yaml parameter file to set the gradient accumulation microbatch size, if an optimal microbatch size is known.

  2024-03-14 03:00:56,227 INFO:   Gradient accumulation trying sub-batch size 8...
  2024-03-14 03:00:57,754 INFO:   Exploring floorplans
  2024-03-14 03:01:09,048 INFO:   Exploring data layouts
  2024-03-14 03:01:20,016 INFO:   Optimizing memory usage
  2024-03-14 03:01:41,237 INFO:   Gradient accumulation trying sub-batch size 128...
  2024-03-14 03:01:42,198 INFO:   Exploring floorplans
  2024-03-14 03:02:48,936 INFO:   Exploring data layouts
  2024-03-14 03:03:12,695 INFO:   Optimizing memory usage
  2024-03-14 03:03:34,141 INFO:   Gradient accumulation trying sub-batch size 32...
  2024-03-14 03:03:35,095 INFO:   Exploring floorplans
  2024-03-14 03:03:45,696 INFO:   Exploring data layouts
  2024-03-14 03:04:07,965 INFO:   Optimizing memory usage
  2024-03-14 03:04:31,229 INFO:   Gradient accumulation trying sub-batch size 256...
  2024-03-14 03:04:32,162 INFO:   Exploring floorplans
  2024-03-14 03:05:04,667 INFO:   Exploring data layouts
  2024-03-14 03:05:29,748 INFO:   Optimizing memory usage
  2024-03-14 03:05:59,501 INFO:   Gradient accumulation trying sub-batch size 64...
  2024-03-14 03:06:01,024 INFO:   Exploring floorplans
  2024-03-14 03:06:20,281 INFO:   Exploring data layouts
  2024-03-14 03:06:43,031 INFO:   Optimizing memory usage
  2024-03-14 03:07:02,906 INFO:   Gradient accumulation trying sub-batch size 512...
  2024-03-14 03:07:03,941 INFO:   Exploring floorplans
  2024-03-14 03:07:37,563 INFO:   Exploring data layouts
  2024-03-14 03:08:07,246 INFO:   Optimizing memory usage
  2024-03-14 03:08:33,514 INFO:   Exploring floorplans
  2024-03-14 03:08:59,225 INFO:   Exploring data layouts
  2024-03-14 03:09:32,121 INFO:   Optimizing memory usage
  2024-03-14 03:10:18,092 INFO:   Post-layout optimizations...
  2024-03-14 03:10:23,067 INFO:   Allocating buffers...
  2024-03-14 03:10:25,499 INFO:   Code generation...
  2024-03-14 03:10:41,290 INFO:   Compiling image...
  2024-03-14 03:10:41,296 INFO:   Compiling kernels
  2024-03-14 03:12:31,593 INFO:   Compiling final image
  2024-03-14 03:14:59,489 INFO:   Compile artifacts successfully written to remote compile directory. Compile hash is: cs_9514045510034748708
  2024-03-14 03:14:59,549 INFO:   Heartbeat thread stopped for wsjob-rc49spjqdebcxj4donzdoy.
  2024-03-14 03:14:59,552 INFO:   Compile was successful!
  2024-03-14 03:14:59,558 INFO:   Programming Cerebras Wafer Scale Cluster for execution. This may take a few minutes.
  2024-03-14 03:15:02,614 INFO:   Initiating a new execute wsjob against the cluster server.
  2024-03-14 03:15:02,700 INFO:   execute job id: wsjob-efefq53ashmq3lvggcvb6a, remote log path: /n1/wsjob/workdir/wsjob-efefq53ashmq3lvggcvb6a
  2024-03-14 03:15:12,726 INFO:   Poll ingress status: Waiting for job running, current job status: Scheduled, msg: lock grant success. 
  2024-03-14 03:15:32,738 INFO:   Poll ingress status: Waiting for coordinator to be ready.
  2024-03-14 03:16:12,813 INFO:   Ingress is ready: Coordinator service ready, poll ingress success.
  2024-03-14 03:16:12,959 INFO:   Preparing to execute using 1 CSX
  2024-03-14 03:16:35,212 INFO:   About to send initial weights
  2024-03-14 03:16:56,637 INFO:   Finished sending initial weights
  2024-03-14 03:16:56,639 INFO:   Finalizing appliance staging for the run
  2024-03-14 03:16:56,673 INFO:   Waiting for device programming to complete
  2024-03-14 03:18:42,151 INFO:   Device programming is complete
  2024-03-14 03:18:42,153 INFO:   Waiting for input workers to prime the data pipeline and begin streaming ...
  2024-03-14 03:18:42,195 INFO:   Input workers have begun streaming input data
  2024-03-14 03:18:46,091 INFO:   Appliance staging is complete
  2024-03-14 03:18:46,095 INFO:   Beginning appliance run
  2024-03-14 03:20:31,277 INFO:   | Train Device=CSX, Step=100, Loss=9.50000, Rate=974.00 samples/sec, GlobalRate=974.00 samples/sec
  2024-03-14 03:22:24,890 INFO:   | Train Device=CSX, Step=200, Loss=8.37500, Rate=930.38 samples/sec, GlobalRate=936.24 samples/sec
  2024-03-14 03:23:53,223 INFO:   | Train Device=CSX, Step=300, Loss=7.96875, Rate=1067.70 samples/sec, GlobalRate=1000.39 samples/sec
  2024-03-14 03:25:25,644 INFO:   | Train Device=CSX, Step=400, Loss=7.56250, Rate=1091.87 samples/sec, GlobalRate=1025.28 samples/sec
  2024-03-14 03:27:10,421 INFO:   | Train Device=CSX, Step=500, Loss=7.50000, Rate=1023.13 samples/sec, GlobalRate=1015.31 samples/sec
  2024-03-14 03:28:42,669 INFO:   | Train Device=CSX, Step=600, Loss=7.37500, Rate=1075.29 samples/sec, GlobalRate=1029.97 samples/sec
  2024-03-14 03:30:02,497 INFO:   | Train Device=CSX, Step=700, Loss=7.37500, Rate=1199.77 samples/sec, GlobalRate=1059.80 samples/sec
  2024-03-14 03:31:09,744 INFO:   | Train Device=CSX, Step=800, Loss=7.25000, Rate=1393.54 samples/sec, GlobalRate=1101.67 samples/sec
  2024-03-14 03:32:46,360 INFO:   | Train Device=CSX, Step=900, Loss=7.21875, Rate=1193.34 samples/sec, GlobalRate=1096.86 samples/sec
  2024-03-14 03:34:06,233 INFO:   | Train Device=CSX, Step=1000, Loss=7.12500, Rate=1246.55 samples/sec, GlobalRate=1112.94 samples/sec
  2024-03-14 03:34:06,236 INFO:   Saving checkpoint at step 1000
  2024-03-14 03:34:46,870 INFO:   Saved checkpoint model_dir_bert_large_pytorch/checkpoint_1000.mdl
  2024-03-14 03:35:51,737 INFO:   Heartbeat thread stopped for wsjob-efefq53ashmq3lvggcvb6a.
  2024-03-14 03:35:51,746 INFO:   Training completed successfully!
  2024-03-14 03:35:51,747 INFO:   Processed 1024000 sample(s) in 1025.603128781 seconds.
    ```
  </details>