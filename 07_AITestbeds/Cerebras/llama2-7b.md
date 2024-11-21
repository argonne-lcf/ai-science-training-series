# LLAMA2-7B on Cerebras

##### Go to directory with llama2-7b example. 
```bash
cd ~/R_2.3.0/modelzoo/src/cerebras/modelzoo/models/nlp/llama
```

#####  Activate PyTorch virtual Environment 
```bash
source ~/R_2.3.0/venv_cerebras_pt/bin/activate
```

#####  Replace config file with correct configurations file. 
```bash
cp /software/cerebras/dataset/params_llama_7b.yaml configs/params_llama_7b.yaml
```

#####  Run Training Job
```bash
export MODEL_DIR=model_dir_llamma
if [ -d "$MODEL_DIR" ]; then rm -Rf $MODEL_DIR; fi

python run.py CSX \
  --job_labels name=llama_7b \
  --params configs/params_llama_7b.yaml \
  --num_csx=1 \
  --mode train \
  --model_dir $MODEL_DIR \
  --mount_dirs /projects /home/ /software \
  --python_paths /home/$(whoami)/R_2.3.0/modelzoo/src \
  --compile_dir $(whoami) |& tee mytest.log
```
<details>
  <summary>Sample Output</summary>
  
  ```bash
    2024-03-21 14:40:57,949 INFO:   Effective batch size is 99.
2024-03-21 14:40:57,970 INFO:   Checkpoint autoloading is enabled. Looking for latest checkpoint in "/srv/projects/datascience/vsastry/model_dir_llama/" directory with the following naming convention: `checkpoint_(step)(_timestamp)?.mdl`.
2024-03-21 14:40:57,971 INFO:   No checkpoints were found in "/srv/projects/datascience/vsastry/model_dir_llama/".
2024-03-21 14:40:57,971 INFO:   No checkpoint was provided. Using randomly initialized model parameters.
2024-03-21 14:40:59,419 INFO:   Saving checkpoint at step 0
2024-03-21 14:48:46,988 INFO:   Saved checkpoint /srv/projects/datascience/vsastry/model_dir_llama/checkpoint_0.mdl
2024-03-21 14:49:05,547 INFO:   Compiling the model. This may take a few minutes.
2024-03-21 14:49:05,550 INFO:   Defaulted to use the job-operator namespace as the usernode config /opt/cerebras/config_v2 only has access to that namespace.
2024-03-21 14:49:06,819 INFO:   Initiating a new image build job against the cluster server.
2024-03-21 14:49:06,898 INFO:   Custom worker image build is disabled from server.
2024-03-21 14:49:06,911 INFO:   Defaulted to use the job-operator namespace as the usernode config /opt/cerebras/config_v2 only has access to that namespace.
2024-03-21 14:49:07,143 INFO:   Initiating a new compile wsjob against the cluster server.
2024-03-21 14:49:07,226 INFO:   compile job id: wsjob-pg4gslxvgsalvh6ppdvydb, remote log path: /n1/wsjob/workdir/job-operator/wsjob-pg4gslxvgsalvh6ppdvydb
2024-03-21 14:49:17,259 INFO:   Poll ingress status: Waiting for job running, current job status: Queueing, msg: job is queueing. Job queue status: current job is top of queue but likely blocked by running jobs, 1 compile job(s) running using 67Gi memory. For more information, please run 'csctl get jobs'.
2024-03-21 15:02:07,673 INFO:   Poll ingress status: Waiting for job running, current job status: Queueing, msg: job is queueing. Job queue status: current job is top of queue but likely blocked by running jobs, 1 execute job(s) running using 1 system(s), 1 compile job(s) running using 67Gi memory. For more information, please run 'csctl get jobs'.
2024-03-21 15:02:17,683 INFO:   Poll ingress status: Waiting for job service readiness.
2024-03-21 15:02:47,717 INFO:   Ingress is ready: Job ingress ready, poll ingress success.
2024-03-21 15:02:58,509 INFO:   Pre-optimization transforms...
2024-03-21 15:03:14,815 INFO:   Optimizing layouts and memory usage...
2024-03-21 15:03:14,839 INFO:   Gradient accumulation enabled
2024-03-21 15:03:14,840 WARNING:   Gradient accumulation will search for an optimal micro batch size based on internal performance models, which can lead to an increased compile time. Specify `micro_batch_size` option in the 'train_input/eval_input' section of your .yaml parameter file to set the gradient accumulation microbatch size, if an optimal microbatch size is known.

2024-03-21 15:03:14,842 INFO:   Gradient accumulation trying sub-batch size 3...
2024-03-21 15:03:21,632 INFO:   Exploring floorplans
2024-03-21 15:03:30,198 INFO:   Exploring data layouts
2024-03-21 15:03:50,589 INFO:   Optimizing memory usage
2024-03-21 15:05:23,008 INFO:   Gradient accumulation trying sub-batch size 33...
2024-03-21 15:05:30,532 INFO:   Exploring floorplans
2024-03-21 15:05:37,304 INFO:   Exploring data layouts
2024-03-21 15:06:11,327 INFO:   Optimizing memory usage
2024-03-21 15:11:37,204 INFO:   Gradient accumulation trying sub-batch size 9...
2024-03-21 15:11:44,383 INFO:   Exploring floorplans
2024-03-21 15:11:50,639 INFO:   Exploring data layouts
2024-03-21 15:12:16,120 INFO:   Optimizing memory usage
2024-03-21 15:15:59,788 INFO:   Gradient accumulation trying sub-batch size 11...
2024-03-21 15:16:06,314 INFO:   Exploring floorplans
2024-03-21 15:16:12,563 INFO:   Exploring data layouts
2024-03-21 15:16:40,965 INFO:   Optimizing memory usage
2024-03-21 15:21:03,938 INFO:   Exploring floorplans
2024-03-21 15:21:10,918 INFO:   Exploring data layouts
2024-03-21 15:22:03,953 INFO:   Optimizing memory usage
2024-03-21 15:30:35,456 INFO:   No benefit from gradient accumulation expected. Compile will proceed at original per-box batch size 99 with 9 lanes

2024-03-21 15:30:35,540 INFO:   Post-layout optimizations...
2024-03-21 15:32:11,639 INFO:   Allocating buffers...
2024-03-21 15:32:18,023 INFO:   Code generation...
2024-03-21 15:32:53,573 INFO:   Compiling image...
2024-03-21 15:32:53,578 INFO:   Compiling kernels
2024-03-21 15:34:39,222 INFO:   Compiling final image
2024-03-21 15:36:54,995 INFO:   Compile artifacts successfully written to remote compile directory. Compile hash is: cs_2599085507768189065
2024-03-21 15:36:55,146 INFO:   Heartbeat thread stopped for wsjob-pg4gslxvgsalvh6ppdvydb.
2024-03-21 15:36:55,160 INFO:   Compile was successful!
2024-03-21 15:36:55,171 INFO:   Programming Cerebras Wafer Scale Cluster for execution. This may take a few minutes.
2024-03-21 15:36:56,403 INFO:   Defaulted to use the job-operator namespace as the usernode config /opt/cerebras/config_v2 only has access to that namespace.
2024-03-21 15:36:56,659 INFO:   Initiating a new execute wsjob against the cluster server.
2024-03-21 15:36:56,758 INFO:   execute job id: wsjob-bdcvvsrwely3kbfwduefqx, remote log path: /n1/wsjob/workdir/job-operator/wsjob-bdcvvsrwely3kbfwduefqx
2024-03-21 15:37:06,789 INFO:   Poll ingress status: Waiting for job running, current job status: Scheduled, msg: job is scheduled. 
2024-03-21 15:37:16,793 INFO:   Poll ingress status: Waiting for job service readiness.
2024-03-21 15:37:36,838 INFO:   Poll ingress status: Waiting for job ingress readiness.
2024-03-21 15:37:46,861 INFO:   Ingress is ready: Job ingress ready, poll ingress success.
2024-03-21 15:37:47,052 INFO:   Preparing to execute using 1 CSX
2024-03-21 15:38:33,999 INFO:   About to send initial weights
2024-03-21 15:40:01,150 INFO:   Finished sending initial weights
2024-03-21 15:40:01,154 INFO:   Finalizing appliance staging for the run
2024-03-21 15:40:01,203 INFO:   Waiting for device programming to complete
2024-03-21 15:41:26,576 INFO:   Device programming is complete
2024-03-21 15:41:27,888 INFO:   Using network type: ROCE
2024-03-21 15:41:27,890 INFO:   Waiting for input workers to prime the data pipeline and begin streaming ...
2024-03-21 15:41:27,942 INFO:   Input workers have begun streaming input data
2024-03-21 15:41:45,009 INFO:   Appliance staging is complete
2024-03-21 15:41:45,021 INFO:   Beginning appliance run
2024-03-21 15:49:45,474 INFO:   | Train Device=CSX, Step=100, Loss=9.84375, Rate=20.61 samples/sec, GlobalRate=20.61 samples/sec
2024-03-21 15:57:49,616 INFO:   | Train Device=CSX, Step=200, Loss=8.35938, Rate=20.51 samples/sec, GlobalRate=20.53 samples/sec
2024-03-21 16:05:53,769 INFO:   | Train Device=CSX, Step=300, Loss=8.26562, Rate=20.47 samples/sec, GlobalRate=20.50 samples/sec
2024-03-21 16:13:58,078 INFO:   | Train Device=CSX, Step=400, Loss=7.02344, Rate=20.45 samples/sec, GlobalRate=20.49 samples/sec
2024-03-21 16:22:02,644 INFO:   | Train Device=CSX, Step=500, Loss=7.07812, Rate=20.44 samples/sec, GlobalRate=20.48 samples/sec
2024-03-21 16:30:06,513 INFO:   | Train Device=CSX, Step=600, Loss=7.34375, Rate=20.45 samples/sec, GlobalRate=20.47 samples/sec
2024-03-21 16:38:10,737 INFO:   | Train Device=CSX, Step=700, Loss=7.19531, Rate=20.45 samples/sec, GlobalRate=20.47 samples/sec
2024-03-21 16:46:15,052 INFO:   | Train Device=CSX, Step=800, Loss=6.52344, Rate=20.44 samples/sec, GlobalRate=20.47 samples/sec
2024-03-21 16:54:19,448 INFO:   | Train Device=CSX, Step=900, Loss=6.46875, Rate=20.44 samples/sec, GlobalRate=20.46 samples/sec
2024-03-21 17:02:24,111 INFO:   | Train Device=CSX, Step=1000, Loss=5.98438, Rate=20.43 samples/sec, GlobalRate=20.46 samples/sec
2024-03-21 17:10:28,632 INFO:   | Train Device=CSX, Step=1100, Loss=6.17188, Rate=20.43 samples/sec, GlobalRate=20.46 samples/sec
2024-03-21 17:18:32,943 INFO:   | Train Device=CSX, Step=1200, Loss=6.04688, Rate=20.44 samples/sec, GlobalRate=20.46 samples/sec
2024-03-21 17:26:37,241 INFO:   | Train Device=CSX, Step=1300, Loss=5.54688, Rate=20.44 samples/sec, GlobalRate=20.45 samples/sec
2024-03-21 17:34:41,491 INFO:   | Train Device=CSX, Step=1400, Loss=5.92188, Rate=20.44 samples/sec, GlobalRate=20.45 samples/sec
2024-03-21 17:42:45,646 INFO:   | Train Device=CSX, Step=1500, Loss=5.68750, Rate=20.45 samples/sec, GlobalRate=20.45 samples/sec
2024-03-21 17:50:50,110 INFO:   | Train Device=CSX, Step=1600, Loss=5.85938, Rate=20.44 samples/sec, GlobalRate=20.45 samples/sec
  ```
</details>
