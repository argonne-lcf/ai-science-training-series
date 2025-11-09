# LLAMA2-7B on Cerebras

##### Go to directory with llama2-7b example. 
```bash
cd ~/R_2.6.0/modelzoo/src/cerebras/modelzoo/models/nlp/llama
```

#####  Activate PyTorch virtual Environment 
```bash
source ~/R_2.6.0/venv_cerebras_pt/bin/activate
```



#####  Replace config file with correct configurations file. 
```bash
cp /software/cerebras/dataset/params_llama2_7b.yaml configs/params_llama2_7b.yaml
```

#####  Run Training Job
```bash
export MODEL_DIR=model_dir_llama2_7b
if [ -d "$MODEL_DIR" ]; then rm -Rf $MODEL_DIR; fi
cszoo fit configs/params_llama2_7b.yaml --job_labels name=llama2_7b --model_dir model_dir_llama2_7b |& tee mytest.log
```
<details>
  <summary>Sample Output</summary>
  
  ```bash
2025-10-13 14:47:37,651 INFO:   Found existing cached compile with hash: "cs_16053036657376785725"
2025-10-13 14:47:41,091 INFO:   Compile artifacts successfully written to remote compile directory. Compile hash is: cs_16053036657376785725
2025-10-13 14:47:46,918 INFO:   Compile was successful!
2025-10-13 14:47:46,918 INFO:   Waiting for weight initialization to complete
2025-10-13 14:47:46,918 INFO:   Programming Cerebras Wafer Scale Cluster for execution. This may take a few minutes.
2025-10-13 14:47:49,008 INFO:   Initiating a new execute wsjob against the cluster server.
2025-10-13 14:47:49,037 INFO:   Job id: wsjob-kgqvqqxnp9zvpmwulcbxuj, workflow id: wflow-cxj2gwf7idcfanryokatnn, namespace: job-operator, remote log path: /n1/wsjob/workdir/job-operator/wsjob-kgqvqqxnp9zvpmwulcbxuj
2025-10-13 14:48:09,058 INFO:   Poll ingress status: Waiting for all Activation pods to be running, current running: 0/24.
2025-10-13 14:48:09,078 WARNING:   Event 2025-10-13 14:47:50 +0000 UTC reason=InconsistentVersion wsjob=wsjob-kgqvqqxnp9zvpmwulcbxuj message='Warning: job image version 2.5.1-202507111115-6-48e76807 is inconsistent with cluster server version 3.0.1-202508200300-150-bba1322a+bba1322aed, there's a risk job could fail due to inconsistent setup.'
2025-10-13 14:48:19,088 INFO:   Poll ingress status: Waiting for all Weight pods to be running, current running: 17/18.
2025-10-13 14:48:29,112 INFO:   Poll ingress status: Waiting for job ingress readiness.
2025-10-13 14:48:39,135 INFO:   Poll ingress status: Job ingress ready, dashboard: https://grafana.anl0.cerebras.internal/d/WebHNShVz/wsjob-dashboard?orgId=1&var-wsjob=wsjob-kgqvqqxnp9zvpmwulcbxuj&from=1760366287000&to=now
2025-10-13 14:48:39,149 INFO:   Poll ingress success: Job ingress ready, dashboard: https://grafana.anl0.cerebras.internal/d/WebHNShVz/wsjob-dashboard?orgId=1&var-wsjob=wsjob-kgqvqqxnp9zvpmwulcbxuj&from=1760366287000&to=now
2025-10-13 14:48:39,240 INFO:   Preparing to execute using 1 CSX
2025-10-13 14:49:14,926 INFO:   About to send initial weights
2025-10-13 14:49:28,149 INFO:   Finished sending initial weights
2025-10-13 14:49:28,150 INFO:   Finalizing appliance staging for the run
2025-10-13 14:49:28,158 INFO:   Waiting for device programming to complete
2025-10-13 14:53:20,585 INFO:   Device programming is complete
2025-10-13 14:53:21,628 INFO:   Using network type: ROCE
2025-10-13 14:53:21,629 INFO:   Waiting for input workers to prime the data pipeline and begin streaming ...
2025-10-13 14:53:21,637 INFO:   Input workers have begun streaming input data
2025-10-13 14:53:22,791 INFO:   Appliance staging is complete
2025-10-13 14:53:22,791 INFO:   Beginning appliance run
2025-10-13 15:20:04,385 INFO:   | Train Device=CSX, Step=50, Loss=7.67126, Rate=31.97 samples/sec, GlobalRate=31.97 samples/sec, LoopTimeRemaining=1:20:41, TimeRemaining=1:20:41
2025-10-13 15:46:44,894 INFO:   | Train Device=CSX, Step=100, Loss=7.05889, Rate=31.98 samples/sec, GlobalRate=31.98 samples/sec, LoopTimeRemaining=0:54:01, TimeRemaining=0:54:01
2025-10-13 16:13:25,156 INFO:   | Train Device=CSX, Step=150, Loss=6.53423, Rate=31.99 samples/sec, GlobalRate=31.98 samples/sec, LoopTimeRemaining=0:27:20, TimeRemaining=0:27:20
2025-10-13 16:40:05,444 INFO:   | Train Device=CSX, Step=200, Loss=6.09834, Rate=31.97 samples/sec, GlobalRate=31.99 samples/sec, LoopTimeRemaining=0:00:40, TimeRemaining=0:00:40
2025-10-13 16:40:05,450 INFO:   Saving checkpoint at step 200
2025-10-13 16:47:49,916 INFO:   Saved checkpoint model_dir_llama2_7b/checkpoint_200.mdl
2025-10-13 16:48:01,419 INFO:   Training completed successfully!
2025-10-13 16:48:01,425 INFO:   Processed 204800 training sample(s) in 7303.586917439 seconds.
  ```
</details>
