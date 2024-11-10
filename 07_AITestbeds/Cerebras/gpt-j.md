# BERT-Large on Cerebras

* Go to the directory with the GPT-J example. 
  ```bash
  cd ~/R_2.3.0/modelzoo/modelzoo/transformers/pytorch/gptj
  ```

* Activate PyTroch virtual Environment 
  ```bash
  source ~/R_2.3.0/venv_pt/bin/activate
  ```

* Replace config file with correct configurations file. 
  ```bash
  cp /software/cerebras/dataset/gptj/params_gptj_6B_sampleds.yaml configs/params_gptj_6B_sampleds.yaml
  ```

* Run Training Job
  ```bash
  export MODEL_DIR=model_dir_gptj_pytorch
  if [ -d "$MODEL_DIR" ]; then rm -Rf $MODEL_DIR; fi
  python run.py CSX --job_labels name=gptj_pt --params configs/params_gptj_6B_sampleds.yaml --num_workers_per_csx=1 --mode train --model_dir $MODEL_DIR --mount_dirs /home/ /software/ --python_paths /home/$(whoami)/R_1.9.2/modelzoo/ --compile_dir $(whoami) |& tee mytest.log
  ```
  <details>
    <summary>Sample Output (last section)</summary>
    
    ```bash
      2023-11-10 20:34:41,113 INFO:   Finished sending initial weights
      2023-11-10 20:34:41,116 INFO:   Finalizing appliance staging for the run
      2023-11-10 20:34:42,548 INFO:   Finished staging the appliance
      2023-11-10 20:34:42,552 INFO:   Beginning appliance run
      2023-11-10 20:40:15,694 INFO:   | Train Device=xla:0, Step=100, Loss=9.18750, Rate=19.81 samples/sec, GlobalRate=19.81 samples/sec
      2023-11-10 20:45:49,090 INFO:   | Train Device=xla:0, Step=200, Loss=8.37500, Rate=19.80 samples/sec, GlobalRate=19.80 samples/sec
      2023-11-10 20:45:49,092 INFO:   Saving checkpoint at global step 200
      2023-11-10 20:56:38,458 INFO:   Saving step 200 in dataloader checkpoint
      2023-11-10 20:56:38,575 INFO:   Saved checkpoint at global step: 200
      2023-11-10 20:56:38,576 INFO:   Training completed successfully! Processed 13200 sample(s) in 1316.022207736969 seconds.
    ```
  </details>