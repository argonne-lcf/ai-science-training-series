# GPT2 on Graphcore 

These instructions are to train a GPT-2 pytorch model on the POD16. 

* Go to direcotry with GPT2 example 
  ```bash
  cd ~/graphcore/examples/nlp/gpt2/pytorch
  ```

* Create a new PopTorch Environment 
  ```bash
  POPLAR_SDK_ROOT=/software/graphcore/poplar_sdk/3.3.0/
  export POPLAR_SDK_ROOT=$POPLAR_SDK_ROOT

  virtualenv ~/venvs/graphcore/poptorch33_gpt2
  source ~/venvs/graphcore/poptorch33_gpt2/bin/activate
  pip install $POPLAR_SDK_ROOT/poptorch-3.3.0+113432_960e9c294b_ubuntu_20_04-cp38-cp38-linux_x86_64.whl
  export PYTHONPATH=$POPLAR_SDK_ROOT/python:$PYTHONPATH
  ```

* Install Requirements 

  ```bash
  pip3 install -r requirements.txt
  ```

* Run GPT2 on 4 IPUs (single Instance)
  ```bash
  /opt/slurm/bin/srun --ipus=4 python /home/$USER/graphcore/examples/nlp/gpt2/pytorch/train_gpt2.py \
  --model gpt2 --ipus-per-replica 4 --replication-factor 1 \
  --gradient-accumulation 2048 --device-iterations 8 \
  --batch-size 1 --layers-per-ipu 0 4 4 4 \
  --matmul-proportion 0.15 0.15 0.15 0.15 --max-len 1024 \
  --optimizer AdamW --learning-rate 0.00015 \
  --lr-schedule cosine --lr-warmup 0.01 \
  --remap-logit True --enable-sequence-serialized True \
  --embedding-serialization-factor 4 --recompute-checkpoint-every-layer True \
  --enable-half-partials True --replicated-tensor-sharding True \
  --dataset 'generated' --epochs 1
  ```

* Run GPT2 on 16 IPUs (4 Instances)
  ```bash
  /opt/slurm/bin/srun --ipus=16 python /home/$USER/graphcore/examples/nlp/gpt2/pytorch/train_gpt2.py --model gpt2 --ipus-per-replica 4 --replication-factor 4 --gradient-accumulation 2048 --device-iterations 8 --batch-size 1 --layers-per-ipu 0 4 4 4 --matmul-proportion 0.15 0.15 0.15 0.15 --max-len 1024 --optimizer AdamW --learning-rate 0.00015 --lr-schedule cosine --lr-warmup 0.01 --remap-logit True --enable-sequence-serialized True --embedding-serialization-factor 4 --recompute-checkpoint-every-layer True --enable-half-partials True --replicated-tensor-sharding True --dataset 'generated' --epochs 1
  ```
  <details>
    <summary>Sample Output</summary>
    
    ```bash
      srun: job 10697 queued and waiting for resources
      srun: job 10697 has been allocated resources
      Building (if necessary) and loading remap_tensor_ce.
      Failed to find compiled extension; rebuilding.
      Building (if necessary) and loading residual_add_inplace_pattern.
      Model initializing
      -------------------- Device Allocation --------------------
      Embedding  --> IPU 0
      Layer 0  --> IPU 1
      Layer 1  --> IPU 1
      Layer 2  --> IPU 1
      Layer 3  --> IPU 1
      Layer 4  --> IPU 2
      Layer 5  --> IPU 2
      Layer 6  --> IPU 2
      Layer 7  --> IPU 2
      Layer 8  --> IPU 3
      Layer 9  --> IPU 3
      Layer 10 --> IPU 3
      Layer 11 --> IPU 3
      LM_head --> IPU 0

      step 0 of epoch 0, loss: 10.913220405578613, acc: 2.0071864128112793e-05, lr: 0.00012803300858899104, throughput: 646.8439205981404 samples/sec
      step 1 of epoch 0, loss: 10.836345672607422, acc: 1.9788742065429688e-05, lr: 7.5e-05, throughput: 1058.0979097185766 samples/sec
      step 2 of epoch 0, loss: 10.831247329711914, acc: 2.0518898963928223e-05, lr: 2.1966991411008938e-05, throughput: 1058.7595523807183 samples/sec
      step 3 of epoch 0, loss: 10.829034805297852, acc: 1.990795135498047e-05, lr: 0.0, throughput: 1059.6762623043378 samples/sec
    ```
  </details>

