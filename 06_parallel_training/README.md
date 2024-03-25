# Parallel Training Methods

[Sam Foreman](https://samforeman.me)  
_2024-03-12_

- **Recording**: 🎬 [Intro to AI Series: Parallel Training Methods for AI](https://www.youtube.com/watch?v=z1t_xmHTJeU)
- Slides: [📊 Parallel Training Slides](https://saforem2.github.io/parallel-training-slides) \[[GitHub](https://github.com/saforem2/parallel-training-slides)\]

## Hands-On

- We will use distributed data parallel training to train a LLM on multiple nodes of Polaris @ ALCF.


#### Wordplay @ ALCF


1. Launch Job:

    ```bash
    $ qsub -A ALCFAITP -q debug -l select=2 -l walltime=01:00:00,filesystems=eagle:home -I
    qsub: waiting for job 1779554.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov to start
    qsub: job 1779554.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov ready
    ```


2. Load conda:

    ```bash
    $ module load conda/2023-10-04 ; conda activate base
    ```

3. Clone [`saforem2/wordplay`](https://github.com/saforem2/wordplay):

    ```bash
    $ git clone https://github.com/saforem2/wordplay
    $ cd wordplay
    ```

4. Make + activate virtual-env:

    ```bash
    $ mkdir -p venvs/polaris/2023-10-04
    $ python3 -m venv venvs/polaris/2023-10-04 --system-site-packages
    $ source venvs/polaris/2023-10-04/bin/activate
    ```

6. Install [`wordplay`](https://github.com/saforem2/wordplay):

    ```bash
    (2023-10-04) $ python3 -m pip install -e "."
    ```

7. Install [`ezpz`](https://github.com/saforem2/ezpz):

    ```bash
    (2023-10-04) $ git clone https://github.com/saforem2/ezpz
    (2023-10-04) $ python3 -m pip install -e "ezpz[dev]"
    (2023-10-04) $ source ezpz/src/ezpz/bin/savejobenv > /tmp/savejobenv.log 2>&1  # || exit
    (2023-10-04) $ source ezpz/src/ezpz/bin/getjobenv  # || exit
    ┌──────────────────────────────────────────────────────────────────
    │ [Hosts]:
    │     • [host:0] - x3006c0s13b1n0.hsn.cm.polaris.alcf.anl.gov
    │     • [host:1] - x3006c0s19b0n0.hsn.cm.polaris.alcf.anl.gov
    └──────────────────────────────────────────────────────────────────
    ┌──────────────────────────────────────────────────────────────────
    │ [DIST INFO]:
    │     • Loading job env from: /home/foremans/.pbsenv
    │     • HOSTFILE: /var/spool/pbs/aux/1779643.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov
    │     • NHOSTS: 2
    │     • NGPU_PER_HOST: 4
    │     • NGPUS (NHOSTS x NGPU_PER_HOST): 8
    │     • WORLD_SIZE: 8
    │     • DIST_LAUNCH: mpiexec --verbose --envall -n 8 -ppn 4 --hostfile /var/spool/pbs/aux/1779643.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov
    └──────────────────────────────────────────────────────────────────
    ┌──────────────────────────────────────────────────────────────────
    │ [Launch]:
    │     • Use: 'launch' (=mpiexec --verbose --envall -n 8 -ppn 4 --hostfile /var/spool/pbs/aux/1779643.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov)
    │       to launch job
    └──────────────────────────────────────────────────────────────────
    ```

8. Prepare data:

    ```bash
    (2023-10-04) $ python3 data/shakespeare_char/prepare.py
    length of dataset in characters: 1,115,394
    all the unique characters:
     !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
    vocab size: 65
    train has 1,003,854 tokens
    val has 111,540 tokens
    ```

9. Launch Training:


    ```bash
    (2023-10-04) $ cd src/wordplay
    (2023-10-04) $ launch python3 __main__.py +experiment=shakespeare data=shakespeare train.backend=DDP train.max_iters=100 train.log_interval=5 train.compile=false
    ```

    <details closed><summary><code>Output:</code></summary>

    ```bash
    Connected to tcp://x3006c0s13b1n0.hsn.cm.polaris.alcf.anl.gov:7919
    Found executable /lus/eagle/projects/datascience/foremans/tmp/wordplay/venvs/polaris/2023-10-04/bin/python3
    Launching application 52482c6c-599c-4d4a-b57b-34f0ac962249
    [2024-03-09 10:38:01][INFO][configs:72] - Setting HF_DATASETS_CACHE to /lus/eagle/projects/datascience/foremans/tmp/wordplay/.cache/huggingface/datasets
    [2024-03-09 10:38:03][INFO][distributed_c10d:442] - Added key: store_based_barrier_key:1 to store for rank: 7
    [2024-03-09 10:38:03][INFO][distributed_c10d:442] - Added key: store_based_barrier_key:1 to store for rank: 4
    [2024-03-09 10:38:03][INFO][distributed_c10d:442] - Added key: store_based_barrier_key:1 to store for rank: 0
    [2024-03-09 10:38:03][INFO][distributed_c10d:442] - Added key: store_based_barrier_key:1 to store for rank: 5
    [2024-03-09 10:38:03][INFO][distributed_c10d:442] - Added key: store_based_barrier_key:1 to store for rank: 6
    [2024-03-09 10:38:03][INFO][distributed_c10d:476] - Rank 0: Completed store-based barrier for key:store_based_barrier_key:1 with 8 nodes.
    [2024-03-09 10:38:03][INFO][distributed_c10d:442] - Added key: store_based_barrier_key:1 to store for rank: 2
    [2024-03-09 10:38:03][INFO][distributed_c10d:442] - Added key: store_based_barrier_key:1 to store for rank: 1
    [2024-03-09 10:38:03][INFO][distributed_c10d:442] - Added key: store_based_barrier_key:1 to store for rank: 3
    [2024-03-09 10:38:03][INFO][distributed_c10d:476] - Rank 7: Completed store-based barrier for key:store_based_barrier_key:1 with 8 nodes.
    [2024-03-09 10:38:03][INFO][distributed_c10d:476] - Rank 5: Completed store-based barrier for key:store_based_barrier_key:1 with 8 nodes.
    [2024-03-09 10:38:03][INFO][distributed_c10d:476] - Rank 2: Completed store-based barrier for key:store_based_barrier_key:1 with 8 nodes.
    [2024-03-09 10:38:03][INFO][distributed_c10d:476] - Rank 6: Completed store-based barrier for key:store_based_barrier_key:1 with 8 nodes.
    [2024-03-09 10:38:03][INFO][distributed_c10d:476] - Rank 3: Completed store-based barrier for key:store_based_barrier_key:1 with 8 nodes.
    [2024-03-09 10:38:03][INFO][distributed_c10d:476] - Rank 4: Completed store-based barrier for key:store_based_barrier_key:1 with 8 nodes.
    [2024-03-09 10:38:03][INFO][distributed_c10d:476] - Rank 1: Completed store-based barrier for key:store_based_barrier_key:1 with 8 nodes.
    [2024-03-09 10:38:03][INFO][dist:290] - [device='cuda'][rank=2/7][local_rank=2/3][node=0/1]
    [2024-03-09 10:38:03][INFO][dist:290] - [device='cuda'][rank=4/7][local_rank=0/3][node=0/1]
    [2024-03-09 10:38:03][INFO][dist:290] - [device='cuda'][rank=3/7][local_rank=3/3][node=1/1]
    [2024-03-09 10:38:03][INFO][dist:290] - [device='cuda'][rank=5/7][local_rank=1/3][node=1/1]
    [2024-03-09 10:38:03][INFO][dist:290] - [device='cuda'][rank=7/7][local_rank=3/3][node=1/1]
    [2024-03-09 10:38:03][INFO][dist:290] - [device='cuda'][rank=1/7][local_rank=1/3][node=1/1]
    [2024-03-09 10:38:03][INFO][dist:290] - [device='cuda'][rank=6/7][local_rank=2/3][node=0/1]
    [2024-03-09 10:38:03][INFO][dist:239] - DistInfo={
        "DEVICE": "cuda",
        "DEVICE_ID": "cuda:0",
        "DISTRIBUTED_BACKEND": "nccl",
        "GPUS_PER_NODE": 4,
        "HOSTFILE": "/var/spool/pbs/aux/1779643.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov",
        "HOSTNAME": "x3006c0s13b1n0.hsn.cm.polaris.alcf.anl.gov",
        "HOSTS": "['x3006c0s13b1n0', 'x3006c0s19b0n0']",
        "LOCAL_RANK": 0,
        "MACHINE": "Polaris",
        "NGPUS": 8,
        "NODE_ID": 0,
        "NUM_NODES": 2,
        "RANK": 0,
        "SCHEDULER": "PBS",
        "WORLD_SIZE_IN_USE": 8,
        "WORLD_SIZE_TOTAL": 8
    }
    [2024-03-09 10:38:03][INFO][dist:605] - [0/8] Using device='cuda' with backend='DDP' + 'nccl' for distributed training.
    [2024-03-09 10:38:03][INFO][dist:290] - [device='cuda'][rank=0/7][local_rank=0/3][node=0/1]
    [2024-03-09 10:38:03][WARNING][dist:296] - Using [8 / 8] available "cuda" devices !!
    [2024-03-09 10:38:03][INFO][configs:308] - Loading val from /lus/eagle/projects/datascience/foremans/tmp/wordplay/data/shakespeare_char/val.bin
    [2024-03-09 10:38:03][INFO][configs:308] - Loading train from /lus/eagle/projects/datascience/foremans/tmp/wordplay/data/shakespeare_char/train.bin
    [2024-03-09 10:38:03][INFO][configs:283] - Rescaling GAS -> GAS // WORLD_SIZE = 8 // 8
    [2024-03-09 10:38:03][INFO][configs:433] - Tokens per iteration: 131,072
    [2024-03-09 10:38:04][INFO][trainer:277] - Initializing a new model from scratch
    [2024-03-09 10:38:04][INFO][trainer:277] - Initializing a new model from scratch
    [2024-03-09 10:38:04][INFO][trainer:277] - Initializing a new model from scratch
    [2024-03-09 10:38:04][INFO][trainer:277] - Initializing a new model from scratch
    [2024-03-09 10:38:04][INFO][trainer:277] - Initializing a new model from scratch
    [2024-03-09 10:38:04][INFO][trainer:277] - Initializing a new model from scratch
    [2024-03-09 10:38:04][INFO][trainer:277] - Initializing a new model from scratch
    [2024-03-09 10:38:04][INFO][configs:456] - Using self.ptdtype=torch.bfloat16 on self.device_type='cuda'
    [2024-03-09 10:38:04][INFO][configs:462] - Initializing a new model from scratch
    [2024-03-09 10:38:04][INFO][dist:751] - Setting up wandb from rank: 0
    [2024-03-09 10:38:04][INFO][dist:752] - Using: WB PROJECT: WordPlay
    [2024-03-09 10:38:04][CRITICAL][trainer:338] - "devid='cuda:2'"
    [2024-03-09 10:38:04][CRITICAL][trainer:338] - "devid='cuda:1'"
    [2024-03-09 10:38:04][CRITICAL][trainer:338] - "devid='cuda:2'"
    [2024-03-09 10:38:04][CRITICAL][trainer:338] - "devid='cuda:3'"
    [2024-03-09 10:38:04][CRITICAL][trainer:338] - "devid='cuda:1'"
    [2024-03-09 10:38:04][CRITICAL][trainer:338] - "devid='cuda:3'"
    [2024-03-09 10:38:04][CRITICAL][trainer:338] - "devid='cuda:0'"
    wandb: Currently logged in as: foremans (aurora_gpt). Use `wandb login --relogin` to force relogin
    wandb: wandb version 0.16.4 is available!  To upgrade, please run:
    wandb:  $ pip install wandb --upgrade
    wandb: Tracking run with wandb version 0.16.2
    wandb: Run data is saved locally in /lus/eagle/projects/datascience/foremans/tmp/wordplay/src/outputs/runs/shakespeare/pytorch/DDP/2024-03-09/10-38-03/wandb/run-20240309_103805-6b22rdws
    wandb: Run `wandb offline` to turn off syncing.
    wandb: Syncing run feasible-sunset-4
    wandb: ⭐️ View project at https://wandb.ai/aurora_gpt/WordPlay
    wandb: 🚀 View run at https://wandb.ai/aurora_gpt/WordPlay/runs/6b22rdws
    [2024-03-09 10:38:06][INFO][dist:782] - W&B RUN: [feasible-sunset-4](https://wandb.ai/aurora_gpt/WordPlay/runs/6b22rdws)
    [2024-03-09 10:38:06][INFO][dist:810] - Running on machine='Polaris'
    [2024-03-09 10:38:06][WARNING][__main__:87] - {
        "train": {
            "framework": "pytorch",
            "backend": "DDP",
            "device": null,
            "seed": null,
            "port": null,
            "ds_config_path": null,
            "precision": null,
            "ngpus": null,
            "use_wandb": true,
            "eval_interval": 250,
            "log_interval": 5,
            "eval_iters": 200,
            "eval_only": false,
            "always_save_checkpoint": false,
            "init_from": "scratch",
            "wandb_project": "WordPlay",
            "max_iters": 100,
            "warmup_iters": 100,
            "dtype": "bfloat16",
            "compile": false
        },
        "model": {
            "n_layer": 6,
            "n_head": 6,
            "n_embd": 384,
            "batch_size": 64,
            "block_size": 256,
            "activation": "gelu",
            "dropout": 0.2,
            "bias": false,
            "vocab_size": 65
        },
        "data": {
            "dataset": "shakespeare_char",
            "out_dir": "out-shakespeare-char",
            "root_path": null
        },
        "optimizer": {
            "learning_rate": 0.001,
            "weight_decay": 0.1,
            "beta1": 0.9,
            "beta2": 0.99,
            "grad_clip": 1.0,
            "decay_lr": true,
            "lr_decay_iters": 5000,
            "min_lr": 0.0001,
            "gradient_accumulation_steps": 1
        }
    }
    [2024-03-09 10:38:06][WARNING][__main__:88] - Output dir: /lus/eagle/projects/datascience/foremans/tmp/wordplay/src/outputs/runs/shakespeare/pytorch/DDP/2024-03-09/10-38-03
    [2024-03-09 10:38:06][INFO][trainer:277] - Initializing a new model from scratch
    [2024-03-09 10:38:06][INFO][model:255] - number of parameters: 10.65M
    [2024-03-09 10:38:06][INFO][model:445] - num decayed parameter tensors: 26, with 10,740,096 parameters
    [2024-03-09 10:38:06][INFO][model:449] - num non-decayed parameter tensors: 13, with 4,992 parameters
    [2024-03-09 10:38:06][INFO][model:465] - using fused AdamW: True
    [2024-03-09 10:38:06][CRITICAL][trainer:338] - "devid='cuda:0'"
    [2024-03-09 10:38:08][INFO][trainer:700] - Startup time: 5.5545
      0%|          | 0/100 [00:00<?, ?it/s][2024-03-09 10:38:09][INFO][trainer:700] - Startup time: 5.5563
    [2024-03-09 10:38:09][INFO][trainer:700] - Startup time: 5.5572
    [2024-03-09 10:38:09][INFO][trainer:700] - Startup time: 6.0207
                                  Training Legend
    ┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    ┃    abbr    ┃ desc                                                        ┃
    ┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
    │    step    │ Current training iteration                                  │
    │    loss    │ Loss value                                                  │
    │     dt     │ Elapsed time per training step (measured in **ms**)         │
    │    dtf     │ Elapsed time per forward step (measured in **ms**)          │
    │    dtb     │ Elapsed time per backward step (measured in **ms**)         │
    │    sps     │ Samples per second                                          │
    │    mtps    │ Tokens per second, measured in MEGA (1 x 10^6) tokens / sec │
    │    mfu     │ Model flops utilization                                     │
    │ train_loss │ Training loss value                                         │
    │  val_loss  │ Validation loss value                                       │
    └────────────┴─────────────────────────────────────────────────────────────┘
    [2024-03-09 10:38:09][INFO][trainer:700] - Startup time: 6.0228
    [2024-03-09 10:38:09][INFO][trainer:700] - Startup time: 6.0243
    [2024-03-09 10:38:09][INFO][trainer:700] - Startup time: 6.0259
    [2024-03-09 10:38:09][INFO][trainer:700] - Startup time: 5.5711
      1%|          | 1/100 [00:03<06:15,  3.79s/it][2024-03-09 10:38:12][INFO][distributed:1140] - Reducer buckets have been rebuilt in this iteration.
    [2024-03-09 10:38:12][INFO][distributed:1140] - Reducer buckets have been rebuilt in this iteration.
    [2024-03-09 10:38:12][INFO][distributed:1140] - Reducer buckets have been rebuilt in this iteration.
    [2024-03-09 10:38:12][INFO][distributed:1140] - Reducer buckets have been rebuilt in this iteration.
    [2024-03-09 10:38:12][INFO][distributed:1140] - Reducer buckets have been rebuilt in this iteration.
    [2024-03-09 10:38:12][INFO][distributed:1140] - Reducer buckets have been rebuilt in this iteration.
    [2024-03-09 10:38:12][INFO][distributed:1140] - Reducer buckets have been rebuilt in this iteration.
    [2024-03-09 10:38:12][INFO][distributed:1140] - Reducer buckets have been rebuilt in this iteration.
      3%|▎         | 3/100 [00:04<01:31,  1.05it/s][2024-03-09 10:38:13][INFO][trainer:768] - step=5 loss=3.6333 dt=84.3107 dtf=4.3360 dtb=78.0501 sps=94.8871 mtps=1.5546 mfu=-100.0000 train_loss=4.2858 val_loss=4.2797
      8%|▊         | 8/100 [00:04<00:23,  3.96it/s][2024-03-09 10:38:13][INFO][trainer:768] - step=10 loss=3.2341 dt=88.2915 dtf=4.5337 dtb=81.7871 sps=90.6089 mtps=1.4845 mfu=4.2204 train_loss=4.2858 val_loss=4.2797
     14%|█▍        | 14/100 [00:04<00:10,  8.37it/s][2024-03-09 10:38:14][INFO][trainer:768] - step=15 loss=2.9544 dt=76.5423 dtf=4.5539 dtb=69.2066 sps=104.5173 mtps=1.7124 mfu=4.2852 train_loss=4.2858 val_loss=4.2797
     18%|█▊        | 18/100 [00:05<00:08,  9.84it/s][2024-03-09 10:38:14][INFO][trainer:768] - step=20 loss=2.7892 dt=110.9556 dtf=4.6069 dtb=103.4925 sps=72.1009 mtps=1.1813 mfu=4.1925 train_loss=4.2858 val_loss=4.2797
     24%|██▍       | 24/100 [00:05<00:06, 12.28it/s][2024-03-09 10:38:14][INFO][trainer:768] - step=25 loss=2.6724 dt=58.4101 dtf=4.5098 dtb=51.2084 sps=136.9626 mtps=2.2440 mfu=4.4112 train_loss=4.2858 val_loss=4.2797
     28%|██▊       | 28/100 [00:06<00:05, 12.77it/s][2024-03-09 10:38:15][INFO][trainer:768] - step=30 loss=2.6189 dt=97.2098 dtf=4.3142 dtb=90.9276 sps=82.2962 mtps=1.3483 mfu=4.3534 train_loss=4.2858 val_loss=4.2797
     34%|███▍      | 34/100 [00:06<00:05, 11.95it/s][2024-03-09 10:38:15][INFO][trainer:768] - step=35 loss=2.5821 dt=98.7089 dtf=4.3972 dtb=91.5855 sps=81.0464 mtps=1.3279 mfu=4.2955 train_loss=4.2858 val_loss=4.2797
     38%|███▊      | 38/100 [00:06<00:05, 12.40it/s][2024-03-09 10:38:16][INFO][trainer:768] - step=40 loss=2.5464 dt=59.2544 dtf=4.4211 dtb=52.8164 sps=135.0110 mtps=2.2120 mfu=4.4948 train_loss=4.2858 val_loss=4.2797
     44%|████▍     | 44/100 [00:07<00:04, 12.77it/s][2024-03-09 10:38:16][INFO][trainer:768] - step=45 loss=2.4945 dt=104.4158 dtf=4.6090 dtb=97.0525 sps=76.6167 mtps=1.2553 mfu=4.4022 train_loss=4.2858 val_loss=4.2797
     48%|████▊     | 48/100 [00:07<00:04, 12.18it/s][2024-03-09 10:38:16][INFO][trainer:768] - step=50 loss=2.4974 dt=111.7754 dtf=4.4986 dtb=105.1647 sps=71.5721 mtps=1.1726 mfu=4.2954 train_loss=4.2858 val_loss=4.2797
     54%|█████▍    | 54/100 [00:08<00:03, 11.94it/s][2024-03-09 10:38:17][INFO][trainer:768] - step=55 loss=2.4952 dt=75.8763 dtf=4.5003 dtb=68.6473 sps=105.4348 mtps=1.7274 mfu=4.3569 train_loss=4.2858 val_loss=4.2797
     58%|█████▊    | 58/100 [00:08<00:03, 11.35it/s][2024-03-09 10:38:17][INFO][trainer:768] - step=60 loss=2.4836 dt=109.1753 dtf=4.4064 dtb=102.7523 sps=73.2766 mtps=1.2006 mfu=4.2625 train_loss=4.2858 val_loss=4.2797
     64%|██████▍   | 64/100 [00:09<00:03,  9.38it/s][2024-03-09 10:38:18][INFO][trainer:768] - step=65 loss=2.4776 dt=118.7971 dtf=4.8723 dtb=111.1970 sps=67.3417 mtps=1.1033 mfu=4.1499 train_loss=4.2858 val_loss=4.2797
     69%|██████▉   | 69/100 [00:09<00:03,  9.22it/s][2024-03-09 10:38:18][INFO][trainer:768] - step=70 loss=2.4589 dt=79.8218 dtf=4.4681 dtb=72.6108 sps=100.2232 mtps=1.6421 mfu=4.2018 train_loss=4.2858 val_loss=4.2797
     73%|███████▎  | 73/100 [00:10<00:02, 10.19it/s][2024-03-09 10:38:19][INFO][trainer:768] - step=75 loss=2.4409 dt=106.6802 dtf=4.2909 dtb=100.3735 sps=74.9905 mtps=1.2286 mfu=4.1309 train_loss=4.2858 val_loss=4.2797
     79%|███████▉  | 79/100 [00:10<00:02, 10.17it/s][2024-03-09 10:38:19][INFO][trainer:768] - step=80 loss=2.4544 dt=114.6547 dtf=4.4237 dtb=107.5242 sps=69.7747 mtps=1.1432 mfu=4.0428 train_loss=4.2858 val_loss=4.2797
     84%|████████▍ | 84/100 [00:11<00:01,  8.56it/s][2024-03-09 10:38:20][INFO][trainer:768] - step=85 loss=2.4350 dt=94.4584 dtf=4.3971 dtb=87.4210 sps=84.6934 mtps=1.3876 mfu=4.0330 train_loss=4.2858 val_loss=4.2797
     89%|████████▉ | 89/100 [00:12<00:01,  9.06it/s][2024-03-09 10:38:21][INFO][trainer:768] - step=90 loss=2.4754 dt=76.8662 dtf=4.5669 dtb=69.5501 sps=104.0770 mtps=1.7052 mfu=4.1145 train_loss=4.2858 val_loss=4.2797
     93%|█████████▎| 93/100 [00:12<00:00, 12.15it/s][2024-03-09 10:38:21][INFO][trainer:768] - step=95 loss=2.4567 dt=79.3622 dtf=4.4828 dtb=72.8414 sps=100.8037 mtps=1.6516 mfu=4.1725 train_loss=4.2858 val_loss=4.2797
     99%|█████████▉| 99/100 [00:12<00:00, 13.33it/s][2024-03-09 10:38:21][INFO][trainer:768] - step=100 loss=2.4527 dt=58.1078 dtf=4.5494 dtb=49.4123 sps=137.6752 mtps=2.2557 mfu=4.3966 train_loss=4.2858 val_loss=4.2797
    100%|██████████| 100/100 [00:12<00:00,  7.81it/s]
    [2024-03-09 10:38:21][INFO][trainer:666] - Saving checkpoint to: /lus/eagle/projects/datascience/foremans/tmp/wordplay/src/outputs/runs/shakespeare/pytorch/DDP/2024-03-09/10-38-03
    [2024-03-09 10:38:21][INFO][trainer:667] - Saving model to: /lus/eagle/projects/datascience/foremans/tmp/wordplay/src/outputs/runs/shakespeare/pytorch/DDP/2024-03-09/10-38-03/model.pth
    [2024-03-09 10:38:21][INFO][configs:132] - Appending /lus/eagle/projects/datascience/foremans/tmp/wordplay/src/outputs/runs/shakespeare/pytorch/DDP/2024-03-09/10-38-03 to /lus/eagle/projects/datascience/foremans/tmp/wordplay/src/ckpts/checkpoints.log
    wandb:
    wandb:
    wandb: Run history:
    wandb:              Loss/iter ▁▁▂▂▂▃▃▄▄▄▅▅▅▆▆▇▇▇██
    wandb:             Loss/lossf █▆▄▃▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁
    wandb:               Loss/mfu ▁███████████████████
    wandb:             Loss/train ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
    wandb:               Loss/val ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
    wandb:          Timing/dt_avg ▄▅▃▇▁▆▆▁▆▇▃▇█▄▇█▅▃▄▁
    wandb:         Timing/dt_iter ▄▄▃▇▁▆▆▁▆▇▃▇█▄▇█▅▃▃▁
    wandb:          Timing/dt_tot ▄▅▃▇▁▆▆▁▆▇▃▇█▄▇█▅▃▄▁
    wandb:         Timing/dtb_avg ▄▅▃▇▁▆▆▁▆▇▃▇█▄▇█▅▃▄▁
    wandb:         Timing/dtb_tot ▄▅▃▇▁▆▆▁▆▇▃▇█▄▇█▅▃▄▁
    wandb:         Timing/dtf_avg ▂▄▄▅▄▁▂▃▅▄▄▂█▃▁▃▂▄▃▄
    wandb:         Timing/dtf_tot ▂▄▄▅▄▁▂▃▅▄▄▂█▃▁▃▂▄▃▄
    wandb:            Timing/iter ▁▁▂▂▂▃▃▄▄▄▅▅▅▆▆▇▇▇██
    wandb: Timing/samples_per_sec ▄▃▅▁█▂▂█▂▁▅▂▁▄▂▁▃▅▄█
    wandb:    Timing/startup_time ▁
    wandb:  Timing/tokens_per_sec ▄▃▅▁█▂▂█▂▁▅▂▁▄▂▁▃▅▄█
    wandb:          Training/iter ▁▁▂▂▂▃▃▄▄▄▅▅▅▆▆▇▇▇██
    wandb:          Training/loss █▆▄▃▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁
    wandb:      Training/loss_tot █▆▄▃▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁
    wandb:            Training/lr ▁▁▂▂▂▃▃▄▄▄▅▅▅▆▆▇▇▇██
    wandb:
    wandb: Run summary:
    wandb:              Loss/iter 100
    wandb:             Loss/lossf 2.45266
    wandb:               Loss/mfu 4.39655
    wandb:             Loss/train 4.28576
    wandb:               Loss/val 4.27968
    wandb:          Timing/dt_avg 0.02698
    wandb:         Timing/dt_iter 0.05811
    wandb:          Timing/dt_tot 0.05396
    wandb:         Timing/dtb_avg 0.04941
    wandb:         Timing/dtb_tot 0.04941
    wandb:         Timing/dtf_avg 0.00455
    wandb:         Timing/dtf_tot 0.00455
    wandb:            Timing/iter 99
    wandb: Timing/samples_per_sec 137.67523
    wandb:    Timing/startup_time 5.55634
    wandb:  Timing/tokens_per_sec 2255670.95332
    wandb:          Training/iter 99
    wandb:          Training/loss 2.45266
    wandb:      Training/loss_tot 2.45266
    wandb:            Training/lr 0.00099
    wandb:
    wandb: 🚀 View run feasible-sunset-4 at: https://wandb.ai/aurora_gpt/WordPlay/runs/6b22rdws
    wandb: ️⚡ View job at https://wandb.ai/aurora_gpt/WordPlay/jobs/QXJ0aWZhY3RDb2xsZWN0aW9uOjE0NzA3MTg5Mw==/version_details/v3
    wandb: Synced 5 W&B file(s), 0 media file(s), 1 artifact file(s) and 1 other file(s)
    wandb: Find logs at: /lus/eagle/projects/datascience/foremans/tmp/wordplay/src/outputs/runs/shakespeare/pytorch/DDP/2024-03-09/10-38-03/wandb/run-20240309_103805-6b22rdws/logs
    Application 52482c6c resources: utime=126s stime=112s maxrss=3541260KB inblock=1704 oublock=506056 minflt=4851205 majflt=0 nvcsw=69557 nivcsw=23617
    ```

    </details>

10. \[**Homework**\] Submit either:
    - Link to W&B run, e.g.

        ```bash
        wandb: 🚀 View run feasible-sunset-4 at: https://wandb.ai/aurora_gpt/WordPlay/runs/6b22rdws
        ```

    - Path to logfile
