# Cerebras CS-3

The Cerebras CS-3 is a wafer-scale deep learning accelerator comprising 900K processing cores, each providing 48KB of dedicated SRAM memory, with a total of 44GB on-chip memory. Its software platform integrates the popular machine learning framework PyTorch.

The ALCF CS-3 Cerebras Wafer-Scale Cluster, is designed to support large-scale models (up to approximatly 200 billion parameters) and large-scale inputs. The cluster contains four CS-3 wafer scale engines (compute nodes), supported by 4 worker nodes, 4 activation servers, 2 sets of 12 MemoryX, and 8 SwarmX nodes.

The Cerebras Wafer-Scale cluster is run as an appliance: a user submits a job to the appliance, and the appliance manages preprocessing and streaming of the data, IO, and device orchestration within the appliance. It provides programming via PyTorch. This installation supports Weight Streaming execution for models being pre-trained or fine-tuned.

![CS-3 connection diagram](./Cerebras_Wafer-Scale_Cluster_login_diagram.png)

## Connecting to CS-3
Users connect via SSH to the login node, cerebras.alcf.anl.gov and then ssh to a user node, using either cer-usn-01 or cer-usn-02.

The trees /home, /projects, and /software are shared across the login nodes and user nodes, the relevant cluster infrastructure nodes, and all ALCF AI testbed platforms.

To connect to a CS-3 login:

ssh to the login node:

```bash
ssh ALCFUserID@cerebras.alcf.anl.gov
```

Then ssh to a cerebras user node:

```bash
ssh cer-usn-01
or
ssh cer-usn-02
```


## Prerequisite: Create Virtual Environment 

### PyTorch virtual environment

```bash
mkdir ~/R_2.6.0
cd ~/R_2.6.0
# Note: "deactivate" does not actually work in scripts.
deactivate
rm -r venv_cerebras_pt
/usr/bin/python3.11 -m venv venv_cerebras_pt
source venv_cerebras_pt/bin/activate
pip install --upgrade pip
```

## Clone Cerebras modelzoo

We use an example from [Cerebras Modelzoo repository](https://github.com/Cerebras/modelzoo) for this hands-on. 
```bash
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
git clone https://github.com/Cerebras/modelzoo.git
cd modelzoo
git tag
git checkout Release_2.6.0
```

## Job Queuing and Submission

The CS-3 cluster has its own Kubernetes-based system for job submission and queuing. Jobs are started automatically through the Python scripts. 

Use Cerebras cluster command line tool to get addional information about the jobs.

* Jobs that have not yet completed can be listed as
    `(venv_pt) $ csctl get jobs`
* Jobs can be canceled as shown:
    `(venv_tf) $ csctl cancel job wsjob-eyjapwgnycahq9tus4w7id`

See `csctl -h` for more options.

## Hands-on Example
* [LLAMA2-7B](./llama2-7b.md)

