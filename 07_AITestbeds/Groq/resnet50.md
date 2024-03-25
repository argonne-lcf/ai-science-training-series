# ResNet50 On Groq

ResNet50 is a Convolutional Neural Network (CNN) model used for image classification. Kaiming He, et al. first introduced ResNet models and the revolutionary residual connection (also known as skip connection) in their 2015 paper, Deep Residual Learning for Image Recognition.

#### Get a Groq node interactively

```bash
qsub -I -l walltime=1:00:00
```

#### Go to directory with Resnet50 example. 
```bash
cd ~/proof_points/computer_vision/resnet50
```

#### Activate groqflow virtual Environment 
```bash
conda activate groqflow
```

#### Install Requirements

Install the python dependencies using the requirements.txt file included with this proof point using the following command:
```bash
pip install -r requirements.txt
```

#### Run Training Job

```bash
python resnet50.py
```
<details>
  <summary>Sample Output</summary>

  ```bash
  $ python resnet50.py 
    Downloading: "https://github.com/pytorch/vision/zipball/v0.10.0" to /home/sraskar/.cache/torch/hub/v0.10.0.zip
    Downloading: "https://download.pytorch.org/models/resnet50-0676ba61.pth" to /home/sraskar/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth
    100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 97.8M/97.8M [00:00<00:00, 254MB/s]



    Building "resnet50"
        ✓ Exporting PyTorch to ONNX   
        ✓ Optimizing ONNX file   
        ✓ Checking for Op support   
        ✓ Converting to FP16   
        ✓ Compiling model   
        ✓ Assembling model   

    Woohoo! Saved to ~/.cache/groqflow/resnet50
    Preprocessing data.
    Downloading: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 342M/342M [00:10<00:00, 31.1MB/s]
    100% [..........................................................................] 2568145 / 2568145
    Info: No inputs received for benchmark. Using the inputs provided during model compilation.
    /projects/datascience/sraskar/groq/groqflow/groqflow/groqmodel/execute.py:87: DeprecationWarning: `product` is deprecated as of NumPy 1.25.0, and will be removed in NumPy 2.0. Please use `prod` instead.
    return tsp_runner(**example)
    Running inference on GroqChip.
    /projects/datascience/sraskar/groq/groqflow/groqflow/groqmodel/execute.py:87: DeprecationWarning: `product` is deprecated as of NumPy 1.25.0, and will be removed in NumPy 2.0. Please use `prod` instead.
    return tsp_runner(**example)
    Running inference using PyTorch model (CPU).
    100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3925/3925 [06:43<00:00,  9.73it/s]
    +--------+----------+-------------------------+----------------+----------------------+-------------+
    | Source | Accuracy | end-to-end latency (ms) | end-to-end IPS | on-chip latency (ms) | on-chip IPS |
    +--------+----------+-------------------------+----------------+----------------------+-------------+
    |  cpu   |  84.54%  |          102.76         |      9.73      |          --          |      --     |
    |  groq  |  84.51%  |           0.40          |    2515.15     |         0.33         |   2985.40   |
    +--------+----------+-------------------------+----------------+----------------------+-------------+
    ```
</details>