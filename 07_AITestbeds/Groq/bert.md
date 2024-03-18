# BERT On Groq



#### Get a Groq node interactively

```bash
qsub -I -l walltime=1:00:00
```

#### Go to directory with Resnet50 example. 
```bash
cd ~groqflow/proof_points/natural_language_processing/bert
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
python bert_tiny.py
```
<details>
  <summary>Sample Output</summary>

  ```bash
  $ python bert_tiny.py 
Downloading tokenizer_config.json: 100%|████████████████████████████████████████████████████████████| 346/346 [00:00<00:00, 3.90MB/s]
Downloading vocab.txt: 100%|██████████████████████████████████████████████████████████████████████| 232k/232k [00:00<00:00, 11.9MB/s]
Downloading (…)cial_tokens_map.json: 100%|███████████████████████████████████████████████████████████| 112/112 [00:00<00:00, 650kB/s]
Downloading config.json: 100%|██████████████████████████████████████████████████████████████████████| 760/760 [00:00<00:00, 6.15MB/s]
Downloading pytorch_model.bin: 100%|████████████████████████████████████████████████████████████| 17.6M/17.6M [00:00<00:00, 98.3MB/s]



Building "bert_tiny"
    ✓ Exporting PyTorch to ONNX   
    ✓ Optimizing ONNX file   
    ✓ Checking for Op support   
    ✓ Converting to FP16   
    ✓ Compiling model   
    ✓ Assembling model   

Woohoo! Saved to ~/.cache/groqflow/bert_tiny
Preprocessing data.
Downloading builder script: 100%|███████████████████████████████████████████████████████████████| 9.13k/9.13k [00:00<00:00, 37.4MB/s]
Downloading readme: 100%|███████████████████████████████████████████████████████████████████████| 6.68k/6.68k [00:00<00:00, 51.1MB/s]
Downloading data: 100%|█████████████████████████████████████████████████████████████████████████| 6.37M/6.37M [00:01<00:00, 5.04MB/s]
Downloading data: 100%|███████████████████████████████████████████████████████████████████████████| 790k/790k [00:00<00:00, 1.47MB/s]
Generating train split: 100%|██████████████████████████████████████████████████████████| 8544/8544 [00:00<00:00, 10927.17 examples/s]
Generating validation split: 100%|██████████████████████████████████████████████████████| 1101/1101 [00:00<00:00, 2031.92 examples/s]
Generating test split: 100%|████████████████████████████████████████████████████████████| 2210/2210 [00:00<00:00, 3774.12 examples/s]

Info: No inputs received for benchmark. Using the inputs provided during model compilation.
/projects/datascience/sraskar/groq/groqflow/groqflow/groqmodel/execute.py:87: DeprecationWarning: `product` is deprecated as of NumPy 1.25.0, and will be removed in NumPy 2.0. Please use `prod` instead.
  return tsp_runner(**example)
Running inference on GroqChip.
/projects/datascience/sraskar/groq/groqflow/groqflow/groqmodel/execute.py:87: DeprecationWarning: `product` is deprecated as of NumPy 1.25.0, and will be removed in NumPy 2.0. Please use `prod` instead.
  return tsp_runner(**example)
Running inference using PyTorch model (CPU).
100%|███████████████████████████████████████████████████████████████████████████████████████████| 2210/2210 [00:05<00:00, 436.96it/s]
+--------+----------+-------------------------+----------------+----------------------+-------------+
| Source | Accuracy | end-to-end latency (ms) | end-to-end IPS | on-chip latency (ms) | on-chip IPS |
+--------+----------+-------------------------+----------------+----------------------+-------------+
|  cpu   |  77.47%  |           2.29          |     436.88     |          --          |      --     |
|  groq  |  77.47%  |           0.06          |    17147.76    |         0.03         |   32358.97  |
+--------+----------+-------------------------+----------------+----------------------+-------------+
    ```
</details>