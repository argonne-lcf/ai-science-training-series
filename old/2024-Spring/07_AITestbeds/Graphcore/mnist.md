# MNIST on Graphcore

Go to directory with mnist example
```bash
cd ~/graphcore/examples/tutorials/simple_applications/pytorch/mnist
```

Activate PopTorch Environment and install requirements 
```bash
source ~/venvs/graphcore/poptorch33_env/bin/activate
python -m pip install -r requirements.txt
```

Submit Job

```bash
/opt/slurm/bin/srun --ipus=1 python mnist_poptorch.py
```
<details>
  <summary>Sample Output</summary>
  
  ```bash
    srun: job 10671 queued and waiting for resources
    srun: job 10671 has been allocated resources
    TrainingModelWithLoss(
    (model): Network(
        (layer1): Block(
        (conv): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
        (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (relu): ReLU()
        )
        (layer2): Block(
        (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
        (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (relu): ReLU()
        )
        (layer3): Linear(in_features=1600, out_features=128, bias=True)
        (layer3_act): ReLU()
        (layer3_dropout): Dropout(p=0.5, inplace=False)
        (layer4): Linear(in_features=128, out_features=10, bias=True)
        (softmax): Softmax(dim=1)
    )
    (loss): CrossEntropyLoss()
    )
    Epochs:   0%|          | 0/10 [00:00<?,[23:27:06.753] [poptorch:cpp] [warning] [DISPATCHER] Type coerced from Long to Int for tensor id 10
    Graph compilation: 100%|██████████| 100/100 [00:00<00:00]
    Epochs: 100%|██████████| 10/10 [01:17<00:00,  7.71s/it]
    Graph compilation: 100%|██████████| 100/100 [00:00<00:00]                          
    Accuracy on test set: 96.85%██████| 100/100 [00:00<00:00]   
  ```
</details>
