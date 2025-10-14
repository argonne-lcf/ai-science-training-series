# Distributed training with PyTorch

## Why distributed training:
Training the best AI models today require a large amount of compute capacity and
memory. Trillion parameter models are available for consumption, and we may try 
to do a back of the envelope calculation to estimate the requirements! 

- Store the parameters 
    - Total number of parameters $10^{12}$ $\times$ Memory per parameters
    - With FP16 -- each parameter requires 2 Bytes
    - $2 \times 10^{12}$ = $2$ TB

And we need these parameters on the GPU to perform each step of the training 
loop! One of the flagship GPUs today have memory of ~140 GB ([Nvidia B200](https://www.nvidia.com/en-us/data-center/dgx-b200/))
Clearly, if we want to train a trillion parameter model, we will require more 
than 1 GPU. If we target one of the largest publicly available models, [llama-405b](https://huggingface.co/meta-llama/Llama-3.1-405B)
we would still require $810$ GB of memory -- just for the parameters! Hence 
distributed training is a must!

Training a large model requires, more than just the parameters. It requires 
training data and optimizers to calculate losses which increases the memory 
requirements. Today, we will talk about a scenario, where our model is small 
enough to fit on a GPU but our dataset is large enough that we will need more 
than 1 GPU. This is commonly known as the Data Parallel training.

_Note_: To quickly estimate the resource requirements of a model training, the
shortcut could be: Trillion($10^{12}$)  parameter models ~ TB, Billion 
($10^{9}$) parameter models ~ GB. If FP16, then $2$ Bytes per parameters, if
FP8, then $1$ Byte per parameter -- each lower precision, reducing by a factor 
of two! Also, gradients will take same memory as parameters, optimizers take 
double the memory. The activations are a bit complicated, but if we consider 
them similar to the optimizers, then our lower estimate by summing all of them
up should be fairly reliable!

## A single GPU program with a random dataset:

Here we are presenting a simple example, where we use the base transformer 
model in its default configuration, with a synthetic dataset to demonstrate 
how to implement a training loop.

```
import torch
+ device = torch.device('cuda')

torch.manual_seed(0)

src = torch.rand((2048, 1, 512))
tgt = torch.rand((2048, 20, 512))
dataset = torch.utils.data.TensorDataset(src, tgt)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

model = torch.nn.Transformer(batch_first=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()
model.train()
+ model = model.to(device)

for epoch in range(10):
    for source, targets in loader:
+         source = source.to(device)
+         targets = targets.to(device)
        optimizer.zero_grad()

        output = model(source, targets)
        loss = criterion(output, targets)

        loss.backward()
        optimizer.step()
```
