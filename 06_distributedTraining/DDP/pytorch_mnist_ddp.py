from __future__ import print_function
import os
import argparse
import time
import socket

import numpy

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data.distributed

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Set global variables for rank, local_rank, world size
try:
    from mpi4py import MPI

    with_ddp=True
    local_rank = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()


    # Pytorch will look for these:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(size)
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)

    # It will want the master address too, which we'll broadcast:
    if rank == 0:
        master_addr = socket.gethostname()
    else:
        master_addr = None

    master_addr = MPI.COMM_WORLD.bcast(master_addr, root=0)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(2345)


except Exception as e:
    with_ddp=False
    local_rank = 0
    size = 1
    rank = 0
    print("MPI initialization failed!")
    print(e)



t0 = time.time()
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--device', default='cpu', choices=['cpu', 'gpu'],
                    help='Whether this is running on cpu or gpu')
parser.add_argument('--num_threads', default=0, help='set number of threads per worker', type=int)
args = parser.parse_args()

args.cuda = torch.cuda.is_available()

# DDP: initialize library.
'''Initialize distributed communication'''

# What backend?  nccl on GPU, gloo on CPU
if args.device == "gpu": backend = 'nccl'
elif args.device == "cpu": backend = 'gloo'

if with_ddp:
    torch.distributed.init_process_group(
        backend=backend, init_method='env://')


torch.manual_seed(args.seed)

if args.device == 'gpu':
    # DDP: pin GPU to local rank.
    torch.cuda.set_device(int(local_rank))
    torch.cuda.manual_seed(args.seed)

if (args.num_threads!=0):
    torch.set_num_threads(args.num_threads)

if rank==0:
    print("Torch Thread setup: ")
    print(" Number of threads: ", torch.get_num_threads())


kwargs = {'num_workers': 1, 'pin_memory': True} if args.device == 'gpu' else {}
train_dataset = \
    datasets.MNIST('datasets/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=size, rank=rank)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)

test_dataset = \
    datasets.MNIST('datasets', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))

test_sampler = torch.utils.data.distributed.DistributedSampler(
    test_dataset, num_replicas=size, rank=rank)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                          sampler=test_sampler, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


model = Net()


if args.device == 'gpu':
    # Move model to GPU.
    model.cuda()

if with_ddp:
    # wrap the model in DDP:
    model = DDP(model)


optimizer = optim.SGD(model.parameters(), lr=args.lr * size,
                      momentum=args.momentum)




def train(epoch):
    model.train()
    running_loss = torch.tensor(0.0)
    training_acc = torch.tensor(0.0)
    if args.device == "gpu":
        running_loss = running_loss.cuda()
        training_acc = training_acc.cuda()
    train_sampler.set_epoch(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1]
        training_acc += pred.eq(target.data.view_as(pred)).float().sum()
        running_loss += loss

        if batch_idx % args.log_interval == 0 and rank == 0 :
            # Horovod: use train_sampler to determine the number of examples in
            # this worker's partition.
            if rank == 0: print('[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(rank,
                epoch, batch_idx * len(data), len(train_sampler), 100. * batch_idx / len(train_loader), loss.item()/args.batch_size))
    running_loss /= len(train_sampler)
    training_acc /= len(train_sampler)
    loss_avg = metric_average(running_loss, 'running_loss')
    training_acc = metric_average(training_acc, 'training_acc')

    if rank==0: print("Training set: Average loss: {:.4f}, Accuracy: {:.2f}%".format(loss_avg, training_acc*100))


def metric_average(val, name):
    if (with_ddp):
        # Sum everything and divide by total size:
        dist.all_reduce(val,op=dist.reduce_op.SUM)
        val /= size
    else:
        pass
    return val


def test():
    model.eval()
    test_loss = torch.tensor(0.0)
    test_accuracy = torch.tensor(0.0)
    if args.device == "gpu":
        test_loss = test_loss.cuda()
        test_accuracy = test_accuracy.cuda()
    n = 0
    for data, target in test_loader:
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target).item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        test_accuracy += pred.eq(target.data.view_as(pred)).float().sum()
        n=n+1

    # Horovod: use test_sampler to determine the number of examples in
    # this worker's partition.
    test_loss /= len(test_sampler)
    test_accuracy /= len(test_sampler)

    # Horovod: average metric values across workers.
    test_loss = metric_average(test_loss, 'avg_loss')
    test_accuracy = metric_average(test_accuracy, 'avg_accuracy')

    # Horovod: print output only on first rank.
    if rank == 0:
        print('Test set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
            test_loss, 100. * test_accuracy))


epoch_times = []
for epoch in range(1, args.epochs + 1):
    e_start = time.time()
    train(epoch)
    test()
    e_end = time.time()
    epoch_times.append(e_end - e_start)
t1 = time.time()
if rank==0:
    print("Total training time: %s seconds" %(t1 - t0))
    print("Average time per epoch in the last 5: ", numpy.mean(epoch_times[-5:]))
