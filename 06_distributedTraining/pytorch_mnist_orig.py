from __future__ import print_function
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data.distributed



import time


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
parser.add_argument('--device', default='cpu',
                    help='Wheter this is running on cpu or gpu')
parser.add_argument('--num_threads', default=0, help='set number of threads per worker', type=int)
args = parser.parse_args()

args.cuda = args.device.find("gpu")!=-1
torch.manual_seed(args.seed)
if args.device.find("gpu")!=-1:
    # Horovod: pin GPU to local rank.
    torch.cuda.manual_seed(args.seed)
if (args.num_threads!=0):
    torch.set_num_threads(args.num_threads)


print("Torch Thread setup: ")
print(" Number of threads: ", torch.get_num_threads())
#    print(" Number of inter_op threads: ", torch.get_num_interop_threads())

kwargs = {'num_workers': 1, 'pin_memory': True} if args.device.find("gpu")!=-1 else {}
train_dataset = \
    datasets.MNIST('datasets/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
# Horovod: use DistributedSampler to partition the training data.
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=1, rank=0)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)

test_dataset = \
    datasets.MNIST('datasets', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))
# Horovod: use DistributedSampler to partition the test data.
test_sampler = torch.utils.data.distributed.DistributedSampler(
    test_dataset, num_replicas=1, rank=0)
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

if args.device.find("gpu")!=-1:
    # Move model to GPU.
    model.cuda()

# Horovod: scale learning rate by the number of GPUs.
optimizer = optim.SGD(model.parameters(), lr=args.lr,
                      momentum=args.momentum)





def train(epoch):
    model.train()
    running_loss = 0.0
    training_acc = 0.0
    # Horovod: set epoch to sampler for shuffling.
    train_sampler.set_epoch(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1]
        training_acc += pred.eq(target.data.view_as(pred)).cpu().float().sum()
        running_loss += loss.item()

        if batch_idx % args.log_interval == 0:
            # Horovod: use train_sampler to determine the number of examples in
            # this worker's partition.
            print('[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(0, 
                epoch, batch_idx * len(data), len(train_sampler), 100. * batch_idx / len(train_loader), loss.item()/args.batch_size))
    running_loss = running_loss / len(train_sampler)
    training_acc = training_acc / len(train_sampler)
    loss_avg = metric_average(running_loss, 'running_loss')
    training_acc = metric_average(training_acc, 'training_acc')



def metric_average(val, name):
    tensor = torch.tensor(val)
    return avg_tensor.item()


def test():
    model.eval()
    test_loss = 0.
    test_accuracy = 0.
    n = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        #test_loss += F.nll_loss(output, target, size_average=False).item()
        test_loss += F.nll_loss(output, target).item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        test_accuracy += pred.eq(target.data.view_as(pred)).cpu().float().sum()
        n=n+1

    # Horovod: use test_sampler to determine the number of examples in
    # this worker's partition.
    test_loss /= len(test_sampler)
    test_accuracy /= len(test_sampler)

    # Horovod: average metric values across workers.
    test_loss = metric_average(test_loss, 'avg_loss')
    test_accuracy = metric_average(test_accuracy, 'avg_accuracy')

    # Horovod: print output only on first rank.
    print('Test set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
        test_loss, 100. * test_accuracy))

t0 = time.time()
for epoch in range(1, args.epochs + 1):
    train(epoch)
    test()
t1 = time.time()
print("Total training time: %s seconds" %(t1 - t0))
