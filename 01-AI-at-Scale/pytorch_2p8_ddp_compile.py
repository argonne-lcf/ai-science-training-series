from mpi4py import MPI
import os, socket, time
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

# DDP: Set environmental variables used by PyTorch
SIZE = MPI.COMM_WORLD.Get_size()
RANK = MPI.COMM_WORLD.Get_rank()
LOCAL_RANK = os.environ.get('PALS_LOCAL_RANKID')
os.environ['RANK'] = str(RANK)
os.environ['WORLD_SIZE'] = str(SIZE)
MASTER_ADDR = socket.gethostname() if RANK == 0 else None
MASTER_ADDR = MPI.COMM_WORLD.bcast(MASTER_ADDR, root=0)
os.environ['MASTER_ADDR'] = f"{MASTER_ADDR}.hsn.cm.polaris.alcf.anl.gov"
os.environ['MASTER_PORT'] = str(2345)
print(f"DDP: Hi from rank {RANK} of {SIZE} with local rank {LOCAL_RANK}.{MASTER_ADDR}")

# DDP: initialize distributed communication with nccl backend for pytorch 2.8
torch.distributed.init_process_group(backend='nccl', init_method='env://', rank=int(RANK), world_size=int(SIZE))


# DDP: pin GPU to local rank.
torch.cuda.set_device(int(LOCAL_RANK))
device = torch.device('cuda')
torch.manual_seed(0)

src = torch.rand((2048, 1, 512))
tgt = torch.rand((2048, 20, 512))
dataset = torch.utils.data.TensorDataset(src, tgt)
# DDP: use DistributedSampler to partition the training data
sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True, num_replicas=SIZE, rank=RANK, seed=0)
loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=32)

model = torch.nn.Transformer(batch_first=True)
# DDP: scale learning rate by the number of GPUs.
optimizer = torch.optim.Adam(model.parameters(), lr=(0.001*SIZE))
criterion = torch.nn.CrossEntropyLoss()
model.train()
model = model.to(device)
criterion = criterion.to(device)
# DDP: wrap the model in DDP
model = DDP(model)

model = torch.compile(model)

start_t = time.time()
for epoch in range(10):
    #if RANK == 0:
        #print(epoch)
    # DDP: set epoch to sampler for shuffling
    sampler.set_epoch(epoch)

    for source, targets in loader:
        source = source.to(device)
        if RANK == 0:
            print(f"Micro-batch size = {source.shape[0]}")
        targets = targets.to(device)
        optimizer.zero_grad()

        output = model(source, targets)
        loss = criterion(output, targets)

        loss.backward()
        optimizer.step()

if RANK == 0:
    print(f'total train time: {time.time() - start_t:.2f}s', flush=True)

# DDP: cleanup
torch.distributed.destroy_process_group()
