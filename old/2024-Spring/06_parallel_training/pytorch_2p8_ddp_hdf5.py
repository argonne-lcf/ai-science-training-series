from mpi4py import MPI
import os, socket, time
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

import h5py

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

# DDP: initialize distributed communication with nccl backend
torch.distributed.init_process_group(backend='nccl', init_method='env://', rank=int(RANK), world_size=int(SIZE))

# DDP: pin GPU to local rank.
torch.cuda.set_device(int(LOCAL_RANK))
device = torch.device('cuda')
torch.manual_seed(0)

src = torch.rand((2048, 1, 512))
tgt = torch.rand((2048, 20, 512))

if RANK == 0:
# Create an HDF5 file and save the tensors
    with h5py.File("tensor_dataset.h5", "w") as hdf5_file:
        # Save the `src` tensor
        hdf5_file.create_dataset("src", data=src.numpy())
        # Save the `tgt` tensor
        hdf5_file.create_dataset("tgt", data=tgt.numpy())

    print("HDF5 dataset created successfully!")

torch.distributed.barrier(device_ids=[torch.cuda.current_device()])

# Custom Dataset to load data from HDF5
class HDF5TensorDataset(torch.utils.data.Dataset):
    def __init__(self, hdf5_file_path):
        # Open the HDF5 file
        self.hdf5_file_path = hdf5_file_path ## Prevents OS error file not found
        self.hdf5_file = h5py.File(self.hdf5_file_path, "r")
        self.src = self.hdf5_file["src"]
        self.tgt = self.hdf5_file["tgt"]

    def __len__(self):
        # Return the number of samples
        return len(self.src)

    def __getitem__(self, idx):
        # Get the src and tgt tensors for the given index
        src_tensor = torch.tensor(self.src[idx], dtype=torch.float32)
        tgt_tensor = torch.tensor(self.tgt[idx], dtype=torch.float32)
        return src_tensor, tgt_tensor

    def close(self):
        # Close the HDF5 file
        self.hdf5_file.close()
# Load the dataset
dataset = HDF5TensorDataset("tensor_dataset.h5")

#dataset = torch.utils.data.TensorDataset(src, tgt)
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
