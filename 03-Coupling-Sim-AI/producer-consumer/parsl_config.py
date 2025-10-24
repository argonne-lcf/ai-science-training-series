import os
import parsl
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider
from parsl.launchers import MpiExecLauncher
from parsl.config import Config

# Get the number of nodes from the PBS_NODEFILE
pbs_node_file = os.getenv('PBS_NODEFILE')
num_nodes = 1
with open(pbs_node_file, 'r') as nf:
    nodes = nf.readlines()
    num_nodes = len(nodes)

# For this cpu only config, bind one worker per cpu core
cpu_affinity = "list"
num_cpu_sockets = 1
num_cores_per_socket = 32
for cpu in range(num_cpu_sockets):
    core_start = cpu*32
    for core in range(num_cores_per_socket):
        cpu_affinity += f":{core_start+core+1}"

# Parsl config for cpu workers
polaris_cpu_config = Config(
    initialize_logging=True, # Set to False for runs more than 1000 nodes
    executors=[
        HighThroughputExecutor(
            max_workers_per_node=32, # We will use 32 workers, one for each CPU core
            cpu_affinity=cpu_affinity,  # Prevents thread contention
            prefetch_capacity=0,  # Increase if you have many more tasks than workers
            provider=LocalProvider(
                launcher=MpiExecLauncher(
                    bind_cmd="--cpu-bind", overrides="--ppn 1"
                ),  # Ensures 1 manger per node and allows it to divide work to all 64 cores
                nodes_per_block=num_nodes,
                init_blocks=1,
                max_blocks=1,
                worker_init="cd runinfo", # Including this will make helper files write to runinfo
            ),
        ),
    ]
)

# Parsl config for GPU workers
polaris_gpu_config = Config(
    initialize_logging=True, # Set to False for runs more than 1000 nodes
    executors=[
        HighThroughputExecutor(
            available_accelerators=4, # 4 GPUs per node
            max_workers_per_node=4, # We will use 4 workers, one for each GPU
            # This gives optimal binding of threads to GPUs on a Polaris node
            cpu_affinity="list:24-31,56-63:16-23,48-55:8-15,40-47:0-7,32-39",
            prefetch_capacity=0,  # Increase if you have many more tasks than workers
            provider=LocalProvider(
                launcher=MpiExecLauncher(
                    bind_cmd="--cpu-bind", overrides="--ppn 1"
                ),  # Ensures 1 manger per node and allows it to divide work to all 64 cores
                nodes_per_block=num_nodes,
                init_blocks=1,
                max_blocks=1,
                worker_init="cd runinfo", # Including this will make helper files write to runinfo
            ),
        ),
    ]
)
