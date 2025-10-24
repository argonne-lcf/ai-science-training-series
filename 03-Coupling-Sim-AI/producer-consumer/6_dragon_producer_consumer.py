import dragon
import multiprocessing as mp

from dragon.data.ddict import DDict
from dragon.native.machine import System, Node
from dragon.native.pool import Pool as DragonPool

import numpy as np
from math import pi as PI
from time import sleep, perf_counter
import torch
import argparse

from model import SimpleCNN


def simulation(period, grid_size):
    """Toy simulaiton function to advance the circular wave equation in 2D
    and produce training data for an autoregressive CNN model (u(x,t+1) -> CNN(u(x,t)))
    Args:
        period: period of wave
        grid_size: size of each dimension of the grid (2D grid is a square of size grid_size x grid_size)
    """
    # Attach to the DDict
    dd = mp.current_process().stash["ddict"]

    # Setup grid
    n_samples = grid_size**2
    x = np.linspace(0,1,num=grid_size)*4*PI-2*PI
    y = np.linspace(0,1,num=grid_size)*4*PI-2*PI
    x, y = np.meshgrid(x, y)
    r = np.sqrt(x**2+y**2)

    # Loop over steps to generate training data
    freq = 2*PI/period
    inputs = []
    outputs = []
    for step in range(10):
        sleep(0.5) # sleep to emulate the compute time
        u = np.sin(2.0*r-freq*step)/(r+1.0)
        udt = np.sin(2.0*r-freq*(step+1))/(r+1.0)
        inputs.append(u)
        outputs.append(udt)

    # Write data to DDict (use period in key to make it unique for each simulation)
    tic = perf_counter()
    dd[f"inputs_{int(period)}"] = inputs
    dd[f"outputs_{int(period)}"] = outputs
    toc = perf_counter()
    return toc - tic


def trainer(kernel_size: int = 3):
    """Train the autoregressive CNN model the simulation data
    Args:
        kernel_size: kernel size for the convolutional layers
    """
    # Attach to the DDict
    dd = mp.current_process().stash["ddict"]

    # Initialize the model, loss function, and optimizer
    model = SimpleCNN(kernel_size=kernel_size)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Get the training data from the DDict and create a DataLoader
    inputs = []
    outputs = []
    io_time = 0.0
    for key in dd.keys(): # not ideal at large scale, but okay for this example
        if key.startswith("inputs"):
            tic = perf_counter()
            arrays = dd[key]
            io_time += perf_counter() - tic
            inputs.append(torch.from_numpy(np.array(arrays)).unsqueeze(1).float())
        if key.startswith("outputs"):
            tic = perf_counter()
            arrays = dd[key]
            io_time += perf_counter() - tic
            outputs.append(torch.from_numpy(np.array(arrays)).unsqueeze(1).float())
    inputs = torch.cat(inputs, dim=0)
    outputs = torch.cat(outputs, dim=0)
    dataset = torch.utils.data.TensorDataset(inputs, outputs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    # Train the model
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(1):
        for i, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()

    tic = perf_counter()
    dd[f"model_{kernel_size}"] = model.to("cpu").state_dict()
    io_time += perf_counter() - tic
    return io_time


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_sims", type=int, default=4, help="Number of simulations to run")
    parser.add_argument("--grid_size", type=int, default=128, help="Size of the grid for each simulation")
    args = parser.parse_args()

    # Set the mp start method
    mp.set_start_method("dragon")

    # Get allocation info
    alloc = System()
    num_nodes = alloc.nnodes
    nodelist = alloc.nodes
    head_node = Node(nodelist[0])
    num_cores_per_node = head_node.num_cpus
    num_gpus_per_node = head_node.num_gpus
    print(f"Dragon running on {num_nodes} nodes")
    print([Node(node).hostname for node in nodelist],"\n",flush=True)

    # Initialize the DDict on all the nodes
    ddict_mem_per_node = 0.5 * head_node.physical_mem # dedicate 50% of each node's memory to the DDict
    tot_ddict_mem = int(ddict_mem_per_node * num_nodes)
    managers_per_node = 2
    dd = DDict(managers_per_node, num_nodes, tot_ddict_mem)
    print(f"Started DDict on {num_nodes} nodes with {tot_ddict_mem/1024/1024/1024:.1f}GB of memory\n",flush=True)

    # Run an ensemble of simulations (producer)
    def setup(dd: DDict):
        me = mp.current_process()
        me.stash = {}
        me.stash["ddict"] = dd

    num_workers = min(num_cores_per_node * num_nodes, args.num_sims)
    sim_args = [(period, args.grid_size) for period in np.linspace(40,80,args.num_sims)]
    print(f"Launching {args.num_sims} simulations on {num_workers} workers to generate training data ...")
    tic = perf_counter()
    pool = DragonPool(num_workers, initializer=setup, initargs=(dd,))
    results = pool.map_async(lambda args: simulation(*args), sim_args).get()
    io_time = sum(results)/len(results)
    pool.close()
    pool.join()
    print(f"Done in {perf_counter() - tic:.2f} seconds")
    print(f"IO time: {io_time:.3f} seconds\n",flush=True)

    # Run an ensemble of CNN models (consumer)
    num_gpus = num_gpus_per_node * num_nodes
    ml_args = [int(kernel_size) for kernel_size in range(3,2*num_gpus+3,2)]
    print(f"Launching {num_gpus} CNN models to train on the simulaiton data ...")
    tic = perf_counter()
    pool = DragonPool(policy=System().gpu_policies(), # launches one process per GPU, binding each process to a single GPU
                      processes_per_policy=1, 
                      initializer=setup, 
                      initargs=(dd,)
    )
    results = pool.map_async(trainer, ml_args).get()
    io_time = sum(results)/len(results)
    pool.close()
    pool.join()
    print(f"Done in {perf_counter() - tic:.2f} seconds")
    print(f"IO time: {io_time:.3f} seconds\n",flush=True)

    # Clean up
    dd.destroy()