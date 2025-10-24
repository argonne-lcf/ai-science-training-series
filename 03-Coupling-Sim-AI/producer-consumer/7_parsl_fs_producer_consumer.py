import os
import numpy as np
from time import perf_counter
import argparse
import shutil

import parsl
from parsl.app.app import python_app
from parsl_config import polaris_cpu_config, polaris_gpu_config
from concurrent.futures import as_completed

@python_app
def simulation(period, grid_size):
    """Toy simulaiton function to advance the circular wave equation in 2D
    and produce training data for an autoregressive CNN model (u(x,t+1) -> CNN(u(x,t)))
    Args:
        period: period of wave
        grid_size: size of each dimension of the grid (2D grid is a square of size grid_size x grid_size)
    """
    # Import packages
    import numpy as np
    from math import pi as PI
    from time import sleep, perf_counter

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

    # Write data to the filesystem
    tic = perf_counter()
    np.save(f"./data/inputs_{int(period)}.npy", inputs)
    np.save(f"./data/outputs_{int(period)}.npy", outputs)
    toc = perf_counter()
    return toc - tic


@python_app
def trainer(kernel_size: int = 3):
    """Train the autoregressive CNN model on the simulation data
    Args:
        kernel_size: kernel size for the convolutional layers
    """
    # Import packages
    import os
    import sys
    import numpy as np
    from time import perf_counter
    import torch
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
    from model import SimpleCNN

    # Initialize the model, loss function, and optimizer
    model = SimpleCNN(kernel_size=kernel_size)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Get the training data from the DDict and create a DataLoader
    inputs = []
    outputs = []
    io_time = 0.0
    for file in os.listdir("./data"): # not ideal at large scale, but okay for this example
        if "inputs" in file:
            tic = perf_counter()
            arrays = np.load(f"./data/{file}")
            io_time += perf_counter() - tic
            inputs.append(torch.from_numpy(np.array(arrays)).unsqueeze(1).float())
        if "outputs" in file:
            tic = perf_counter()
            arrays = np.load(f"./data/{file}")
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
    torch.save(model.to("cpu").state_dict(), f"./data/model_{kernel_size}.pt")
    io_time += perf_counter() - tic
    return io_time


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_sims", type=int, default=4, help="Number of simulations to run")
    parser.add_argument("--grid_size", type=int, default=128, help="Size of the grid for each simulation")
    args = parser.parse_args()

    # Create a fresh data directory
    if os.path.exists("./runinfo/data"):
        shutil.rmtree("./runinfo/data")
    os.makedirs("./runinfo/data")

    # Run an ensemble of simulations (producer)
    with parsl.load(polaris_cpu_config):
        # Compute number of workers
        num_nodes = polaris_cpu_config.executors[0].provider.nodes_per_block
        num_workers_pn = polaris_cpu_config.executors[0].workers_per_node
        num_workers = min(num_nodes * num_workers_pn, args.num_sims)

        # Launch the simulations
        print(f"Launching {args.num_sims} simulations on {num_workers} workers to generate training data ...")
        sim_args = [(period, args.grid_size) for period in np.linspace(40,80,args.num_sims)]
        tic = perf_counter()
        sim_futures = [simulation(*args) for args in sim_args]
        results = []
        while len(sim_futures) > 0:
            future = next(as_completed(sim_futures))
            sim_futures.remove(future)
            results.append(future.result())
        io_time = sum(results)/len(results)
        print(f"Done in {perf_counter() - tic:.2f} seconds")
        print(f"IO time: {io_time:.3f} seconds\n",flush=True)

    # Run an ensemble of CNN models (consumer)
    with parsl.load(polaris_gpu_config):
        # Compute number of workers
        num_nodes = polaris_gpu_config.executors[0].provider.nodes_per_block
        num_workers_pn = polaris_gpu_config.executors[0].workers_per_node
        num_workers = num_nodes * num_workers_pn

        # Launch the CNN models
        print(f"Launching {num_workers} CNN models to train on the simulaiton data ...")
        ml_args = [int(kernel_size) for kernel_size in range(3,2*num_workers+3,2)]
        tic = perf_counter()
        ml_futures = [trainer(arg) for arg in ml_args]
        results = []
        while len(ml_futures) > 0:
            future = next(as_completed(ml_futures))
            ml_futures.remove(future)
            results.append(future.result())
        io_time = sum(results)/len(results)
        print(f"Done in {perf_counter() - tic:.2f} seconds")
        print(f"IO time: {io_time:.3f} seconds\n",flush=True)