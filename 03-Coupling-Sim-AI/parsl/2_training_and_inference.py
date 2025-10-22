from parsl_config import polaris_config
from chemfunctions import compute_vertical, train_model, run_model
import parsl
from parsl.app.app import python_app
from time import monotonic
from random import sample
import pandas as pd
import numpy as np
from concurrent.futures import as_completed

# This example will
# 1. Collect training data by running several simulations
# 2. Train a model using the training data
# 3. Run the model on a large search space of molecules to predict their properties

# Define parameters for the workflow
initial_count: int = 16  # Number of simulations to run for first model training

# Define Parsl apps for each step in the workflow
# Simulation app to compute the ionization energy of a molecule
compute_vertical_app = python_app(compute_vertical)
# Model training app
train_model_app = python_app(train_model)
# Inference app to run the model on a list of SMILES strings
inference_app = python_app(run_model)
# Convenience app to combine multiple inferences into a single DataFrame
@python_app
def combine_inferences(inputs=[]):
    """Concatenate a series of inferences into a single DataFrame
    Args:
        inputs: a list of the component DataFrames
    Returns:
        A single DataFrame containing the same inferences
    """
    import pandas as pd
    return pd.concat(inputs, ignore_index=True)

# Search space of molecules to sample from
search_space = pd.read_csv('../data/QM9-search.tsv', sep=r'\s+')  # Our search space of molecules
search_space_size = len(search_space)

if __name__ == "__main__":

    # Load the Parsl configuration
    with parsl.load(polaris_config):

        start_time = monotonic()  # Start a timer to measure how long the simulations take
        
        print(f"Create initial training data composed of {initial_count}/{search_space_size} random molecules\n")
        
        # Create training data by running several simulations
        # randomly sample molecules from the search space to simulate
        smiles = search_space.sample(initial_count)['smiles']
        futures = [compute_vertical_app(s) for s in smiles]
        print(f'Submitted {len(futures)} simulations to start training ...')

        # Now we wait for the calculations to complete to populate training data
        train_data = []
        while len(futures) > 0:
            # First, get the next completed computation from the list
            future = next(as_completed(futures))
            
            # Remove it from the list of still-running tasks
            futures.remove(future)
            
            # Get the input 
            smiles = future.task_record['args'][0]
            
            # Check if the run completed successfully
            if future.exception() is not None:
                # If it failed, pick a new SMILES string at random and submit it    
                print(f'Computation for {smiles} failed, submitting a replacement computation')
                smiles = search_space.sample(1).iloc[0]['smiles'] # pick one molecule
                new_future = compute_vertical_app(smiles) # launch a simulation in Parsl
                futures.append(new_future) # store the Future so we can keep track of it
            else:
                # If it succeeded, store the result
                print(f'Computation for {smiles} succeeded')
                train_data.append({
                    'smiles': smiles,
                    'ie': future.result(),
                    'batch': 0,
                    'time': monotonic() - start_time
                })
        print("Training data collected!\n")
        train_data = pd.DataFrame(train_data)
        print(train_data)
        
        # Train model
        print("\nStarting training and inference ...")
        train_future = train_model_app(train_data)

        # Chunk the search space into smaller pieces, so that each inference task can run in parallel
        # Use the number of nodes and workers per node to determine how many chunks to create
        num_nodes = polaris_config.executors[0].provider.nodes_per_block  # Get the number of nodes from the config
        num_workers_pn = polaris_config.executors[0].workers_per_node  # Get the number of workers per node from the config
        num_chunks = min(num_nodes * num_workers_pn * 2, len(search_space['smiles']))  # Limit the number of chunks by the number of workers
        chunks = np.array_split(np.array(search_space['smiles']), num_chunks)
        # Create inference tasks, we can pass the train_future to the funtion
        inference_futures = [inference_app(train_future, chunk) for chunk in chunks]

        # We pass the inputs explicitly as a named argument "inputs" for Parsl to recognize this as a "reduce" step
        #  See: https://parsl.readthedocs.io/en/stable/userguide/workflow.html#mapreduce
        predictions = combine_inferences(inputs=inference_futures).result()
        predictions.sort_values('ie', ascending=False, inplace=True)
        print("Training and inference completed!\n")
        print("Inference predictions (sorted by ionization energy):")
        print(predictions)
