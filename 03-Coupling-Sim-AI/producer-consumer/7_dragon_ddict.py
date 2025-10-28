import dragon
import multiprocessing as mp

from dragon.data.ddict import DDict
from dragon.native.machine import System, Node

def toy_function(dd: DDict):
    """Toy function to demonstrate the use of a Dragon DDict
    Args:
        dd: dragon DDict object
    """
    input = dd["input"]
    dd["output"] = input + "from the toy function"
    

if __name__ == "__main__":
    # Set the mp start method
    mp.set_start_method("dragon")

    # Get allocation info
    alloc = System()
    num_nodes = alloc.nnodes
    nodelist = alloc.nodes
    head_node = Node(nodelist[0])
    print(f"Dragon running on {num_nodes} nodes")
    print([Node(node).hostname for node in nodelist],"\n",flush=True)

    # Initialize the DDict on all the nodes
    ddict_mem_per_node = 0.05 * head_node.physical_mem # dedicate 5% of each node's memory to the DDict
    tot_ddict_mem = int(ddict_mem_per_node * num_nodes)
    managers_per_node = 1
    dd = DDict(managers_per_node, num_nodes, tot_ddict_mem)
    print(f"Started DDict on {num_nodes} nodes with {tot_ddict_mem/1024/1024/1024:.1f}GB of memory\n",flush=True)

    # Put input data into the DDict
    dd["input"] = "Hello "

    # Run the toy function
    toy_function(dd)

    # Get the result back from the DDict
    result = dd['output']
    print(result, flush=True)

    # Clean up
    dd.destroy()