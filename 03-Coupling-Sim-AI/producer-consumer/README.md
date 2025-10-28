# Producer-Consumer Workflow with Parsl and DragonHPC




## Run Instructions

1. Submit an interactive job:

    ```bash
    qsub -I -l select=1 -l walltime=01:00:00 -q ALCFAITP -l filesystems=home:eagle -A ALCFAITP
    ```

2. Source the environment provided:

    ```bash
    source ../0_activate_env.sh
    ```

3. Run the workflow implemented with Parsl and transfering data through concurrent futures

    ```bash
    python 5_parsl_fut_producer_consumer.py
    ```

4. Run the workflow implemented with Parsl and sharing data through the file system 

    ```bash
    python 6_parsl_fs_producer_consumer.py
    ```

5. Run the introductory example to DragonHPC and the Distributed Dictionary (DDict)

    ```bash
    python 7_dragon_ddict.py
    ```

5. Run the workflow implemented with DragonHPC and the DDict

    ```bash
    python 8_dragon_producer_consumer.py
    ```


## Data Transfer Performance (Homework)

Using the Parsl and DragonHPC implementations of the producer-consumer workflow, fill in the table below with the execution and I/O times reported by the scripts. Feel free to add as many rows as desired based on your tests by editing this README.md file and adding lines to the table.

To increase the size of the training data produced, use the `--num_sims` and `--grid_size` arguments to the workflow scripts. For an example of how to do this, and in order to run on multiple nodes, see the [9_submit_multinode.sh](./9_submit_multinode.sh) job submit script. 
Please note:

* To submit the script, execute `qsub ./9_submit_multinode.sh` from a Polaris login node. The results will be written to a file called `producer-consumer.o<PBS_job_ID>`. Outside of lecture time, please use the `debug` or `debug-scaling` queues (the `#PBS -q` parameter in the script).
* To run on a different number of nodes, change the value of the line `#PBS -l select=2`, although 2 nodes should be sufficient to capture the trends.
* The `--num_sims` argument determines the total number of simulations to execute, and thus how much total training data is produced, not the number of simulations per node. As you run on multiple nodes, also increase the number of simulations if you wish to produce more data.
* The size of the training data generated is reported in GB by the workflow scripts. 
* For the Parsl + futures implementation, there is no reported IO time since the data is serialized and "streamed" from the main process to the workers. Use the run time values and comparisons with the Parsl + file system implementation results with the same setup (nodes and data size) to infer how this transfer time changes as the problem is scaled up.
* Parsl and DragonHPC use different methods for launching processes, which can impact the run time reported by the scripts. Focus on the IO time when comparing the DDict and file system performance. 


| Implementation   | Number of Nodes | Training Data Size (GB) | Simulation Run / IO Time (sec) | Training Run / IO Time (sec) |
|------------------|-----------------|--------------------|-----------------|---------------|
| Parsl + futures | 1   | 0.62   | 14.38 / NA   | 26.59 / NA   |
| Parsl + file system | 1   | 0.62   | 11.22 / 0.094   | 14.90 / 0.422   |
| DragonHPC + DDict | 1   | 0.62   | 7.01 / 0.233   | 17.92 / 1.194   |
| ...   | ...   | ...   | ... / ...  | ... / ...  |


**Observations**

Write a short paragraph on your observations based on the results collected in the table above. Which solution is best depending on the number on the size of data being produced and transferred and the number of nodes used? Does this match your expectations? 
