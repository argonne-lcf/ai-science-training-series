# BERT on Sambanova


##### Clone Repo

```bash
git clone https://github.com/argonne-lcf/ai-science-training-series.git/
cd ai-science-training-series/07_AITestbeds/Sambanova/bert/
chmod +x BertLarge.sh
```

##### Run the script 

This script will compile as well as run BERT example. 
```bash
./BertLarge.sh
```
The script outputs the path of the compile log saying 
```bash
Using <path_to_compile_logs>/BertLarge.out for output
```
You can view the log in a seperate terminal while the compilation is ongoing. Once the compilation is complete, look for the slurm-id in the same log file. Or you can also use the `squeue` command to know the slurm ID associated with your run. Use ~/slurm-<id>.out to get the path of the training log. 
