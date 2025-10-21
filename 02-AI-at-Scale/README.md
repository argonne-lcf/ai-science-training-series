
## Slides

Lecture style slides can be found in: [slides/ALCF_AI-students-advanced-02.pdf](slides/ALCF_AI-students-advanced-02.pdf)

## Hands On

The exercise is experimenting with tensor parallelism with [ezpz.examples.fsdp_tp](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/fsdp_tp.py)


1. Submit interactive job:

    ```bash
    qsub -I -l select=1 -l walltime=0:59:00 -q ALCFAITP -l filesystems=home:eagle -A ALCFAITP
    ```

2. Activate the deep learning frameworks:

    ```bash
    module use /soft/modulefiles
    module load conda/2025-09-25
    conda activate base
    ```
3. Run the example:
    ```bash
    export PATH="/opt/pbs/bin:${PATH}"  # workaround, for now
    export HF_HOME=./.cache 
    ezpz-launch python3 -m ezpz.examples.fsdp_tp --dataset random --tp=4 #--n-layers=8
    ```

If the program does not exit cleanly, you might need to kill the python processes manually:

```bash
pkill python
```

## Homework

- Try different combinations of model sizes (layer count with ```--n-layers=?```) and tp-degrees (```--tp=?```) to get an idea of what works
- Document how the performance changes with 8-layer model and TP of 1,2,4 (```--n-layhers=8 --tp=?```)

