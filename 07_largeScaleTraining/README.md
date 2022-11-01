# Large Scale Training 

AI-Driven Science on Supercomputers @ ALCF 2022

**Contact:** [Sam Foreman](samforeman.me) ([foremans@anl.gov](mailto:///foremans@anl.gov))

- **Accompanying slides**: 
    - [📊 Large Scale Training](https://saforem2.github.io/ai4sci-large-scale-training/#/) 
    - [PDF](https://github.com/saforem2/ai4sci-large-scale-training/blob/main/slides.pdf)
    - [github repo](https://github.com/saforem2/ai4sci-large-scale-training)

This section of the workshop will introduce you to some of the tools that we use to run large-scale distributed training on supercomputers.

>  **Note** 
>  <br> Additional examples (+ resources) can be found at:
>  - [ALCF: Computational Performance Workshop](https://github.com/argonne-lcf/CompPerfWorkshop/tree/main/05_scaling-DL)
>  - [ALCF: Simulation, Data, and Learning Workshop for AI](https://github.com/argonne-lcf/sdl_ai_workshop)

## Running

1. Clone / update repo:
    ```bash
    # if not already cloned:
    git clone https://github.com/argonne-lcf/ai-science-training-series
    cd ai-science-training-series/
    git pull
    ```
2. Navigate into `07_largeScaleTraining/src/ai4sci`
3. Run (with `batch_size=512`, for example):
   ```bash
    export BS=512; ./main.sh "batch_size=${BS}" > "main-bs-${BS}.log" 2>&1 &
    ```
4. View output:
    ```bash
    tail "main-bs-${BS}.log" $(tail -1 logs/latest)
    ```
