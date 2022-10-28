# Large Scale Training 

AI-Driven Science on Supercomputers @ ALCF 2022

**Contact:** [Sam Foreman](samforeman.me) ([foremans@anl.gov](mailto:///foremans@anl.gov))

This section of the workshop will introduce you to some of the tools that we use to run large-scale distributed training on supercomputers.

>  **Note** 
>  <br> Additional examples (+ resources) can be found at:
>  - [ALCF: Computational Performance Workshop](https://github.com/argonne-lcf/CompPerfWorkshop/tree/main/05_scaling-DL)
>  - [ALCF: Simulation, Data, and Learning Workshop for AI](https://github.com/argonne-lcf/sdl_ai_workshop)

To setup:

1. Clone / update repo:
  ```bash
  # if not already cloned:
  git clone https://github.com/argonne-lcf/ai-science-training-series
  cd ai-science-training-series/07_largeScaleTraining/
  git pull
  ```

  2. Install `ai4sci` into virtual environment:
  ```bash
  python3 -m venv --system-site-packages
  source venv/bin/activate
  python3 -m pip install --upgrade pip
  python3 -m pip install -e .
  # to be sure
  python3 -c "import ai4sci; print(ai4sci.__file__)"
  # should print /path/to/ai-science-training-series/src/ai4sci/__init__.py
  ```
