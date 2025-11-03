# Building Agentic AI

Material created by Thang Pham and Murat Keçeli from Argonne National Laboratory.

## 📘 Introduction

This hands-on tutorial introduces **LangGraph** and **LangChain** for constructing agentic AI workflows in scientific computing.  
Participants will learn how to build **single-agent** and **multi-agent** systems that can use scientific tools (e.g., molecule generation, simulation) to automate research tasks.

By the end of the session, you will understand:

- How LangGraph represents agent workflows as graphs.
- How to create a simple **React agent** using prebuilt tools.
- How to build a custom **React agent** from scratch.
- How to design a **multi-agent system** with structured outputs and message passing.

## Hands On Exercises

0. Clone the repository or pull from the main branch:

    ```bash
    cd /path/to/desired_location
    git clone https://github.com/argonne-lcf/ai-science-training-series.git

    OR

    cd /path/to/desired_location/ai-science-training-series
    git pull origin main
    ```

1. Submit an interactive job:

    ```bash
    qsub -I -l select=1 -l walltime=01:00:00 -q ALCFAITP -l filesystems=home:eagle -A ALCFAITP
    ```

2. Source the environment provided:

    ```bash
    cd /path/to/desired_location/ai-science-training-series/04-Inference-Workflows
    source 0_activate_env.sh
    ```
## Tutorials Overview

Each exercise builds upon the previous one.
The goal is to gradually move from using prebuilt agents to designing your own.

### Example 1: 1_simple_react_agent.py
#### Objective:
Learn how to use the prebuilt React agent provided by LangGraph:
- Basic LangGraph setup.

- Prompting the agent with natural language.

- Understanding how the agent calls tools automatically.

- Using ALCF Inference Endpoints service.

#### Run:
```bash
python 1_simple_react_agent.py
```

#### Optional:
Adjust the current query: "Optimize the structure of a water molecule using MACE" to some others and see how the results change:

- "Calculate the energy of a carbon dioxide molecule"

- "Run geometry optimization for a methanol molecule using MACE"

### Example 2: 2_build_react_agent.py
#### Objective:
Construct your own React agent using LangGraph:
- Building `StateGraph`.
- Building `science_agent` and routing logics.

#### Run:
```bash
python 2_build_react_agent.py
```

### Example 3: 3_build_multi_agent.py
#### Objective:
Create a simple multi-agent system that combines two agents:
- A Tool Agent — handles tool calls.
- A Summary Agent — returns a JSON output.

#### Run:
```bash
python 3_build_multi_agent.py
```
