---
date: 6 Feb 2024
lecturer: Shouwei Gao
---

# Intro to AI on ALCF Super Computer

## The Evolution of Computing Systems
- File: Evolution.md
- Computing is a method using the environment to model quantitative ideas in a way that allows us to manipulate the ideas by manipulating the environment.
    - The simplest computer is counting on our fingers. However, this is limited by the number of registers they have (10, or 9 if you use one finger for counting)
    - The earliest computer are the chinese "counting rods" (~1600 BC)
    - The abacus appeared around 900 AD
    - The slide rule appeared around 1600 AD 
    - All of these require the mind to know the rules for operating the machine.
    - The first fully automated computer, which required no human counting, were invented by Leibniz (1642) and Pascal (16XX)
    - The first electronic computer was Eniac, invented by Alan Turing (1942). It could perform 5k operations / second. Humans can do ~1-2 math operations / second.
    - The electronic calculator was invented in 1970
    - The power of of a current Macbook Pro (2024) is ~1 Teraflop (10^12 operations per second)
- Moore's Law
    - Moore's Law = as transistors get smaller, you can fit more and more on a board. However, there is a limit to how small they can get.
    - Quantum Computers are the absolute limit. May be available to the consumer public in ~10 years or so.
- Artificial Intelligence
    - Traditional CPU architecture is not great for training AI, so new architectures are being invented
- Super Computers 
    - Massively parallelize computing by connecting many CPUs and GPUs together with high-speed cable
    - Traning a large language models can take 90 days using >350GPUS and 1000s of CPUs. More than a human lifetime on a laptop.
    - Aurora, the second largest super computer in the world, is >1 Exoflop or 10^18 flops (~1 million times faster than my laptop)

## How Parallel Computing Works to Create Super Computers 
- Parallel computing is basically MapReduce: 
    - it is breaking a problem down into small pieces in order to allow many machines to work on them at once and then combine their results together at the end.
    - Example (mpi_pi notebook): Calculate PI by generating points randomly distributed inside a square inscribed with a circle 
        - Map: Each CPU generates 1m points, then calculates how many are inside the square. 
        - Reduce: One CPU recieves the 5m points and sums the number of points inside each square.
        - Run the Example: Try running with different numbers of CPUs (-np 1, 2, etc.) and see the time differences for running the calculation.
- Model Parallelism for Training AI
    1. You can partition the dataset, giving part to each CPU (simplest, but not the best)
    2. If you have a very large model that is too big to fit in 1 GPU, give a 1 layer of neurons to each GPU 
    - Guiding Principles:
        1. Load balance work across compute nodes in order to minimize compute time (ideally, every cpu/gpu should be pegged; nobody should be blocked or idle).
        2. minimize communication between worker nodes


## How Large Language Models Work (LLM AI)
- AI in a Nutshell
    - This part of lecture is based on: 01_linear_regession_sgd.ipynb
    - Two ways of learning: 
        - Rules Based Learning (Grammar/Translation Method: slow and error prone)
        - Example Bases Learning (SLA Method: requires lots of comprehensible imput, allows patterns to be discovered)
            - data driven language learning
            - if parents give lots of examples of good patterns, children pick it up much better than making a rule they should follow.
                - the efficeny of the rule (its lack of practice) is its WEAKNESS for learning
    - Input Example: With enough examples and context, LLM can predict the appropriate word 
- Linear Regression is probably the simplest form of AI training
    - Figuring out where the line is simply means learning the pattern of where the dots are most likely to fall.
    - Housing Price Example:
        - Theory: the larger the house (GrLivArea), the higher the price (SalePrice). 
        - This is a simple linear relationship 
            - The Rule Based (grammar/translation) method is using the formula y = mx+b
            - The Example Based (SLA) method (actually is Stochastic Gradient Descent (since we know we want a line).
                - Loss Function: Mean Square Error
                - Calculate the lowest point of the loss function by walking step by step down the slope until we find the minimum (partial derivative)
                - TERM: Learning Rate (= slope of the loss function at a point)
                - Best practice: in order to avoid getting stuck in a local minimum, start with bigger step size in step function, then go to smaller steps 
                    - What we want to do in this situation is start with a large learning rate and slowly reduce its size as we progress

## Homework: Mini Batch Training
- Assignment is at the bottom of 01_linear_regession_sgd.ipynb
- Task: Adjust the SGD training code in the notebook so that it partitions the training data into small batches

## Science Talk: Arvind Ramanathan (Data Scientist)
- Title: Autonomous Discovery for Biological Systems Design
- Problem: How do we design new organisims (life) from scratch?
- Importance: Can be used to create new medicinces or better enzymes (e.g. for beer making) that are faster, but don't break anything else in the system.
- Methods: Bring robotics, ai, and biology together 
- Sample Question: How did Sars-Cov2 genomes evolve?
- Train a Transformer-architectures LLVM on the genetic data and see what it predicts in 2 years.
- Scientific Achievment: Figured out a better architecture to speed up the training process (from 2 weeks on Polaris to 1.5 days on much smaller Cerberas)
- Method: Start with pre-trained LLVM (seed model), use it to generate "the right answers" for training a new LLVM that can create new, unseen sequences that are still "correct" (i.e. a new organism/protein structure).

- You can ask the AI to design the experiement without human intervention
    - Give the AI a high level description of a starting hypothesis, then have it close the loop on planning, execution, and analysis of experiements.



