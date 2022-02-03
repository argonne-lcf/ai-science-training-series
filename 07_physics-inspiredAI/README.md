# Physics-Inspired AI
Led by Bethany Lusch from ALCF (<blusch@anl.gov>) and Shawn Rosofsky from University of Illinois at Urbana-Champaign

Outline:
1. Overview of approaches (by Bethany, see below)
2. Invariances in CNNs (by Bethany, [Rotated-Mnist](Rotated-Mnist.ipynb))
3. Physics-Informed Neural Networks (PINNs) by Shawn: 
    * [Slides](PhysicsInspiredAI.pdf)
    * [Burgers Equation](Burgers.ipynb)
    * [Poisson Lshape](Poisson_Lshape.ipynb)
    * [Complex Geometry](Complex_Geometry.ipynb)
    * [Lorenz Inverse System](Lorenz_inverse_system.ipynb)
4. Physics-Informed DeepONets by Shawn: 
    * [Slides](PhysicsInspiredAI.pdf)
    * [DeepONetsPI Diffusion Reaction](DeepONetsPI_Diffusion_Reaction.ipynb)


## Overview
There are many ways to incorporate domain knowledge, such as known physics, into AI, and this is a rapidly-growing area of research. Incorporating previous knowledge into AI can have many benefits, such as: 
* A more accurate or simpler model or less need for training data because it doesn't have to "rediscover" patterns we already know
* More interpretable models
* Models that are more robust or trustworthy because, for example, they do not violate conservation of energy
* Leveraging the generalizability of physical laws or equations to make AI models more generalizable 

For these reasons and more, incorporating domain knowledge in ML and AI was listed as a grand challenge in the Department of Energy's [AI for Science Report](https://www.anl.gov/ai-for-science-report). 

Some examples of existing approaches to incorporating physics in AI are:
1. Embedding symmetries or invariances in a network, as we will see in the [Rotated-Mnist](Rotated-Mnist.ipynb) notebook.
2. Relatedly: applying physical constraints, such as conservation laws.
3. Creating custom loss functions.
4. Carefully choosing input representations so that all relevant physical information is provided.
5. Constraining the network to learn a solution to the known differential equation, as we will see in the Physics-Informed Neural Networks (PINNs) section.
6. Or, to generalize PDE solutions across changes like different initial conditions, etc. learning an operator network, as we will see in the Physics-Informed DeepONets section.
7. Learning governing differential equations from data, with the assumption that certain types of terms are allowed and/or there should only be a few terms.
8. Learning a hybrid model, such as having a neural network learn only the behavior not covered by given differential equations. (Related terms: gray-box modeling, closure modeling, and discrepancy modeling.)
9. Training an ML model as a surrogate for only part of a simulation, then feeding the prediction back into the simulation. 

For more, check out the "ML for Physics and Physics for ML" [tutorial](https://neurips.cc/Conferences/2021/ScheduleMultitrack?event=21896) from NeurIPS 2021 by Shirley Ho and Miles Cranmer. 
