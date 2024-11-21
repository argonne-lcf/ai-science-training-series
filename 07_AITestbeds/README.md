# Intro to AI Series: AI Accelerators


Scientific applications are increasingly adopting Artificial Intelligence (AI) techniques to advance science. There are specialized hardware accelerators designed and built to run AI applications efficiently. With a wide diversity in the hardware architectures and software stacks of these systems, it is challenging to understand the differences between these accelerators, their capabilities, programming approaches, and how they perform, particularly for scientific applications. 

We will cover an overview of the AI accelerators landscape with a focus on SambaNova, Cerebras, Graphcore, and Groq systems along with architectural features and details of their software stacks. We will have hands-on exercises that will help attendees understand how to program these systems by learning how to refactor codes written in standard AI framework implementations and compile and run the models on these systems. 



## Slides

* [Intro to AI Series: AI Accelerators]() 
    > Slides will be uploaded shortly after the talk.

## Hands-On Sessions


* [Cerebras](./Cerebras/README.md)
* [Graphcore](./Graphcore/README.md)  
* [SambaNova](./Sambanova/README.md)                                    
* [Groq](./Groq/README.md)        


### Director’s Discretionary Allocation Program

To gain access to AI Testbeds at ALCF after current allocation expires apply for [Director’s Discretionary Allocation Program](https://www.alcf.anl.gov/science/directors-discretionary-allocation-program)

The ALCF Director’s Discretionary program provides “start up” awards to researchers working to achieve computational readiness for for a major allocation award.

## Homework 

You need to submit either Theory Homework or Hands-on Homework. 

#####  Theory Homework
* What are the key architectural features that make these systems suitable for AI workloads?
* Identify the primary differences between these AI accelerator systems in terms of their architecture and programming models.
* Based on hands-on sessions, describe a typical workflow for refactoring an AI model to run on one of ALCF's AI testbeds (e.g., SambaNova or Cerebras). What tools or software stacks are typically used in this process?
* Give an example of a project that would benefit from AI accelerators and why?


<details>
<summary>Theory Homework Solutions</summary>

1. **What are the key architectural features that make these systems suitable for AI workloads?**
   The key architectural features that make AI accelerators like SambaNova, Cerebras, Graphcore, and Groq systems suitable for AI workloads are:
   1. Specialized Hardware Design to accelerate matrix multiplications and tensor operations.
   2. High Memory Bandwidth and larger amount of on-chip memory help to accelerate memory intensive AI worklaods. 
   3. Scalability and Parallelism: Parallel processing of data across many cores or processing units, which significantly speeds up training and inference tasks


2. **Identify the primary differences between these AI accelerator systems in terms of their architecture and programming models.**
   
    1.  Sambanovas Reconfigurable Dataflow Unit (RDU) allows for flexible dataflow processing that features a multi-tiered memory architecture with terabytes of addressable memory for efficinet handling of large data. 
    2.  Cerebras Wafer-Scale Engine (WSE) consists of processing elements (PEs) with its own memory and operates independently. Fine-grained dataflow control mechanism within its PEs make the system highly parallel and scalable.
    3. Graphcore’s Intelligence Processing Unit (IPU) consists of many interconnected processing tiles, each with its own core and local memory. The IPU operates in two phases—computation and communication—using Bulk Synchronous Parallelism (BSP).
    4. Groq’s Tensor Streaming Processor (TSP) architecture focuses on deterministic execution which s particularly advantageous for inference tasks where low latency is critical.


3. **Based on hands-on sessions, describe a typical workflow for refactoring an AI model to run on one of ALCF's AI testbeds (e.g., SambaNova or Cerebras). What tools or software stacks are typically used in this process?**

    Typical worksflow involves using vendor specific implementation of ML framework like PyTorch to port model. Refer to following documentation examples to understand details of workflow. 
    * [PyTroch to PopTroch](https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/pytorch_to_poptorch.html)
    * [Sambaflow Model Conversion](https://docs.sambanova.ai/developer/latest/porting-overview.html)
</details>


##### Hands-on Homework

* [Cerebras Homework](./Cerebras/README.md#homework)
* [Sambanova Homework](./Sambanova/README.md#homework)
* [Graphcore Homework](./Graphcore/README.md#homework)
* [Groq Homework](./Groq/README.md#homework)

## Useful Links 

* [Overview of AI Testbeds at ALCF](https://www.alcf.anl.gov/alcf-ai-testbed)
* [ALCF AI Testbed Documentation](https://www.alcf.anl.gov/support/ai-testbed-userdocs/)
* [Director’s Discretionary Allocation Program](https://www.alcf.anl.gov/science/directors-discretionary-allocation-program)
* [ALCF Events Page](https://www.alcf.anl.gov/events/intro-ai-series-ai-accelerators-0)  

##### Acknowledgements

Contributors: [Siddhisanket (Sid) Raskar](https://sraskar.github.io/), [Murali Emani](https://memani1.github.io/), [Varuni Sastry](https://www.alcf.anl.gov/about/people/varuni-katti-sastry), [Bill Arnold](https://www.alcf.anl.gov/about/people/bill-arnold), and  [Venkat Vishwanath](https://www.alcf.anl.gov/about/people/venkatram-vishwanath).

> This research used resources of the Argonne Leadership Computing Facility, which is a DOE Office of Science User Facility supported under Contract DE-AC02-06CH11357.
