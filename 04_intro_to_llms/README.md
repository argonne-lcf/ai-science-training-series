# Introduction to Large Language Models 

Author/Perpetrator: Carlo Graziani, including materials on LLMs by Varuni Sastri, and discussion/editorial work by Taylor Childers, Archit Vasan, Bethany Lusch, and Venkat Vishwanath (Argonne)

Word embedding visualizations adapted from Kevin Gimpel (Toyota Technological Institute at Chicago) [Visualizing BERT](https://home.ttic.edu/~kgimpel/viz-bert/viz-bert.html).


This tutorial covers the some fundamental concepts necessary to to study of large language models (LLMs).  The goal is to set the table for Archit Vasan's exploration of LLM pipelines, next week.

## Environment Setup (thanks, Bethany)
1. If you are using ALCF, first log in. From a terminal run the following command:
```
ssh username@polaris.alcf.anl.gov
```

2. Although we already cloned the repo before, you'll want the updated version. To be reminded of the instructions for syncing your fork, click [here](https://github.com/argonne-lcf/ai-science-training-series/blob/main/00_introToAlcf/03_githubHomework.md).

3. We will be downloading data in our Jupyter notebook, which runs on hardware that by default has no Internet access. From the terminal on Polaris, edit the ~/.bash_profile file to have these proxy settings:
```
export HTTP_PROXY="http://proxy-01.pub.alcf.anl.gov:3128"
export HTTPS_PROXY="http://proxy-01.pub.alcf.anl.gov:3128"
export http_proxy="http://proxy-01.pub.alcf.anl.gov:3128"
export https_proxy="http://proxy-01.pub.alcf.anl.gov:3128"
export ftp_proxy="http://proxy-01.pub.alcf.anl.gov:3128"
export no_proxy="admin,polaris-adminvm-01,localhost,*.cm.polaris.alcf.anl.gov,polaris-*,*.polaris.alcf.anl.gov,*.alcf.anl.gov"
```

4. Now that we have the updated notebooks, we can open them. If you are using ALCF JupyterHub or Google Colab, you can be reminded of the steps [here](https://github.com/argonne-lcf/ai-science-training-series/blob/main/01_intro_AI_on_Supercomputer/01_linear_regression_sgd.ipynb). 

5. Reminder: Change the notebook's kernel to `datascience/conda-2023-01-10` (you may need to change kernel each time you open a notebook for the first time):

    1. select *Kernel* in the menu bar
    2. select *Change kernel...*
    3. select *datascience/conda-2023-01-10* from the drop-down menu



## __References:__

I strongly recommend reading ["The Illustrated Transformer"](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar, before next week's deeper dive into Transformer tech by Archit Vasan. Alammar also has a useful post dedicated more generally to Sequence-to-Sequence modeling ["Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/), which illustrates the attention mechanism in the context of a more generic language translation model. 