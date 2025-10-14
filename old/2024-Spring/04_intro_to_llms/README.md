# Introduction to Large Language Models 

Author: Archit Vasan , including materials on LLMs by Varuni Sastri and Carlo Graziani at Argonne, and discussion/editorial work by Taylor Childers, Bethany Lusch, and Venkat Vishwanath (Argonne)

Inspiration from the blog posts "The Illustrated Transformer" and "The Illustrated GPT2" by Jay Alammar, highly recommended reading.

This tutorial covers the some fundamental concepts necessary to to study of large language models (LLMs).

## Brief overview
* Scientific applications for language models
* General overview of Transformers
* Tokenization
* Model Architecture
* Pipeline using HuggingFace
* Model loading

## Sophia Setup
1. If you are using ALCF, first log in. From a terminal run the following command:
```
ssh username@sophia.alcf.anl.gov
```

2. Although we already cloned the repo before, you'll want the updated version.  To be reminded of the instructions for syncing your fork, click [here](https://github.com/argonne-lcf/ai-science-training-series/blob/main/00_introToAlcf/03_githubHomework.md).

3. Now that we have the updated notebooks, we can open them. If you are using ALCF JupyterHub or Google Colab, you can be reminded of the steps [here](https://github.com/argonne-lcf/ai-science-training-series/blob/main/01_intro_AI_on_Supercomputer/01_linear_regression_sgd.ipynb). 

5. Reminder: Change the notebook's kernel to `datascience/conda-2024-08-08` (you may need to change kernel each time you open a notebook for the first time):

    1. select *Kernel* in the menu bar
    2. select *Change kernel...*
    3. select *datascience/conda-2024-08-08* from the drop-down menu
  
## Google colab setup
In case you have trouble accessing Sophia, all notebook material can be run in google colab.

Just:
1. Go to this link: [Colab](https://colab.research.google.com/#scrollTo=Wf5KrEb6vrkR)
2. Click on `File/Open notebook`
3. Nagivate to the `GitHub` tab and find `argonne-lcf/ai-science-training-series`
4. Click on `04_intro_to_llms/IntroLLMs.ipynb`
   
## __References:__

I strongly recommend reading ["The Illustrated Transformer"](https://jalammar.github.io/illustrated-transformer/) by Jay AlammarAlammar also has a useful post dedicated more generally to Sequence-to-Sequence modeling ["Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/), which illustrates the attention mechanism in the context of a more generic language translation model.

## Homework solutions
Solutions to homework problems are posted in IntroLLMHWSols.ipynb
To see BertViz attention mechanisms, simply open the notebook in google colab.
