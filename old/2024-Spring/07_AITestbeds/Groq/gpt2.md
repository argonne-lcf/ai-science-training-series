# GPT2 On Groq

GPT-2 is a transformers model pretrained on a very large corpus of English data in a self-supervised fashion. This means it was pretrained on the raw texts only, with no humans labelling them in any way (which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those texts. More precisely, it was trained to guess the next word in sentences.

More precisely, inputs are sequences of continuous text of a certain length and the targets are the same sequence, shifted one token (word or piece of word) to the right. The model uses internally a mask-mechanism to make sure the predictions for the token i only uses the inputs from 1 to i but not the future tokens.

This way, the model learns an inner representation of the English language that can then be used to extract features useful for downstream tasks. The model is best at what it was pretrained for however, which is generating texts from a prompt.

#### Get a Groq node interactively

```bash
qsub -I -l walltime=1:00:00
```

#### Go to directory with GPT2 example. 
```bash
git clone git@github.com:argonne-lcf/ai-science-training-series.git
cd 07_AITestbeds/Groq
```

#### Activate groqflow virtual Environment 
```bash
conda activate groqflow
```

#### Install Requirements 

Install the python dependencies using the following command:
```bash
pip install transformers
```

#### Run Inference Job

```bash
python gpt2.py
```

<!-- #### Run end-to-end Inference Job with WikiText dataset

```bash
python GPT2-wiki.py
``` -->