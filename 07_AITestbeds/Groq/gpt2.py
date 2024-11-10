import torch
import torch.nn as nn
import transformers
try:
    from groqflow import groqit
except:
    raise ImportError("GroqFlow module not found!")

# Instantiate model from transformers library with the corresponding config
model = transformers.GPT2Model(transformers.GPT2Config())

# Create dummy inputs with static dimensions and specified data type
inputs = {
   "input_ids": torch.ones(1, 256, dtype=torch.long),
   "attention_mask": torch.ones(1, 256, dtype=torch.float),
}

# Rock it with GroqIt to compile the model
gmodel = groqit(model, inputs, rebuild="never")
groq_output = gmodel(**inputs) # Run inference on the model on Groq with the GroqIt runtime

print(groq_output) # print outputs in raw form, this should be decoded with a tokenizer in a real life example
