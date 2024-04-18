# Evaluating LLMs and Potential Pitfalls

Intro to AI-Driven Science on Supercomputers @ ALCF 2024

**Contact:** Marieme Ngom ([mngom@anl.gov](mailto:///mngom@anl.gov)), Bethany Lusch ([blusch@anl.gov](mailto:///blusch@anl.gov)), Sandeep Madireddy  ([smadireddy@anl.gov](mailto:///smadireddy@anl.gov)) 


[Overview of LLMs Evaluation](https://github.com/argonne-lcf/ai-science-training-series/blob/main/08_Evaluating_LLMs/LLM_Evaluation_Overview.pdf)

[Potential Pitfalls of LLMs](https://github.com/argonne-lcf/ai-science-training-series/blob/main/08_Evaluating_LLMs/LLM-Pitfalls.pdf)
    
[Link to breakout rooms forms](https://drive.google.com/drive/folders/1BN_aBlNU-7KVIcySntRtbkBXRGpkMSyz)

Other helpful links:
- [OpenAI tokenizer](https://platform.openai.com/tokenizer)
- [Chatbot Arena](https://chat.lmsys.org/)
- [Chatbot Guardrails Arena](https://huggingface.co/spaces/lighthouzai/guardrails-arena)

 
 **Homework**
 
What do you think is a particularly good use case for LLMs for science? How would you evaluate it?
Your answer does not need to be in paragraphs. When you submit your homework form, you can link to a file in your Github repo where you wrote your answer.

**Answer**

I think a particularly interesting and powerful use for LLMs are those that are trained on DNA, such as DNABERT2, Nucleotide Transformer, EVO, HyenaDNA and Caduceus.  This is a very new area of research, and no one has yet trained an extremely large model (at the scale of the commonly used LLMs) but I think when someone has enough money and resources to do that, it will have uses in precision medicine, potential pharmaceudical targets, and in basic science research by helping us understand the language of DNA.

Currently the evaluations of these genomic language models has used some simple classification tasks as benchmarks, such as identifying promoters, regulatory elements and splice sites.  Since these elements have been identified experimentally, we have a ground truth that we can measure against and use metrics like accuracy, f1, precision and recall to evaluate their performance.

As the tasks become more challenging and generative, we will need to come up with different metrics to evaluate them.  SInce humans do not understand the langauge of DNA, the evaluation of things like generative sequences will be more difficult. One idea is to use a second model to judge the output of the first  model to determine if it "makes sense", in the same way a human would give a LLM a fluency or accuracy score.

