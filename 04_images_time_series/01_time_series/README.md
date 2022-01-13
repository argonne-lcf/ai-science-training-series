# Time series data


- Classification
- Regression 
- Prediction
- Compression


Applications
- natural language processing
- speech recognition
- traffic forecasting (see Diffusion Convolutional Recurrent Neural Network (DCRNN))


Successors:
- Transformers and their variants (GPT, BERT, BART, Reformer, Longformer, ...)
- Vision transformers

All RNN diagrams from Chirstopher Olah's famous 2015 blog post, [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

## Vanilla/Simple Recurrent Neural Network (RNN)

![Unrolled simple RNN](media/colah-RNN-unrolled.png)

![Simple RNN cell](media/colah-simple-RNN.png)


## Long short-term memory (LSTM)
Introduced by [Hochreiter and Schmidhuber (1997)](https://direct.mit.edu/neco/article-abstract/9/8/1735/6109/Long-Short-Term-Memory?redirectedFrom=fulltext)

$$
\begin{align}
f_t &= \sigma_g(W_{f} x_t + U_{f} c_{t-1} + b_f) \\
i_t &= \sigma_g(W_{i} x_t + U_{i} c_{t-1} + b_i) \\
o_t &= \sigma_g(W_{o} x_t + U_{o} c_{t-1} + b_o) \\
c_t &= f_t \circ c_{t-1} + i_t \circ \sigma_c(W_{c} x_t + b_c) \\
h_t &= o_t \circ \sigma_h(c_t)
\end{align}
$$


## Gated recurrent unit (GRU)


## Example application: predicting anomalies in fusion reactors (tokamaks)
See [Kates-Harbeck (2019)](https://www.nature.com/articles/s41586-019-1116-4) for more details.

![DIII-D tokamak](media/d3d_main.jpg)

![FRNN model](media/frnn-model.png)

![FRNN model](media/shot-159593.png)


