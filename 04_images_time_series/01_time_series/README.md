# Time series data

**Types of problems**
- Classification
- Regression 
  - Prediction/Forecasting
- Compression

**Example applications**
- natural language processing
- speech recognition
- traffic forecasting (see Diffusion Convolutional Recurrent Neural Network (DCRNN))
- electrical grid management
- earthquake prediction

**Historically popular models and deep learning architectures**
- Autoregressive (AR, MA, ARMA, ARIMA)
  - https://www.stat.pitt.edu/stoffer/tsa4/
- MLP
- Recurrent neural network (this tutorial)
- Temporal convolutional network (TCN)

**Successors**:
- Transformers and their variants (GPT, BERT, BART, Reformer, Longformer, ...)
- Vision transformers

All RNN diagrams from Chirstopher Olah's famous 2015 blog post, [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

## Vanilla/Simple Recurrent Neural Network (RNN)

![Unrolled simple RNN](media/colah-RNN-unrolled.png)

![Simple RNN cell](media/colah-simple-RNN.png)

![Simple RNN equations](media/simple-rnn-eqs.png)
(technically an Elman, not Jordan RNN). Train through backpropagation through time (BPTT)

## Long short-term memory (LSTM)
Introduced by [Hochreiter and Schmidhuber (1997)](https://direct.mit.edu/neco/article-abstract/9/8/1735/6109/Long-Short-Term-Memory?redirectedFrom=fulltext), greatly ameliorates the vanishing/exploding gradient problem that simple RNNs suffer from.

![LSTM cell](media/colah-lstm.png)

![LSTM equations](media/lstm-eqs.png)

## Gated recurrent unit (GRU)
Introduce by [Cho (2014)](https://arxiv.org/abs/1406.1078), the fully gated version is just an LSTM minus the output gate and has fewer parameters. 

![GRU cell](media/colah-GRU.png)

![GRU equations](media/gru-eqs.png)

## Example application: predicting anomalies in fusion reactors (tokamaks)
See [Kates-Harbeck (2019)](https://www.nature.com/articles/s41586-019-1116-4) for more details.

![DIII-D tokamak](media/d3d_main.jpg)

![FRNN model](media/frnn-model.png)

![FRNN model](media/shot-159593.png)


