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
- medicine (EEG, outcomes)

**Historically popular models and deep learning architectures**
- Autoregressive (AR, MA, ARMA, ARIMA, ARCH, GARCH)
  - Shumway, Robert H., David S. Stoffer. [Time series analysis and its applications](https://www.stat.pitt.edu/stoffer/tsa4/). Vol. 4. New York: Springer, 2017. 
- MLP
- Recurrent neural network (**this tutorial**)
  - with an Attention layer
- Temporal convolutional network (TCN)

**Successors**:
- Transformers and their variants (GPT, BERT, BART, Reformer, Longformer, ...)
- Vision transformers



## Vanilla/Simple Recurrent Neural Network (RNN)
All RNN diagrams from Chirstopher Olah's famous 2015 blog post, [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

![Unrolled simple RNN](media/colah-RNN-unrolled.png)

![Simple RNN cell](media/colah-simple-RNN.png)

![Simple RNN equations](media/simple-rnn-eqs.png)

(this is technically an Elman RNN, not a Jordan RNN). Train through backpropagation through time (BPTT). Both forward and backward passes can be slow since we cannot compute the time-dependencies in parallel (big advantage of transformers)
- Techniques for handling long or uneven sequences: https://machinelearningmastery.com/handle-long-sequences-long-short-term-memory-recurrent-neural-networks/
- `SimpleRNN` in TF/Keras uses a different formulation. See the source code for [`SimpleRNNCell`](https://github.com/keras-team/keras/blob/v2.7.0/keras/layers/recurrent.py#L1255-L1456)

## Long short-term memory (LSTM)
Introduced by [Hochreiter and Schmidhuber (1997)](https://direct.mit.edu/neco/article-abstract/9/8/1735/6109/Long-Short-Term-Memory?redirectedFrom=fulltext), greatly ameliorates the vanishing/exploding gradient problem that simple RNNs suffer from.

![LSTM cell](media/colah-lstm.png)

![LSTM equations](media/lstm-eqs.png)

## Gated recurrent unit (GRU)
Introduce by [Cho (2014)](https://arxiv.org/abs/1406.1078), the fully gated version is just an LSTM minus the output gate and has fewer parameters. 

![GRU cell](media/colah-gru.png)

![GRU equations](media/gru-eqs.png)

## Example application: predicting anomalies in fusion reactors (tokamaks)
See [Kates-Harbeck (2019)](https://www.nature.com/articles/s41586-019-1116-4) for more details.

![DIII-D tokamak](media/d3d_main.jpg)

![FRNN model](media/frnn-model.png)

![FRNN model](media/shot-159593.png)


