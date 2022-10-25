# this code is based on pieces of the first assignment from Stanford's CSE231n course, 
# hosted at https://github.com/cs231n/cs231n.github.io with the MIT license

import numpy as np
from scipy.special import logsumexp

def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, D) and contains a minibatch of N
    examples, where each example x[i] has length D. We will
    transform each example to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, D)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = np.dot(x, w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, D)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, D)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None

    dx = np.dot(dout, w.T) # (N, M) x (M, D) = (N, D)
    dw = np.dot(x.T, dout) # (D, N) x (N, M) = (D, M)
    db = dout.sum(axis=0)

    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = np.maximum(0,x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    mask = x >= 0
    dx = dout * mask

    return dx

def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.
    More commonly known as categorical cross-entropy loss

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    def softmax(x):
        # non-standard definition in order to have better numerical stability
        # logsumexp function avoids some issues with overflow during sum of exponentials
        #normalization = np.sum(np.exp(x),axis=1) # (N, 1)
        #normalization = np.exp(logsumexp(x, axis=1)) 
        temp = logsumexp(x, axis=1)
        #q = np.exp(x) / normalization[:,None] # (N, C) 
        q = np.exp(x - temp[:,None])
        # logsumexp function avoids some issues with overflow during sum of exponentials
        #normalization = np.sum(np.exp(x),axis=1) # (N, 1)
        #normalization = np.exp(logsumexp(x, axis=1)) 
        temp = logsumexp(x, axis=1)
        #q = np.exp(x) / normalization[:,None] # (N, C) 
        q = np.exp(x - temp[:,None])
        return q
    
    q = softmax(x)

    # some clipping to avoid log of 0 or negative number
    epsilon = 1e-07
    q = np.clip(q, a_min = epsilon, a_max=1-epsilon)

    N = x.shape[0]
    loss = -np.sum(y * np.log(q), axis=1) # sum over classes
    loss = np.mean(loss) # average over examples

    # calculating gradients
    dx = (1./N) * softmax(x) # (N, C)
    # for each example, column for correct class needs another term: -(1/N)
    extra_term = (-1./N) * y  # element-wise multiply, p acts as mask
    dx = dx + extra_term

    return loss, dx
