import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]
  C = W.shape[1]
  D = X.shape[1]
  dS = np.zeros([N, C])

  for i in range(N):
    s = np.matmul(X[i], W)
    expssum = 0
    for j in range(C):
      expssum += np.exp(s[j])

    Pi = np.exp(s[y[i]]) / expssum
    Li = -np.log(Pi)
    loss += Li

    dS[i, :] = -1 / N / Pi
    for j in range(C):
      if j != y[i]:
        dS[i, j] *= np.exp(s[y[i]]) * (-1) / (expssum ** 2) * np.exp(s[j])
      else:
        dS[i, y[i]] *= (np.exp(s[y[i]]) * expssum - np.exp(s[y[i]]) ** 2 ) / (expssum ** 2)
      for k in range(D):
        dW[k, j] += dS[i, j] * X[i, k]

  loss /= N
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]
  S = np.matmul(X, W) # (N, C)
  expS = np.exp(S)
  expSy = expS[np.arange(N), y].reshape(-1, 1)
  expSsum = np.sum(expS, axis=1).reshape(-1, 1)
  expSsum2 = np.square(expSsum).reshape(-1, 1)

  Pi = np.divide( expSy, expSsum )
  loss = np.mean(-np.log(Pi))

  dS = -1 * np.multiply(expSy, expS)
  dS[np.arange(N), y] += np.multiply(expSy, expSsum).reshape(-1)
  dS = np.divide(dS, expSsum2)
  dS = - 1/N * np.divide(dS, Pi)
  dW = np.matmul(X.T, dS)

  loss += reg * np.sum(W * W)
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

