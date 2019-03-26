import numpy as np


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def softmax_grad(softmax_arr):
    s = np.reshape(softmax_arr, (-1, 1))
    return np.diagflat(s) - np.dot(s, s.T)


def softmax_likelihood_ratio(state, probs, action):
    dsoftmax = softmax_grad(probs)[action, :]
    dlog = dsoftmax / probs[action, 0]
    grad = state.T.dot(dlog)
    return grad[0]
