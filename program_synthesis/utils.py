import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def softmax_grad(softmax_arr):
    s = np.reshape(softmax_arr, (-1, 1))
    return np.diagflat(s) - np.dot(s, s.T)


def softmax_likelihood_ratio(state, probs, action):
    dsoftmax = softmax_grad(probs)[action, :]
    dlog = dsoftmax / probs[action, 0]
    dlog = dlog[:, None]
    grad = state.T.dot(dlog)
    return grad

def calc_gradient_ascent(grads, rewards, GAMMA, LEARNING_RATE):
    discounted_rewards = []
    sum = 0
    for t in range(len(rewards)):
        Gt = 0
        pw = 0
        for r in rewards[t:]:
            Gt = Gt + GAMMA ** pw * r
            pw = pw + 1
        discounted_rewards.append(Gt)

    for i in range(len(grads)):
        # Loop through everything that happend in the episode and update towards the log policy gradient times
        # **FUTURE** reward
        sum += LEARNING_RATE * grads[i] * discounted_rewards[i]
    return sum
