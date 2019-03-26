import numpy as np
import random
from utils import *
import brainfuck

chars = ['>', '<', '+', '-', '.', ',', '[', ']', '\n']
vocab_size = len(chars)
char_to_ix = {ch: i for i, ch in enumerate(sorted(chars))}
ix_to_char = {i: ch for i, ch in enumerate(sorted(chars))}
program_max_length = 500


def char_to_ind():
    return char_to_ix


def clip(gradients, maxValue):
    """
    Clips the gradients' values between minimum and maximum.

    Arguments:
    gradients -- a dictionary containing the gradients "dWaa", "dWax", "dWya", "db", "dby"
    maxValue -- everything above this number is set to this number, and everything less than -maxValue is set to -maxValue

    Returns:
    gradients -- a dictionary with the clipped gradients.
    """
    dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients[
        'dby']

    # clip to mitigate exploding gradients, loop over [dWax, dWaa, dWya, db, dby]. (≈2 lines)
    for gradient in [dWax, dWaa, dWya, db, dby]:
        np.clip(gradient, -maxValue, maxValue, out=gradient)

    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}

    return gradients


# GRADED FUNCTION: sample

def sample(parameters, char_to_ix, seed):
    """
    Sample a sequence of characters according to a sequence of probability distributions output of the RNN

    Arguments:
    parameters -- python dictionary containing the parameters Waa, Wax, Wya, by, and b.
    char_to_ix -- python dictionary mapping each character to an index.
    seed -- used for getting same random numbers using random.

    Returns:
    indices -- a list of length n containing the indices of the sampled characters.
    """

    # Retrieve parameters and relevant shapes from "parameters" dictionary
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    n_a = Waa.shape[1]

    # Step 1: Create the one-hot vector x for the first character (initializing the sequence generation). (≈1 line)
    x = np.zeros([vocab_size, 1])

    # Step 1': Initialize a_prev as zeros (≈1 line)
    a_prev = np.zeros([n_a, 1])

    # Create an empty list of indices, this is the list which will contain the list of indices of the characters to generate (≈1 line)
    indices = []
    grads = []

    # Idx is a flag to detect a newline character, we initialize it to -1
    idx = -1

    # Loop over time-steps t. At each time-step, sample a character from a probability distribution and append
    # its index to "indices". We'll stop if we reach 50 characters (which should be very unlikely with a well
    # trained model), which helps debugging and prevents entering an infinite loop.
    counter = 0
    newline_character = char_to_ix['\n']

    while idx != newline_character and counter != program_max_length:
        # Step 2: Forward propagate x using the equations (1), (2) and (3)
        a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)
        z = np.dot(Wya, a) + by
        y = softmax(z)

        np.random.seed(counter + seed)

        # Step 3: Sample the index of a character within the vocabulary from the probability distribution y
        idx = np.random.choice(range(len(y)), p=y.ravel())

        # Append the index to "indices"
        indices.append(idx)

        # Calculate likelihood ratio of policy: grad of log of softmax
        # TODO: define state in our case, for now using wya as state
        grad = softmax_likelihood_ratio(Wya, y, idx)
        grads.append(grad)

        # Step 4: Overwrite the input character as the one corresponding to the sampled index.
        x = np.zeros((vocab_size, 1))
        x[idx] = 1

        # Update "a_prev" to be "a"
        a_prev = a

        # for grading purposes
        seed += 1
        counter += 1

    if counter == program_max_length:
        indices.append(char_to_ix['\n'])

    return ''.join([ix_to_char[i] for i in indices]), grads


def init_parameters(hidden_state_size=100):
    np.random.seed(2)
    # _, n_a = 20, 100
    n_a = hidden_state_size
    Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
    b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}
    return parameters


def update_params(parameters, gradient_ascent):
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    Waa += gradient_ascent
    Wax += gradient_ascent
    Wya += gradient_ascent
    by += gradient_ascent
    b += gradient_ascent

    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}
    return parameters

# source_code = """>,[>+++++++[<------->-]<+[-<+>],]>-[+<[->+>+<<]>>[-<<+>>]<<<[->>>+>+<<<<]>>>>[-<<<<+>>>>]<[-[-[-[-[-[-[-[-[-[<<+<---------->>>[-]]]]]]]]]]]<<[->->+<<]>>[-<<+>>]<]<[>+++++++[<+++++++>-]<-.[-]]+++++++[<+++++++>-]<-."""
#
# brainfuck.evaluate(source_code)
parameters = init_parameters()
indices = sample(parameters, char_to_ix, 0)
print("Sampling:")
print("list of sampled indices:", indices)
# print("list of sampled characters:", [ix_to_char[i] for i in indices])

