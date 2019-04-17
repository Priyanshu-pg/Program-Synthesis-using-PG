import numpy as np
from program_synthesis import utils

chars = ['>', '<', '+', '-', '.', ',', '[', ']', '\n']
vocab_size = len(chars)
char_to_ix = {ch: i for i, ch in enumerate(sorted(chars))}
ix_to_char = {i: ch for i, ch in enumerate(sorted(chars))}


class lstm:
    def __init__(self, hidden_units = 100):
        self.n_x = vocab_size
        self.hidden_units = hidden_units
        self.n_y = vocab_size
        self.Wf = np.random.randn(self.hidden_units, self.n_x + self.hidden_units)
        self.bf = np.random.randn(self.hidden_units, 1)
        self.Wu = np.random.randn(self.hidden_units, self.hidden_units + self.n_x)
        self.bu = np.random.randn(self.hidden_units, 1)
        self.Wc = np.random.randn(self.hidden_units, self.hidden_units + self.n_x)
        self.bc = np.random.randn(self.hidden_units, 1)
        self.Wo = np.random.randn(self.hidden_units, self.hidden_units + self.n_x)
        self.bo = np.random.randn(self.hidden_units, 1)
        self.Wy = np.random.randn(self.n_y, self.hidden_units)
        self.by = np.random.randn(self.n_y, 1)

    def lstm_forward(self, xt, hidden_state_prev, cell_state_prev):
        # Concatenate a_prev and xt
        concat = np.zeros((self.hidden_units + self.n_x, 1))
        concat[: self.hidden_units, :] = hidden_state_prev
        concat[self.hidden_units:, :] = xt

        # forget gate
        ft = utils.sigmoid(np.dot(self.Wf, concat) + self.bf)

        # update gate
        ut = utils.sigmoid(np.dot(self.Wu, concat) + self.bu)

        # new candidate layer
        new_candidate = np.tanh(np.dot(self.Wc, concat) + self.bc)

        # next cell state
        cell_state_next = ft*cell_state_prev + ut*new_candidate

        # output
        ot = utils.sigmoid(np.dot(self.Wo, concat) + self.bo)

        # next hidden unit
        hidden_state_next = ot*np.tanh(cell_state_next)

        # prediction of LSTM cell
        yt_pred = utils.softmax(np.dot(self.Wy, hidden_state_next) + self.by)

        return hidden_state_next, cell_state_next, yt_pred

    def sample(self, seed, max_program_length=500):
        """
       Sample a sequence of characters according to a sequence of probability distributions output of the RNN

       Arguments:
       parameters -- python dictionary containing the parameters Waa, Wax, Wya, by, and b.
       max_program_length -- maximum length of a program
       seed -- used for getting same random numbers using random.

       Returns:
       indices -- a list of length n containing the indices of the sampled characters.
       grads -- to calculate gradient ascent for reinforce
       """

        # Step 1: Create the one-hot vector x for the first character (initializing the sequence generation). (≈1 line)
        x = np.zeros([vocab_size, 1])

        # Step 1: Initialize hidden_state_prev as zeros
        hidden_unit_prev = np.zeros([self.hidden_units, 1])
        cell_state_prev = np.zeros([self.hidden_units, 1])

        # Create an empty list of indices, this is the list which will contain the list of indices of the characters to generate
        indices = []

        # Create an empty list of gradients, this is the list which will contain the likelihod ratio of policy for each action i.e. generating each token
        grads = []

        # Idx is a flag to detect a newline character, we initialize it to -1
        idx = -1

        # Loop over time-steps t. At each time-step, sample a character from a probability distribution and append
        # its index to "indices". We'll stop if we reach max program limit characters (which should be very unlikely with a well
        # trained model), which helps debugging and prevents entering an infinite loop.
        counter = 0
        newline_character = char_to_ix['\n']

        while idx != newline_character and counter != max_program_length:
            # Step 2: Forward propogate x
            hidden_unit, cell_state, y = self.lstm_forward(x, hidden_unit_prev, cell_state_prev)
            np.random.seed(counter + seed)

            # Step 3: Sample the index of a character within the vocabulary from the probability distribution y
            idx = np.random.choice(range(len(y)), p=y.ravel())

            # Append the index to "indices"
            indices.append(idx)

            # Calculate likelihood ratio of policy: grad of log of softmax
            grad = utils.softmax_likelihood_ratio(x, y, idx)
            grads.append(grad)

            # Step 4: Overwrite the input character as the one corresponding to the sampled index.
            x = np.zeros((vocab_size, 1))
            x[idx] = 1

            # Update "hidden_unit_prev" to be "hidden_unit"
            hidden_unit_prev = hidden_unit

            seed += 1
            counter += 1

        if counter == max_program_length:
            indices.append(char_to_ix['\n'])

        return ''.join([ix_to_char[i] for i in indices]), grads

    def update_params(self, gradient_ascent):
        self.Wf += gradient_ascent
        self.bf += gradient_ascent
        self.Wu += gradient_ascent
        self.bu += gradient_ascent
        self.Wc += gradient_ascent
        self.bc += gradient_ascent
        self.Wo += gradient_ascent
        self.bo += gradient_ascent
        self.Wy += gradient_ascent
        self.by += gradient_ascent

