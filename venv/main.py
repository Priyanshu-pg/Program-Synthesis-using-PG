import RNN as rnn
import reward as r



def update_param():
    pass


def get_gradient(reward):
    return None


num_iterations = 500
char_to_ix = rnn.char_to_ind()
task = None
#initialize rnn
parameters = rnn.init_parameters()
for j in range(num_iterations):
    code_string = rnn.sample(parameters, char_to_ix, j)
    reward = r.get_reward(code_string)
    print(reward)
    gradient = get_gradient(reward)
    update_param()

    if j % 100 == 0:
        print('Iteration: %d' % (j) + '\n')
        print(rnn.sample(parameters, char_to_ix, 0))
        print('\n')

