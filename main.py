from program_synthesis import RNN as rnn
from program_synthesis import agent
from program_synthesis import reward as r
from  program_synthesis import utils


LEARNING_RATE = 0.000025
GAMMA = 0.99
num_iterations = 500
char_to_ix = rnn.char_to_ind()
# TODO: Task should be passed from here





# # initialize rnn
# parameters = rnn.init_parameters()

# initialize lstm
lstm = agent.lstm()

for j in range(num_iterations):
    code_string, grads = lstm.sample(j)
    # TODO: convert string to array automatically
    expected_string = [0]
    reward = r.get_reward(code_string, expected_string)
    print(reward)
    gradient_ascent = utils.calc_gradient_ascent(grads, reward.episode_rewards, GAMMA, LEARNING_RATE)
    lstm.update_params(gradient_ascent)

    if j % 100 == 0:
        print('Iteration: %d' % (j) + '\n')
        print(lstm.sample(0))
        print('\n')

