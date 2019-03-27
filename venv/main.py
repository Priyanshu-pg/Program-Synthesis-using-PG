import RNN as rnn
import reward as r

LEARNING_RATE = 0.000025
GAMMA = 0.99


def calc_gradient_ascent(grads, rewards):
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
        # Loop through everything that happend in the episode and update towards the log policy gradient times **FUTURE** reward
        sum += LEARNING_RATE * grads[i] * discounted_rewards[i]
    return sum


num_iterations = 500000
char_to_ix = rnn.char_to_ind()
task = None
#initialize rnn
parameters = rnn.init_parameters()
for j in range(num_iterations):
    code_string, grads = rnn.sample(parameters, char_to_ix, j)
    # print("gradients : ", grads)
    # code_string = "+[----->+++<]>+.---.+++++++..+++." # code for print-hello
    reward = r.get_reward(code_string)
    gradient_ascent = calc_gradient_ascent(grads, reward.episode_rewards)
    parameters = rnn.update_params(parameters, gradient_ascent)

    if j % 200 == 0:
        print("code : ", code_string)
        print(reward)
        print(reward.correct_output, reward.code_output)
        print('Iteration: %d' % (j) + '\n')
        print(rnn.sample(parameters, char_to_ix, 0))
        print('\n')

