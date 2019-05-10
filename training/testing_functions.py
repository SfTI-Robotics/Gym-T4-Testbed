import os
import numpy as np

from gym import wrappers
from time import time

from agents.image_input import AbstractBrain
from utils.summary import Summary


test_episodes = 10


def save_stuff_to_file(episode_rewards, episode_actions, predictions, test_episode, path, train_episode=None):
    base_name = 'test' + str(test_episode)
    if train_episode is not None:
        base_name = 'episode' + str(train_episode) + '_' + base_name
        path += 'episode' + str(train_episode) + '/'
        # create folder, if necessary
        if not os.path.exists(path):
            os.makedirs(path)

    np.savetxt(path + base_name + '_rewards.txt', episode_rewards, fmt='%d')
    np.savetxt(path + base_name + '_actions.txt', episode_actions, fmt='%d')
    np.savetxt(path + base_name + '_predictions.txt', predictions)


# def test(learner, env, filename, path, config, min_reward=None, max_reward=None):
def test(learner: AbstractBrain, env, config, processor, filename, path, episode=None):

    # displays video during testing (for CartPole)
    # env = wrappers.Monitor(env, path + 'videos/' + str(time()) + '/')

    # add episode number to filename
    if episode is not None:
        filename += 'episode' + str(episode)

    test_summary = Summary(['sumiz_step', 'sumiz_reward'],
                           name=filename,
                           save_path=path,
                           min_reward=processor.reward_min,
                           max_reward=processor.reward_max)

    temp_epsilon = learner.epsilon
    learner.epsilon = 0.0

    test_rewards = []
    test_steps = []

    episode_rewards = []
    episode_actions = []

    for e in range(test_episodes):
        test_state = env.reset()
        test_state = processor.process_state_for_memory(test_state, True)

        test_sum = 0
        test_step = 0
        while True:
            test_action = learner.choose_action(processor.process_state_for_network(test_state), print_predictions=True)
            test_next_state, test_reward, test_done, _ = env.step(test_action)
            test_next_state = processor.process_state_for_memory(test_next_state, False)
            test_reward = processor.process_reward(test_reward)

            if config['environment'] == 'CartPole-v1':
                # punish if terminal state reached
                if test_done:
                    test_reward = -test_reward

            test_state = test_next_state

            test_sum += test_reward
            test_step += 1
            # episode_rewards.append(test_reward)
            # episode_actions.append(test_action)

            if test_done:
                print('Test Episode = ' + str(e), ' epsilon =', "%.4f" % learner.epsilon,
                      ', steps = ', test_step,
                      ", total reward = ", test_sum)
                test_rewards.append(test_sum)
                test_steps.append(test_step)
                # save_stuff_to_file(episode_rewards, episode_actions, learner.get_predictions(),
                #                    e, path, train_episode=episode)
                # episode_rewards = []
                # episode_actions = []
                break

    # plot test-results
    # test_summary.summarize(step_counts=test_steps, reward_counts=test_rewards)
    # print('Sum Rewards = ', np.sum(test_rewards),
    #       ' Mean Reward = ', np.mean(test_rewards),
    #       ' Median Rewards = ', np.median(test_rewards),
    #       ' Standard Deviation = ', np.std(test_rewards))

    learner.epsilon = temp_epsilon
