import time
from os.path import expanduser
import tensorflow

from agents.memory import Memory
from agents.image_input.AbstractBrain import AbstractLearning
from utils.preprocessing.Abstract_Preprocess import AbstractProcessor
from utils.storing import load_model_from_file, make_gif, save_episode_to_summary, save_model_to_file
from utils.summary import Summary

home = expanduser("~")


def train(env: any, learner: AbstractLearning, memory: Memory, graph: Summary, processor: AbstractProcessor,
          config, save_path) -> None:
    """
    Trains learner in env and plots results
    :param env: gym environment
    :param learner: untrained learner
    :param memory: memory for storing experiences as tuples containing state, action, reward, next_state, done
    :param graph: summary for result plotting
    :param processor: pre-processor for given environment
    :param config: configurations for training
    :param save_path: path to folder: [home]/Gym-T4-Testbed/output/[algorithm]/
    """

    # loading neural network weights and parameters
    if config['load_model']:
        load_model_from_file(learner, home + config['model_load_path'])

    summary_writer = tensorflow.summary.FileWriter(save_path + 'tensorboard_summary/')

    print("\n ==== initialisation complete, start training ==== \n")

    # ============================================================================================================== #

    # keeping track of best episode within the last store_frequency steps
    max_reward = -1
    max_episode_number = -1
    max_episode_frames = []

    training_step = 0

    # for episode in range(int(episodes)):
    for episode in range(config['episodes']):
        # storing frames as gifs, array emptied each new episode
        episode_frames = []

        state = env.reset()

        episode_frames.append(state)
        # Processing initial image cropping, grayscale, and stacking 4 of them
        state = processor.preprocessing(state, True)

        start_time = time.time()
        sum_rewards_array = 0  # total rewards for graphing
        step = 0  # count total steps for each episode for the graph

        while True:
            # action chooses from  simplified action space without useless keys
            action = learner.choose_action(state)
            # actions map the simplified action space to the environment action space
            # if action space has no useless keys then action = action_mapped
            action_mapped = processor.mapping_actions_to_keys(action)
            # takes a step
            next_state, reward, done, _ = env.step(action_mapped)

            episode_frames.append(next_state)

            if config['environment'] == 'CartPole-v1':
                # punish if terminal state reached
                if done:
                    reward = -reward

            sum_rewards_array += reward
            next_state = processor.preprocessing(next_state, False)

            # TODO: remember action or action_mapped?
            # append <s, a, r, s', d> to learner.transitions
            memory.store_transition(state, action, reward, next_state, done)
            # train algorithm using experience replay
            if len(memory.stored_transitions) >= config['initial_exploration_steps']:
                states, actions, rewards, next_states, dones = memory.sample(config['batch_size'])
                learner.train_network(states, actions, rewards, next_states, dones, episode, training_step)

            step += 1
            state = next_state
            training_step += 1

            if done:
                print('Completed Episode = ' + str(episode), ' epsilon =', "%.4f" % learner.epsilon, ', steps = ', step,
                      ", total reward = ", sum_rewards_array)
                # update data for best episode
                if sum_rewards_array > max_reward:
                    max_reward = sum_rewards_array
                    max_episode_number = episode
                    max_episode_frames = episode_frames
                break

    # ============================================================================================================== #

        # summarize plots the graph
        if config['save_plot']:
            graph.summarize(episode, step, time.time() - start_time, sum_rewards_array, learner.epsilon,
                            learner.e_greedy_formula)

        # make gif from episode frames
        # no image data available for cartpole
        if config['save_gif'] and \
                config['environment'] != 'CartPole-v1' \
                and (episode == 0 or (episode+1) % config['gif_save_frequency'] == 0):
            make_gif(max_episode_number, max_reward, save_path + '/gifs/', max_episode_frames)
            max_reward = -1
            max_episode_number = -1
            max_episode_frames = []

        if config['save_tensorboard_summary']:
            # save episode data to tensorboard summary
            save_episode_to_summary(summary_writer, episode, step, time.time() - start_time,
                                    sum_rewards_array, learner.epsilon)

        if config['save_model'] and episode != 0 and (episode+1) % config['model_save_frequency'] == 0:
            # store model weights and parameters when episode rewards are above a certain amount
            # and after every number of episodes
            save_model_to_file(learner, save_path + '/models/', config['environment'], episode)

    # ============================================================================================================== #

    # killing environment to prevent memory leaks
    env.close()
