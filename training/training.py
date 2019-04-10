import time
from os.path import expanduser
import tensorflow

from agents import Memory
from agents.image_input.AbstractBrain import AbstractLearning
from utils.preprocessing.Abstract_Preprocess import AbstractProcessor
from utils.storing import save_model_to_file, save_episode_to_summary, make_gif, load_model_from_file
from utils.summary import Summary

home = expanduser("~")


def train(env: any, learner: AbstractLearning, memory: Memory, graph: Summary, processor: AbstractProcessor,
          config, model_name, is_cartpole) -> None:
    """
    Trains learner in env and plots results
    :param env: gym environment
    :param learner: untrained learner
    :param memory: memory for storing experiences as tuples containing state, action, reward, next_state, done
    :param graph: summary for result plotting
    :param processor: pre-processor for given environment
    :param config: configurations for training
    :param model_name: name of model as [environment]_[algorithm]
    :param is_cartpole: should be true if environment used for training is "CartPole-v1"
    """

    # loading neural network weights and parameters
    if config['load_model']:
        load_model_from_file(learner, home + config['model_save_path'] + model_name)

    summary_writer = tensorflow.summary.FileWriter(home + config['model_save_path'] + '/' + model_name + '_Summary')

    print("\n ==== initialisation complete, start training ==== \n")

# ============================================================================================================== #

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

            if is_cartpole:
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
                learner.train_network(states, actions, rewards, next_states, dones, episode, step)

            step += 1
            state = next_state

            if done:
                print('Completed Episode = ' + str(episode), ' epsilon =', "%.4f" % learner.epsilon, ', steps = ', step,
                      ", total reward = ", sum_rewards_array)
                break

# ============================================================================================================== #

        # summarize plots the graph
        graph.summarize(episode, step, time.time() - start_time, sum_rewards_array, learner.epsilon,
                        learner.e_greedy_formula)

        # make gif from episode frames
        # no image data available for cartpole
        if config['save_gif'] and not is_cartpole and episode != 0 and episode % config['gif_save_frequency'] == 0:
            make_gif(episode, model_name, episode_frames)

        if config['save_model']:
            # save episode data to tensorboard summary
            save_episode_to_summary(summary_writer, episode, step, time.time() - start_time,
                                    sum_rewards_array, learner.epsilon)

            # store model weights and parameters when episode rewards are above a certain amount
            # and after every number of episodes
            if episode % config['model_save_frequency'] == 0:
                save_model_to_file(learner, home + config['model_save_path'] + model_name)

# ============================================================================================================== #

    # killing environment to prevent memory leaks
    env.close()
