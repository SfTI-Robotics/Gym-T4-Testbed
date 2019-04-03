import time
from os.path import expanduser

import gym
import numpy as np
import imageio

from LearningAlgorithms.AbstractLearningAlgorithm.AbstractBrain import AbstractLearning
from Preprocessors.Abstract_Preprocess import AbstractProcessor
from resources.gif_matplotlib import save_frames_as_gif
from temp_Graphs.summary import Summary

home = expanduser("~")

SAVE_MODEL_FREQUENCY = 100


def train(env: any, learner: AbstractLearning, graph: Summary, processor: AbstractProcessor, episodes: int,
          is_cartpole: bool, save_model=False, model_filename='', model_nr=0):
    # storing neural network weights and parameters
    # if model_nr is not None:
    #    load_model(learner, model_filename, model_nr)

    print("\n ==== initialisation complete, start training ==== \n")

    reward_episode = []

    # for episode in range(int(model_nr) + 1, int(model_nr) + int(episodes) + 1):
    for episode in range(int(episodes)):
        # storing frames as gifs, array emptied each new episode
        episode_frames = []

        observation = env.reset()
        episode_frames.append(observation)
        # Processing initial image cropping, grayscale, and stacking 4 of them
        observation = processor.preprocessing(observation, True)

        start_time = time.time()
        sum_rewards_array = 0  # total rewards for graphing

        game_step = 0  # for discounted rewards, steps for each round
        step = 0  # count total steps for each episode for the graph

        # these arrays are used to calculated and store discounted rewards
        # arrays for other variable are needed for appending to transitions in our learner to work
        # arrays emptied after every round in an episode
        reward_array = []
        if episode % 20 == 0:
            reward_episode = []
        states = []
        actions = []
        next_states = []
        dones = []

        while True:
            # remove comment to watch learning process in cartpole environment
            # if (episode > 150) and is_cartpole:
            #     env.render()

            # action chooses from  simplified action space without useless keys
            action = learner.choose_action(observation, episode)
            # actions map the simplified action space to the environment action space
            # if action space has no useless keys then action = action_mapped
            action_mapped = processor.mapping_actions_to_keys(action)
            # takes a step
            next_observation, reward, done, _ = env.step(action_mapped)

            episode_frames.append(next_observation)

            if is_cartpole:
                # punish if terminal state reached
                if done:
                    reward = -reward

            # appending <s, a, r, s', d> into arrays for storage
            states.append(observation)
            actions.append(action)  # only append the '1 out of 3' action

            reward_array.append(reward)
            sum_rewards_array += reward
            reward_episode.append(sum_rewards_array)

            next_observation = processor.preprocessing(next_observation, False)
            next_states.append(next_observation)
            dones.append(done)

            game_step += 1
            step += 1

            if done:
                # append each <s, a, r, s', d> to learner.transitions for each game round
                for i in range(game_step):
                    learner.transitions.append((states[i], actions[i], reward_array[i], next_states[i], dones[i]))
                print('Completed Episode = ' + str(episode), ' epsilon =', "%.4f" % learner.epsilon, ', steps = ', step,
                      ", total reward = ", sum_rewards_array)

                # empty arrays after each round is complete
                states, actions, reward_episode, next_states, dones = [], [], [], [], []
                # record video of environment render
                # env = make_video(env, model_filename)
                break

            observation = next_observation

            if is_cartpole:
                # train algorithm using experience replay
                learner.memory_replay(episode)

        # make gif from episode frames
        # make_gif(episode, episode_frames)

        # store model weights and parameters when episode rewards are above a certain amount
        # and after every number of episodes
        # if save_model and episode % SAVE_MODEL_FREQUENCY == 0:
        #     save_model(learner, model_filename, episode)

        # summarize plots the graph
        graph.summarize(episode, step, time.time() - start_time, sum_rewards_array, learner.epsilon,
                        learner.e_greedy_formula)

    # always save last model, even outside of normal save-steps
    # if save_model:
    #     save_model(learner, model_filename, episode)

    # killing environment to prevent memory leaks
    env.close()


# TODO: make this work (respect previous episode numbers, epsilon?)
def load_model_from_file(learner, model_filename, model_nr):
    # TODO: check if file actually exists, if not just emit warning and proceed without loading anything
    learner.network.model.load_weights(home + model_filename + str(model_nr) + '_model.h5')
    # update learner's epsilon to match episode number of loaded model
    learner.update_epsilon(int(model_nr))
    print("Loaded model " + home + model_filename + str(model_nr)
          + '_model.h5 from disk and updated epsilon to ' + str(learner.epsilon))


# TODO: make this work (respect previous episode numbers, epsilon?)
def save_model_to_file(learner, model_filename, episode):
    learner.network.model.save_weights(home + model_filename + str(episode) + '_model.h5',
                                       overwrite=True)
    print("Saved model to disk as " + home + model_filename + str(episode) + '_model.h5')


# TODO: fix function, make function call in train optional
def make_video(env, model_filename):
    return gym.wrappers.Monitor(env, directory='Videos/' + model_filename + '/',
                                video_callable=lambda episode_id: True, force=True, write_upon_reset=False)


# TODO: fix function, make function call in train optional
def make_gif(episode, episode_frames):
    if episode != 0 and episode % 5 == 0:
        images = np.array(episode_frames)
        print('gif = ', len(episode_frames))
        print('im = ', len(images))

        fname = './gifs/episode' + str(episode) + '.gif'
        with imageio.get_writer(fname, mode='I') as writer:
            for frame in images:
                writer.append_data(frame)

    if episode != 0 and episode % 5 == 0:
        fname = './gifs/episode' + str(episode) + '.gif'
        save_frames_as_gif(episode_frames, fname)
