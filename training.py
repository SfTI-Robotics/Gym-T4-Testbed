import time
from os.path import expanduser
import tensorflow

from LearningAlgorithms.AbstractLearningAlgorithm.AbstractBrain import AbstractLearning
from Preprocessors.Abstract_Preprocess import AbstractProcessor
from storing import save_model_to_file, save_episode_to_summary, make_gif, load_model_from_file
from summary import Summary

home = expanduser("~")

SAVE_MODEL_FREQUENCY = 100
SAVE_GIF_FREQUENCY = 10
SAVE_PATH = home + "/Gym-T4-Testbed/temp_Models/"


def train(env: any, learner: AbstractLearning, graph: Summary, processor: AbstractProcessor, episodes: int,
          model_name: str, is_cartpole: bool, save_model=False, load_model=False, gif=False) -> None:
    """
    Trains learner in env and plots results
    :param env: gym environment
    :param learner: untrained learner
    :param graph: summary for result plotting
    :param processor: pre-processor for given environment
    :param episodes: number of training episodes
    :param model_name: name of model as [environment]_[algorithm]
    :param is_cartpole: should be true if environment used for training is "CartPole-v1"
    :param save_model: true if model should be saved every SAVE_MODEL_FREQUENCY steps
    :param load_model: true if previous model for given algorithm and environment should be loaded before training
    :param gif: true if episodes should be saved as gifs
    """

    # loading neural network weights and parameters
    if load_model:
        load_model_from_file(learner, SAVE_PATH + model_name)

    reward_episode = []
    summary_writer = tensorflow.summary.FileWriter(SAVE_PATH + '/' + model_name + '_Summary')
    print("\n ==== initialisation complete, start training ==== \n")

    # for episode in range(int(episodes)):
    for episode in range(episodes):
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

            learner.memory_replay()
            # if is_cartpole:
            #    # train algorithm using experience replay
            #    learner.memory_replay(episode)

        # make gif from episode frames
        # no image data available for cartpole
        if gif and not is_cartpole and episode != 0 and episode % SAVE_GIF_FREQUENCY == 0:
            make_gif(episode, model_name, episode_frames)

        # summarize plots the graph
        graph.summarize(episode, step, time.time() - start_time, sum_rewards_array, learner.epsilon,
                        learner.e_greedy_formula)

        if save_model:
            # save episode data to tensorboard summary
            save_episode_to_summary(summary_writer, episode, step, time.time() - start_time,
                                    sum_rewards_array, learner.epsilon)

            # store model weights and parameters when episode rewards are above a certain amount
            # and after every number of episodes
            if episode % SAVE_MODEL_FREQUENCY == 0:
                save_model_to_file(learner, SAVE_PATH + model_name)

    # killing environment to prevent memory leaks
    env.close()
