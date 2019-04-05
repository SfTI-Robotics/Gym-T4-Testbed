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

    summary_writer = tensorflow.summary.FileWriter(SAVE_PATH + '/' + model_name + '_Summary')
    print("\n ==== initialisation complete, start training ==== \n")

    # for episode in range(int(episodes)):
    for episode in range(episodes):
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
            # remove comment to watch learning process in cartpole environment
            # if (episode > 150) and is_cartpole:
            #     env.render()

            # action chooses from  simplified action space without useless keys
            action = learner.choose_action(state, episode)
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
            learner.remember(state, action, reward, next_state, done)

            step += 1
            state = next_state
            # TODO: make experience replay work for ALL environments
            # if is_cartpole:
            # train algorithm using experience replay
            learner.memory_replay()

            if done:
                # takes care of updating target model for Double_DQN
                learner.finish_episode_training()
                print('Completed Episode = ' + str(episode), ' epsilon =', "%.4f" % learner.epsilon, ', steps = ', step,
                      ", total reward = ", sum_rewards_array)

                # empty arrays after each round is complete
                # record video of environment render
                # env = make_video(env, model_filename)
                break

        # summarize plots the graph
        graph.summarize(episode, step, time.time() - start_time, sum_rewards_array, learner.epsilon,
                        learner.e_greedy_formula)

        # make gif from episode frames
        # no image data available for cartpole
        if gif and not is_cartpole and episode != 0 and episode % SAVE_GIF_FREQUENCY == 0:
            make_gif(episode, model_name, episode_frames)

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
