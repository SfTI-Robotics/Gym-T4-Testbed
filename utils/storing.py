import datetime

import imageio
import tensorflow
import os
import numpy as np

from agents.image_input.AbstractBrain import AbstractLearning


def load_model_from_file(learner: AbstractLearning, model_path: str) -> None:
    """
    Loads previously saved model file to learner.network
    :param learner: learner where model will be stored
    :param model_path: path to model file
    :return:
    """
    if os.path.isfile(model_path):
        learner.network.model.load_weights(model_path)
        print('Loaded model ' + model_path + ' from disk')


# TODO: now we store all of this twice, once as a tensorboard summary (for recovery of info in case of a crash),
#  once in summary (to plot with python). Can we unify this?
def save_episode_to_summary(summary_writer: tensorflow.summary.FileWriter,
                            episode: int, step: int, time: float, reward: float, epsilon: float) -> None:
    """
    Adds summary of episode to summary file
    HINT: to view summary execute "tensorboard --logdir output/[algorithm]/tensorboard_summary
    :param summary_writer: summary writer
    :param episode: number of episode
    :param step: total steps of episode
    :param time: time needed to complete episode
    :param reward: total reward received in episode
    :param epsilon: value of epsilon at the end of episode
    """
    # create summary of episode
    summary = tensorflow.Summary()
    summary.value.add(tag='Reward', simple_value=float(reward))
    summary.value.add(tag='Step', simple_value=int(step))
    summary.value.add(tag='Time', simple_value=float(time))
    summary.value.add(tag='Epsilon', simple_value=float(epsilon))
    # add summary to file writer
    summary_writer.add_summary(summary, episode)
    summary_writer.flush()


def save_model_to_file(learner: AbstractLearning, save_path: str, environment_name: str, episode: int) -> None:
    """
    Saves current model to .h5 file, overrides previous model for same environment and algorithm
    HINT: to avoid overriding, add episode number and/or time string to model name
    :param learner: learner containing model that should be saved
    :param save_path: path to model folder
    :param environment_name: name of the environment
    :param episode: episode during which model is saved
    """
    # create folder for model, if necessary
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # save model weights
    learner.network.save_weights(save_path + '/' + environment_name + '_model_episode' + str(episode) + '_' +
                                 str(datetime.datetime.now()) + '.h5', overwrite=True)


def make_gif(episode: int, reward: int, save_path: str, episode_frames: []) -> None:
    """
    Creates gif using episode frames
    :param episode: number of episode to be stored
    :param reward: reward received in episode
    :param save_path: path to folder
    :param episode_frames: array of episode frames
    """
    # create folder for model, if necessary
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    images = np.array(episode_frames)
    print('gif = ', len(episode_frames))
    print('im = ', len(images))

    fname = save_path + 'episode' + str(episode) + '_reward' + str(reward) + '.gif'
    with imageio.get_writer(fname, mode='I') as writer:
        for frame in images:
            writer.append_data(frame)
