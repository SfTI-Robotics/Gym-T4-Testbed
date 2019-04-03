import gym
import imageio
import tensorflow
import os
import numpy as np

from LearningAlgorithms.AbstractLearningAlgorithm.AbstractBrain import AbstractLearning


# TODO: This only reloads a previously saved file
#  to continue training effectively, epsilon would have to be updated according to episode number
def load_model_from_file(learner: AbstractLearning, save_path: str) -> None:
    """
    Loads previously saved model file to learner.network
    :param learner: learner where model will be stored
    :param save_path: path to folder containing model file
    :return:
    """
    if os.path.isfile(save_path + '/' + 'model.h5'):
        learner.network.model.load_weights(save_path + '/' + 'model.h5')
        print('Loaded model ' + save_path + '/' + 'model.h5 from disk')


# TODO: now we store all of this twice, once as a tensorboard summary (for recovery of info in case of a crash),
#  once in summary (to plot with python). Can we unify this?
def save_episode_to_summary(summary_writer: tensorflow.summary.FileWriter,
                            episode: int, step: int, time: float, reward: float, epsilon: float) -> None:
    """
    Adds summary of episode to summary file
    HINT: to view summary execute "tensorboard --logdir path/to/summary_file
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


def save_model_to_file(learner: AbstractLearning, save_path: str) -> None:
    """
    Saves current model to .h5 file, overrides previous model for same environment and algorithm
    HINT: to avoid overriding, add episode number and/or time string to model name
    :param learner: learner containing model that should be saved
    :param save_path: path to model folder
    """
    # create folder for model, if necessary
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # save model weights
    learner.network.model.save_weights(save_path + '/' + 'model.h5',
                                       overwrite=True)
    print("Saved model to disk as " + save_path + '/' + 'model.h5')


# TODO: Is this needed, even when we already have the gifs?
#  if yes, fix function, make function call in train optional
#  delete otherwise
def make_video(env, model_filename):
    return gym.wrappers.Monitor(env, directory='Videos/' + model_filename + '/',
                                video_callable=lambda episode_id: True, force=True, write_upon_reset=False)


def make_gif(episode: int, model_filename: str, episode_frames: []) -> None:
    """
    Creates gif using episode frames
    :param episode: number of last episode in episode_frames
    :param model_filename: name of model file ([environment]_[algorithm])
    :param episode_frames: list of episode frames
    """
    images = np.array(episode_frames)
    print('gif = ', len(episode_frames))
    print('im = ', len(images))

    fname = './gifs/' + model_filename + 'episode' + str(episode) + '.gif'
    with imageio.get_writer(fname, mode='I') as writer:
        for frame in images:
            writer.append_data(frame)
