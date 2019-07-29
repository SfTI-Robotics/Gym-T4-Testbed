import os
import numpy as np
import gym
import cv2
import random
from PIL import Image

#import matplotlib.pyplot as plt

import argparse

FRAMES_DIR = './data/frames/'
ACTIONS_DIR = './data/actions/'


def main(args):

    env_name = args.env_name
    total_episodes = args.total_episodes
    time_steps = args.time_steps
    action_refresh_rate = args.action_refresh_rate

    envs_to_generate = [env_name]

    for current_env_name in envs_to_generate:
        print("Generating data for env {}".format(current_env_name))

        env = gym.make(current_env_name) # Create the environment

        s = 0

        while s < total_episodes:

            frames_file = os.path.join(FRAMES_DIR,  'frames-%d.npy' % s) 
            actions_file = os.path.join(ACTIONS_DIR 'actions-%d.npy' % s)

            observation = env.reset()

            t = 0

            obs_sequence = []
            action_sequence = []

            while t < time_steps:  
                if t % action_refresh_rate == 0:
                    action = env.action_space.sample()
                
                # convert image to greyscale, downsize
                converted_obs = Image.fromarray(observation, 'RGB')
                converted_obs = converted_obs.convert('L')  # to gray
                converted_obs = converted_obs.resize((84, 84), Image.ANTIALIAS)
                converted_obs = np.array(converted_obs).astype('uint8')

                obs_sequence.append(converted_obs)
                action_sequence.append(action)

                observation, _, _, _ = env.step(action) # Take a random action
                t = t + 1

            print("Episode {} finished after {} timesteps".format(s, t))


            np.save(os.path.join(frames_file, obs_sequence)
            np.save(os.path.join(actions_file, action_sequence)

            s = s + 1

        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Create new training data'))
    parser.add_argument('--env_name', type=str, help='name of environment', default="Breakout-v0")
    parser.add_argument('--total_episodes', type=int, default=200,
                        help='total number of episodes to generate per worker')
    parser.add_argument('--time_steps', type=int, default=300,
                        help='how many timesteps at start of episode?')
    parser.add_argument('--render', default=0, type=int,
                        help='render the env as data is generated')
    parser.add_argument('--action_refresh_rate', default=4, type=int,
                        help='how often to change the random action, in frames')
    parser.add_argument('--run_all_envs', action='store_true',
                        help='if true, will ignore env_name and loop over all envs in train_envs variables in config.py')

    args = parser.parse_args()
    main(args)