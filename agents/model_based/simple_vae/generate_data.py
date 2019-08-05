import os
import numpy as np
import gym
import cv2
import random
from collections import deque
from PIL import Image

#import matplotlib.pyplot as plt

import argparse

ROLLOUT_DIR = './data/rollout/'


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
            
            rollout_file = os.path.join(ROLLOUT_DIR,  'rollout-%d.npz' % s) 

            observation = env.reset()
            frame_queue = deque(maxlen=4)
            

            t = 0

            obs_sequence = []
            action_sequence = []
            next_sequence = []

            while t < time_steps:  
                if t % action_refresh_rate == 0:
                    action = env.action_space.sample()
                
                # convert image to greyscale, downsize
                converted_obs = preprocess_frame(observation)

                if t == 0:
                    for i in range(4):
                        frame_queue.append(converted_obs)
                else:
                    frame_queue.pop()
                    frame_queue.appendleft(converted_obs)
                
                stacked_state = np.stack(frame_queue, axis=2)
                obs_sequence.append(stacked_state)
                action_sequence.append(encode_action(4,action))

                observation, _, _, _ = env.step(action) # Take a random action  
                t = t + 1

                next_sequence.append(np.expand_dims(preprocess_frame(observation), axis=3))

            print("Episode {} finished after {} timesteps".format(s, t))


            np.savez_compressed(rollout_file, obs=obs_sequence, actions=action_sequence, next_frame=next_sequence)

            s = s + 1

        env.close()


def encode_action(size, action):
    action_vector = [ 0 for i in range(size) ]
    action_vector[action] = 1
    return action_vector

def preprocess_frame(frame):
    # convert image to greyscale, downsize
    converted_obs = Image.fromarray(frame, 'RGB')
    converted_obs = converted_obs.convert('L')  # to gray
    converted_obs = converted_obs.resize((84, 84), Image.ANTIALIAS)
    converted_obs = np.array(converted_obs).astype('uint8')
    return converted_obs/255.

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