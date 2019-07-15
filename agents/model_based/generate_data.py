import numpy as np
import gym
import cv2
import random
from skimage.transform import resize

#import matplotlib.pyplot as plt

import argparse

DIR_NAME = './data/rollout/'


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
            
            episode_id = random.randint(0, 2**31 - 1)
            filename = DIR_NAME + str(episode_id) + ".npz"

            observation = env.reset()

            t = 0

            obs_sequence = []
            action_sequence = []
            reward_sequence = []
            done_sequence = []

            reward = -0.1
            done = False

            while t < time_steps:  
                if t % action_refresh_rate == 0:
                    action = env.action_space.sample()

                resized_obs = cv2.resize(observation[40:200,:], (64,64), interpolation = cv2.INTER_NEAREST)
                normalised_obs = resized_obs/255.0
                obs_sequence.append(normalised_obs)
                action_sequence.append(action)
                reward_sequence.append(reward)
                done_sequence.append(done)

                observation, reward, done, _ = env.step(action) # Take a random action
                t = t + 1

            print("Episode {} finished after {} timesteps".format(s, t))
            
            np.savez_compressed(filename, obs=obs_sequence, action=action_sequence,
                                reward=reward_sequence, done=done_sequence)  

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